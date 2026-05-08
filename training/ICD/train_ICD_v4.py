"""
Training Script cho ICD Model v4 – LLM-Assisted Reasoning (Pha 6)
===================================================================
Kế thừa cấu trúc train_ICD_v3.py, mở rộng với:
- ClickbaitReasoningDataset: 5 loại tokenization per sample
- ClickbaitDetectorV4: shared backbone + 4 reasoning encoders
- ICDv4CombinedLoss: Focal + KL + Contrastive + R-Drop
- Pre-tokenized cache để giảm bottleneck
- MLflow + WandB tracking

Chạy:
    conda run -n MLE python training/ICD/train_ICD_v4.py --hw_profile rtx3050
    conda run -n MLE python training/ICD/train_ICD_v4.py --hw_profile rtx3050 --epochs 30 --dry_run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import py_vncorenlp

# Path setup
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.ICD.ICD_Model_v4 import ClickbaitDetectorV4
from src.ICD.losses import ICDv4CombinedLoss
from training.ICD.train_ICD_v3 import (
    extract_aux_features,
    load_and_preprocess_data,
    preprocess_text,
)

# Thử import build logic để tự động hóa
try:
    from data.build_icdv4_dataset import build_dataset
except ImportError:
    build_dataset = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = BASE_DIR / "data" / "processed"
ICDV4_PKL   = DATA_DIR / "icdv4" / "Cleaned_Clickbait_with_reasoning.parquet"
CLEANED_DIR = DATA_DIR / "cleaned"

# Thư mục gốc cho kinh nghiệm (checkpoints và kết quả)
BASE_EXP_DIR = BASE_DIR / "src" / "experience" / "icdv4"


# ===========================================================================
# Dataset
# ===========================================================================
class ClickbaitReasoningDataset(Dataset):
    """
    Dataset cho ICDv4: mỗi sample trả về 5 tensor inputs:
      1. (title_seg, lead_seg) – tokenized sentence pair
      2. agree_reason_seg      – tokenized TF-agree reasoning
      3. disagree_reason_seg   – tokenized TF-disagree reasoning
      4. (title_seg, agree_reason_seg) – tokenized TA-agree pair
      5. (title_seg, disagree_reason_seg) – tokenized TA-disagree pair
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        tokenizer,
        max_len_news: int = 256,
        max_len_reason: int = 128,
    ):
        self.df = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len_news   = max_len_news
        self.max_len_reason = max_len_reason

        # Pre-check required columns
        required = ["title_seg", "lead_seg", "agree_reason_seg",
                    "disagree_reason_seg", "label_bin", "p_llm_final",
                    "p_llm_agree", "p_llm_disagree"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Thiếu column: {col}")

    def __len__(self):
        return len(self.df)

    def _encode(self, text_a: str, text_b: str = None, max_len: int = 256) -> dict:
        """Tokenize 1 hoặc 2 chuỗi."""
        if text_b is None:
            enc = self.tokenizer(
                text_a,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                text_a, text_b,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0).float(),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        title_seg    = str(row["title_seg"])
        lead_seg     = str(row["lead_seg"])
        agree_seg    = str(row["agree_reason_seg"])
        disagree_seg = str(row["disagree_reason_seg"])

        # 1. News pair (title, lead)
        enc_news = self._encode(title_seg, lead_seg, max_len=self.max_len_news)

        # 2. TF-Agree: chỉ reasoning text
        enc_agree_tf = self._encode(agree_seg, max_len=self.max_len_reason)

        # 3. TF-Disagree: chỉ reasoning text
        enc_disagree_tf = self._encode(disagree_seg, max_len=self.max_len_reason)

        # 4. TA-Agree: (title, agree_reason) pair
        enc_agree_ta = self._encode(title_seg, agree_seg, max_len=self.max_len_reason)

        # 5. TA-Disagree: (title, disagree_reason) pair
        enc_disagree_ta = self._encode(title_seg, disagree_seg, max_len=self.max_len_reason)

        # Auxiliary features (từ raw text)
        raw_title = str(row.get("title", ""))
        raw_lead  = str(row.get("lead_paragraph", row.get("lead", "")))
        aux_feats = extract_aux_features(raw_title, raw_lead)

        return {
            # News
            "input_ids":             enc_news["input_ids"],
            "attention_mask":        enc_news["attention_mask"],
            "aux_features":          torch.tensor(aux_feats, dtype=torch.float32),
            # TF-Agree
            "input_ids_agree":       enc_agree_tf["input_ids"],
            "attention_mask_agree":  enc_agree_tf["attention_mask"],
            # TF-Disagree
            "input_ids_disagree":    enc_disagree_tf["input_ids"],
            "attention_mask_disagree": enc_disagree_tf["attention_mask"],
            # TA-Agree
            "input_ids_ta_agree":    enc_agree_ta["input_ids"],
            "attention_mask_ta_agree": enc_agree_ta["attention_mask"],
            # TA-Disagree
            "input_ids_ta_disagree": enc_disagree_ta["input_ids"],
            "attention_mask_ta_disagree": enc_disagree_ta["attention_mask"],
            # Labels & soft labels
            "labels":       torch.tensor([float(row["label_bin"])], dtype=torch.float32),
            "p_llm_final":  torch.tensor([float(row["p_llm_final"])], dtype=torch.float32),
            "agree_score":  torch.tensor([float(row["p_llm_agree"])], dtype=torch.float32),
            "disagree_score": torch.tensor([float(row["p_llm_disagree"])], dtype=torch.float32),
        }


# ===========================================================================
# Data Loading
# ===========================================================================
# Note: VnCoreNLP segmenter is lazy-loaded via preprocess_text from v3


def load_icdv4_split(split: str, parquet_path: Path) -> pd.DataFrame:
    """
    Load split từ parquet ICDv4 và apply VnCoreNLP segmentation lên
    tất cả text fields (title, lead, agree_reason, disagree_reason).
    Cache segmented result để không redo.
    """
    cache_file = parquet_path.parent / f"{split}_segmented.parquet"

    if cache_file.exists():
        print(f"[*] Load segmented cache: {cache_file.name}")
        return pd.read_parquet(str(cache_file))

    print(f"[*] Load & segment {split} split từ parquet...")
    df_all = pd.read_parquet(str(parquet_path))
    df = df_all[df_all["split"] == split].reset_index(drop=True)
    print(f"    {split}: {len(df)} rows")

    def seg(text):
        if pd.isna(text) or str(text).strip() == "":
            return ""
        return preprocess_text(str(text))

    tqdm.pandas(desc=f"Segment {split} title")
    df["title_seg"] = df["title"].progress_apply(seg)

    tqdm.pandas(desc=f"Segment {split} lead")
    df["lead_seg"] = df["lead_paragraph"].progress_apply(seg)

    tqdm.pandas(desc=f"Segment {split} agree_reason")
    df["agree_reason_seg"] = df["agree_reason"].progress_apply(seg)

    tqdm.pandas(desc=f"Segment {split} disagree_reason")
    df["disagree_reason_seg"] = df["disagree_reason"].progress_apply(seg)

    df.to_parquet(str(cache_file), index=False)
    print(f"    Đã lưu cache: {cache_file.name}")
    return df


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate_v4(model, dataloader, loss_fn, device,
                threshold: float = 0.5) -> tuple:
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            to_dev = lambda x: x.to(device)

            logits, z_T, z_A, z_D = model(
                to_dev(batch["input_ids"]),
                to_dev(batch["attention_mask"]),
                to_dev(batch["aux_features"]),
                to_dev(batch["input_ids_agree"]),
                to_dev(batch["attention_mask_agree"]),
                to_dev(batch["input_ids_disagree"]),
                to_dev(batch["attention_mask_disagree"]),
                to_dev(batch["input_ids_ta_agree"]),
                to_dev(batch["attention_mask_ta_agree"]),
                to_dev(batch["input_ids_ta_disagree"]),
                to_dev(batch["attention_mask_ta_disagree"]),
            )

            labels     = to_dev(batch["labels"])
            p_llm      = to_dev(batch["p_llm_final"])
            v_agree    = to_dev(batch["agree_score"])
            v_disagree = to_dev(batch["disagree_score"])

            losses = loss_fn(logits, labels, p_llm, z_T, z_A, z_D,
                             v_agree, v_disagree)
            total_loss += losses["total"].item()

            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= threshold).int()
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(-1).cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, prec, rec, f1, all_labels, all_probs, all_preds


# ===========================================================================
# Main Training Loop
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Train ICD Model v4 – LLM-Assisted Reasoning")
    parser.add_argument("-hw", "--hw_profile", type=str,
                        choices=["rtx3050", "ada5000", "rtxa4000"], default="rtx3050")
    parser.add_argument("-e",  "--epochs", type=int, default=30)
    parser.add_argument("-lr", "--lr", type=float, default=3e-5)
    parser.add_argument("-ld", "--lr_decay", type=float, default=0.98)
    parser.add_argument("-wr", "--warmup_ratio", type=float, default=0.15)
    parser.add_argument("-fa", "--focal_alpha", type=float, default=0.65)
    parser.add_argument("-fg", "--focal_gamma", type=float, default=2.0)
    parser.add_argument("-fl", "--freeze_layers", type=int, default=8)
    parser.add_argument("-t",  "--threshold", type=float, default=0.5)
    parser.add_argument("-ml", "--max_len_news", type=int, default=256)
    parser.add_argument("-mlr","--max_len_reason", type=int, default=128)
    parser.add_argument("-p",  "--patience", type=int, default=7)
    parser.add_argument("-ra", "--rdrop_alpha", type=float, default=1.0)
    parser.add_argument("-lk", "--lambda_kl", type=float, default=0.5)
    parser.add_argument("-ac", "--alpha_contrastive", type=float, default=0.5)
    parser.add_argument("-cm", "--contrastive_margin", type=float, default=0.3)
    parser.add_argument("-en", "--experiment_name", type=str, default="ICDv4-ReasoningClickbait")
    parser.add_argument("-rn", "--run_name", type=str, default=None)
    parser.add_argument("-nw", "--no_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Chạy 1 batch để verify pipeline")
    args = parser.parse_args()

    # Hardware profiles
    if args.hw_profile == "rtx3050":
        BATCH_SIZE = 2
        GRAD_ACC   = 16
        NUM_WORKERS = 2
        USE_AMP = True
        print("[*] HW: RTX 3050 (4GB). Batch=2, Acc=16, AMP=True")
    elif args.hw_profile == "ada5000":
        BATCH_SIZE = 16
        GRAD_ACC   = 2
        NUM_WORKERS = 8
        USE_AMP = True
        print("[*] HW: ADA 5000 (16GB). Batch=16, Acc=2, AMP=True")
    else:
        BATCH_SIZE = 8
        GRAD_ACC   = 4
        NUM_WORKERS = 4
        USE_AMP = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # Thiết lập tên Run và thư mục lưu trữ riêng biệt
    timestamp = pd.Timestamp.now().strftime("%m%d_%H%M")
    run_name = args.run_name or f"icdv4-rtx3050-{timestamp}"

    # Folder riêng cho run này để tránh ghi đè kết quả cũ
    RUN_CKPT_DIR = BASE_EXP_DIR / "checkpoints" / run_name
    RUN_RESULTS_DIR = BASE_EXP_DIR / "results" / run_name
    RUN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_ckpt = RUN_CKPT_DIR / "icdv4_best.pt"

    # Load tokenizer
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load & segment data
    print("[*] Loading ICDv4 dataset với reasoning...")
    if not ICDV4_PKL.exists():
        print(f"[*] Parquet không tồn tại: {ICDV4_PKL}")
        if build_dataset:
            print("[*] Đang tự động khởi tạo ICDv4 dataset (Merge CSV + Reasoning)...")
            build_dataset()
        else:
            print("[ERROR] Không tìm thấy logic build dataset. Hãy chạy 'python data/build_icdv4_dataset.py' thủ công.")
            sys.exit(1)

    df_train = load_icdv4_split("train", ICDV4_PKL)
    df_val   = load_icdv4_split("validate", ICDV4_PKL)
    df_test  = load_icdv4_split("test", ICDV4_PKL)

    # Build datasets
    train_ds = ClickbaitReasoningDataset(df_train, tokenizer,
                                         args.max_len_news, args.max_len_reason)
    val_ds   = ClickbaitReasoningDataset(df_val, tokenizer,
                                          args.max_len_news, args.max_len_reason)
    test_ds  = ClickbaitReasoningDataset(df_test, tokenizer,
                                          args.max_len_news, args.max_len_reason)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"[*] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    # Build model
    model = ClickbaitDetectorV4(
        model_name=model_name,
        sep_token_id=tokenizer.sep_token_id,
        dropout_rate=0.3,
    )
    model.freeze_backbone_layers(args.freeze_layers)
    model.to(device)

    # Loss
    loss_fn = ICDv4CombinedLoss(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        lambda_kl=args.lambda_kl,
        alpha_contrastive=args.alpha_contrastive,
        beta_rdrop=args.rdrop_alpha,
        contrastive_margin=args.contrastive_margin,
    )

    # Optimizer với layer-wise LR decay
    param_groups = model.get_parameter_groups(lr=args.lr, lr_decay=args.lr_decay)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Scheduler
    total_steps = (len(train_loader) // GRAD_ACC) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler("cuda") if USE_AMP else None

    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN] Chạy 1 batch để verify...")
        batch = next(iter(train_loader))
        to_dev = lambda x: x.to(device)
        with autocast("cuda", enabled=USE_AMP):
            logits, z_T, z_A, z_D = model(
                to_dev(batch["input_ids"]), to_dev(batch["attention_mask"]),
                to_dev(batch["aux_features"]),
                to_dev(batch["input_ids_agree"]), to_dev(batch["attention_mask_agree"]),
                to_dev(batch["input_ids_disagree"]), to_dev(batch["attention_mask_disagree"]),
                to_dev(batch["input_ids_ta_agree"]), to_dev(batch["attention_mask_ta_agree"]),
                to_dev(batch["input_ids_ta_disagree"]), to_dev(batch["attention_mask_ta_disagree"]),
            )
            losses = loss_fn(logits, to_dev(batch["labels"]), to_dev(batch["p_llm_final"]),
                             z_T, z_A, z_D, to_dev(batch["agree_score"]),
                             to_dev(batch["disagree_score"]))
        print(f"  logits: {logits.shape}, z_T: {z_T.shape}, z_A: {z_A.shape}, z_D: {z_D.shape}")
        print(f"  losses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")
        print("[✓] Dry run thành công!")
        return

    # WandB - Sử dụng experiment_name làm Project Name trên giao diện WandB
    if not args.no_wandb:
        wandb.init(project=args.experiment_name, name=run_name, config=vars(args))

    # MLflow
    mlflow.set_experiment(args.experiment_name)
    mlflow_run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(vars(args))

    # Training
    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")):
            to_dev = lambda x: x.to(device)

            with autocast("cuda", enabled=USE_AMP):
                # Pass 1 (for R-Drop, need 2 passes if rdrop_alpha > 0)
                logits, z_T, z_A, z_D = model(
                    to_dev(batch["input_ids"]),     to_dev(batch["attention_mask"]),
                    to_dev(batch["aux_features"]),
                    to_dev(batch["input_ids_agree"]),    to_dev(batch["attention_mask_agree"]),
                    to_dev(batch["input_ids_disagree"]), to_dev(batch["attention_mask_disagree"]),
                    to_dev(batch["input_ids_ta_agree"]),    to_dev(batch["attention_mask_ta_agree"]),
                    to_dev(batch["input_ids_ta_disagree"]), to_dev(batch["attention_mask_ta_disagree"]),
                )

                # Pass 2 for R-Drop
                logits2 = None
                if args.rdrop_alpha > 0:
                    logits2, _, _, _ = model(
                        to_dev(batch["input_ids"]),     to_dev(batch["attention_mask"]),
                        to_dev(batch["aux_features"]),
                        to_dev(batch["input_ids_agree"]),    to_dev(batch["attention_mask_agree"]),
                        to_dev(batch["input_ids_disagree"]), to_dev(batch["attention_mask_disagree"]),
                        to_dev(batch["input_ids_ta_agree"]),    to_dev(batch["attention_mask_ta_agree"]),
                        to_dev(batch["input_ids_ta_disagree"]), to_dev(batch["attention_mask_ta_disagree"]),
                    )

                losses = loss_fn(
                    logits, to_dev(batch["labels"]),
                    to_dev(batch["p_llm_final"]),
                    z_T, z_A, z_D,
                    to_dev(batch["agree_score"]),
                    to_dev(batch["disagree_score"]),
                    logits2=logits2,
                )
                loss = losses["total"] / GRAD_ACC

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_train_loss += losses["total"].item()

            if (step + 1) % GRAD_ACC == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluate_v4(
            model, val_loader, loss_fn, device, args.threshold
        )

        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
              f"P: {val_prec:.4f} | R: {val_rec:.4f}")

        # Logging
        metrics_ep = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
        }
        history.append(metrics_ep)
        mlflow.log_metrics({k: v for k, v in metrics_ep.items() if k != "epoch"}, step=epoch)
        if not args.no_wandb:
            wandb.log(metrics_ep, step=epoch)

        # Early stopping & checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), str(best_ckpt))
            print(f"  [✓] New best F1={val_f1:.4f} → saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[!] Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final test evaluation
    print("\n[*] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(str(best_ckpt), map_location=device, weights_only=True))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_probs, test_preds = \
        evaluate_v4(model, test_loader, loss_fn, device, args.threshold)

    print("\n" + "="*55)
    print("ICDv4 – Final Test Results")
    print("="*55)
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(classification_report(test_labels, test_preds,
                                target_names=["Non-Clickbait", "Clickbait"]))

    # Save test metrics
    test_metrics = {
        "model": "ICDv4",
        "checkpoint": str(best_ckpt),
        "threshold": args.threshold,
        "test_loss": test_loss,
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1": test_f1,
    }
    metrics_path = RUN_RESULTS_DIR / "test_metrics_full.json"
    with open(str(metrics_path), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Test metrics: {metrics_path}")

    # Save predictions
    test_ids = df_test["id"].tolist()[:len(test_labels)]
    pred_df = pd.DataFrame({
        "id":    test_ids,
        "label": [int(l) for l in test_labels],
        "prob":  test_probs,
        "pred":  [int(p) for p in test_preds],
    })
    pred_path = RUN_RESULTS_DIR / "test_predictions_full.parquet"
    pred_df.to_parquet(str(pred_path), index=False)
    print(f"[+] Predictions: {pred_path}")

    # Log to MLflow
    mlflow.log_metrics({
        "test_loss": test_loss, "test_acc": test_acc,
        "test_precision": test_prec, "test_recall": test_rec,
        "test_f1": test_f1,
    })
    mlflow.log_artifact(str(best_ckpt))
    mlflow.end_run()

    if not args.no_wandb:
        # Upload best model weight lên WandB Artifacts
        print(f"[*] Đang đẩy best model ({best_ckpt.name}) lên WandB Artifacts...")
        artifact = wandb.Artifact(name=f"model-{run_name}", type="model",
                                 description=f"Best ICDv4 model (Test F1={test_f1:.4f})")
        artifact.add_file(str(best_ckpt))
        wandb.log_artifact(artifact)

        wandb.log({"test_f1": test_f1, "test_precision": test_prec, "test_recall": test_rec})
        wandb.finish()

    print("\n[✓] Training ICDv4 hoàn thành!")


if __name__ == "__main__":
    main()
