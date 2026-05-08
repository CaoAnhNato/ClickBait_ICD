"""
Training Script cho ICD Model v5
=================================
Mô hình ICDv5 với kiến trúc Modular Pattern-Expert Framework.
Training 2 Phase:
  - Phase 1 (Expert Pretrain): Freeze backbone, train experts + router.
  - Phase 2 (Joint Training): Unfreeze backbone (top layers), train toàn bộ với combined loss + R-Drop.
Có hỗ trợ ablation: --no_router_sup, --no_router
"""

import argparse
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import mlflow
import wandb
from pathlib import Path

# Thêm đường dẫn root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.ICD.ICD_Model_v5 import ClickbaitDetectorV5
from src.ICD.dataset_icdv5 import ClickbaitDatasetV5
from src.ICD.losses_v5 import ICDv5CombinedLoss
from training.ICD.train_ICD_v3 import preprocess_text

# ===========================================================================
# Cấu hình Paths
# ===========================================================================
DATA_DIR = BASE_DIR / "data" / "processed"
ICDV5_DIR = DATA_DIR / "icdv5"
EXP_DIR = BASE_DIR / "src" / "experience" / "icdv5"

def load_icdv5_split(split: str) -> pd.DataFrame:
    parquet_path = ICDV5_DIR / f"icdv5_{split}_full.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {parquet_path}. Hãy chạy build_icdv5_dataset.py")
        
    cache_file = ICDV5_DIR / f"icdv5_{split}_segmented.parquet"
    if cache_file.exists():
        print(f"[*] Load segmented cache: {cache_file.name}")
        return pd.read_parquet(cache_file)
        
    print(f"[*] Phân đoạn VnCoreNLP cho split {split} (có thể hơi lâu ở lần đầu tiên)...")
    df = pd.read_parquet(parquet_path)
    
    def seg(text):
        if pd.isna(text) or str(text).strip() == "":
            return ""
        return preprocess_text(str(text))
        
    tqdm.pandas(desc=f"Segment {split} title")
    df["title_seg"] = df["title"].progress_apply(seg)
    
    tqdm.pandas(desc=f"Segment {split} lead")
    df["lead_seg"] = df["lead_paragraph"].progress_apply(seg)
    
    df.to_parquet(cache_file, index=False)
    print(f"    Đã lưu cache: {cache_file.name}")
    return df

# ===========================================================================
# Hàm Evaluate dùng chung
# ===========================================================================
def evaluate_v5(model, dataloader, loss_fn, device, threshold=0.5):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            to_dev = lambda k: batch[k].to(device)
            input_ids = to_dev("input_ids_news")
            attn_mask = to_dev("attention_mask_news")
            cat_id = to_dev("category_id")
            src_id = to_dev("source_id")
            pat_tags = to_dev("pattern_tags")
            labels = to_dev("label")
            soft_labels = to_dev("soft_label_llm")
            
            with autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(input_ids, attn_mask, cat_id, src_id, pat_tags, soft_labels)
                total, _ = loss_fn(
                    logits=outputs["logits"],
                    labels=labels,
                    soft_labels=soft_labels,
                    router_logits=outputs["router_logits"],
                    pattern_tags=pat_tags,
                    expert_aux_logits=outputs["expert_aux_logits"]
                )
            
            total_loss += total.item()
            probs = torch.sigmoid(outputs["logits"])
            pred_labels = (probs >= threshold).int().cpu().numpy().flatten()
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(pred_labels)
            all_labels.extend(labels.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(dataloader)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, prec, rec, f1, all_labels, all_preds, all_probs

# ===========================================================================
# Training Loop
# ===========================================================================
def train_phase(model, train_loader, val_loader, loss_fn, optimizer, scheduler, scaler, device, args, phase, best_f1=0.0):
    patience_counter = 0
    epochs = args.phase1_epochs if phase == 1 else args.phase2_epochs
    grad_acc = args.grad_accumulation
    use_amp = args.use_amp
    
    checkpoint_dir = EXP_DIR / args.experiment_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"best_model_phase{phase}.pth"
    
    print(f"\n{'='*50}\nBắt đầu Phase {phase} ({epochs} epochs)\n{'='*50}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Phase {phase} - Epoch {epoch}/{epochs}")
        
        for step, batch in enumerate(progress_bar):
            to_dev = lambda k: batch[k].to(device)
            input_ids = to_dev("input_ids_news")
            attn_mask = to_dev("attention_mask_news")
            cat_id = to_dev("category_id")
            src_id = to_dev("source_id")
            pat_tags = to_dev("pattern_tags")
            labels = to_dev("label")
            soft_labels = to_dev("soft_label_llm")
            
            with autocast("cuda", enabled=use_amp):
                # Pass 1
                outputs = model(input_ids, attn_mask, cat_id, src_id, pat_tags, soft_labels)
                logits2 = None
                
                # R-Drop: pass 2 (only if rdrop > 0, usually in phase 2)
                if args.rdrop_alpha > 0 and phase == 2:
                    outputs2 = model(input_ids, attn_mask, cat_id, src_id, pat_tags, soft_labels)
                    logits2 = outputs2["logits"]
                    
                loss, loss_dict = loss_fn(
                    logits=outputs["logits"],
                    labels=labels,
                    soft_labels=soft_labels,
                    router_logits=outputs["router_logits"],
                    pattern_tags=pat_tags,
                    expert_aux_logits=outputs["expert_aux_logits"],
                    logits2=logits2
                )
                loss = loss / grad_acc
                
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            total_train_loss += loss.item() * grad_acc
            
            if (step + 1) % grad_acc == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            progress_bar.set_postfix({"Loss": f"{loss.item() * grad_acc:.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        current_lr = optimizer.param_groups[-1]['lr']
        
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluate_v5(
            model, val_loader, loss_fn, device, args.threshold
        )
        
        print(f"\nPhase {phase} - Epoch {epoch}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        print(f"  Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        
        metrics_dict = {
            f"p{phase}/train_loss": avg_train_loss,
            f"p{phase}/lr": current_lr,
            f"p{phase}/val_loss": val_loss,
            f"p{phase}/val_f1": val_f1,
            f"p{phase}/val_acc": val_acc
        }
        mlflow.log_metrics(metrics_dict, step=epoch)
        if not args.no_wandb:
            wandb.log(metrics_dict, step=epoch)
            
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            print(f"  [!] New best F1 for phase {phase}: {best_f1:.4f}. Saving...")
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_metric(f"best_val_f1_p{phase}", best_f1, step=epoch)
        else:
            patience_counter += 1
            print(f"  [EarlyStopping] No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n[!] Early stopping ở epoch {epoch}. Best F1: {best_f1:.4f}")
                break
                
    return best_f1, best_model_path


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    # HW Profile & Base configs
    parser.add_argument('--hw_profile', type=str, choices=['rtx3050', 'ada5000', 'rtxa4000'], default='rtx3050')
    parser.add_argument('--experiment_name', type=str, default="ICDv5_8FL")
    parser.add_argument('--run_name', type=str, default="run_1")
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    
    # Model configs
    parser.add_argument('--model_name', type=str, default="vinai/phobert-base-v2")
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--num_categories', type=int, default=50)
    parser.add_argument('--num_sources', type=int, default=200)
    
    # Ablation
    parser.add_argument('--no_router_sup', action='store_true', help="Ablation: turn off router loss (lambda_router=0)")
    parser.add_argument('--no_router', action='store_true', help="Ablation: remove router, avg experts")
    
    # Training Params
    parser.add_argument('--phase1_epochs', type=int, default=10)
    parser.add_argument('--phase2_epochs', type=int, default=20)
    parser.add_argument('--skip_phase1', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--freeze_layers', type=int, default=8, help="Freeze layers in Phase 2")
    
    parser.add_argument('--lr_p1', type=float, default=1e-3, help="Learning rate for Phase 1")
    parser.add_argument('--lr_p2', type=float, default=5e-5, help="Learning rate for Phase 2")
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    
    # Losses Params
    parser.add_argument('--focal_alpha', type=float, default=0.65)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--lambda_kl', type=float, default=0.5)
    parser.add_argument('--lambda_expert', type=float, default=1.0)
    parser.add_argument('--lambda_router', type=float, default=0.3)
    parser.add_argument('--rdrop_alpha', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # HW adjustments
    if args.hw_profile == 'rtx3050':
        args.batch_size = 2
        args.grad_accumulation = 16
        args.num_workers = 4
        args.use_amp = True
    elif args.hw_profile == 'ada5000':
        args.batch_size = 32
        args.grad_accumulation = 1
        args.num_workers = 16
        args.use_amp = True
    else:
        args.batch_size = 16
        args.grad_accumulation = 2
        args.num_workers = 8
        args.use_amp = True
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device} | HW: {args.hw_profile} | Batch: {args.batch_size} | GradAcc: {args.grad_accumulation}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    df_train = load_icdv5_split("train")
    df_val = load_icdv5_split("valid")
    df_test = load_icdv5_split("test")
    # Chuẩn hóa các cột
    for df in [df_train, df_val, df_test]:
        if "label" in df.columns and "label_bin" not in df.columns:
            df["label_bin"] = df["label"]
        if "score_init" in df.columns and "soft_label_llm" not in df.columns:
            df["soft_label_llm"] = df["score_init"] / 100.0
            
    # Map category và source sang ID
    all_cats = pd.concat([df_train["category"], df_val["category"], df_test["category"]]).unique()
    all_srcs = pd.concat([df_train["source"], df_val["source"], df_test["source"]]).unique()
    
    cat2id = {c: i for i, c in enumerate(all_cats)}
    src2id = {s: i for i, s in enumerate(all_srcs)}
    
    for df in [df_train, df_val, df_test]:
        df["category_id"] = df["category"].map(cat2id).fillna(0).astype(int)
        df["source_id"] = df["source"].map(src2id).fillna(0).astype(int)
        
    args.num_categories = len(cat2id) + 1
    args.num_sources = len(src2id) + 1
    
    train_ds = ClickbaitDatasetV5(df_train, tokenizer, args.max_len)
    val_ds = ClickbaitDatasetV5(df_val, tokenizer, args.max_len)
    test_ds = ClickbaitDatasetV5(df_test, tokenizer, args.max_len)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"[*] Data size: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")
    
    # Model
    model = ClickbaitDetectorV5(
        model_name_or_path=args.model_name,
        num_categories=args.num_categories,
        num_sources=args.num_sources,
        use_router=not args.no_router
    ).to(device)
    
    # Ablation adjustments
    if args.no_router_sup or args.no_router:
        args.lambda_router = 0.0
        
    if not args.no_wandb:
        wandb.init(project="ICDv5", name=f"{args.experiment_name}_{args.run_name}", config=vars(args))
        
    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(run_name=args.run_name)
    mlflow.log_params(vars(args))
    
    scaler = GradScaler("cuda") if args.use_amp else None
    
    best_model_p1 = None
    best_model_p2 = None
    
    # ---------------------------------------------------------
    # PHASE 1
    # ---------------------------------------------------------
    if not args.skip_phase1 and args.phase1_epochs > 0:
        print("[*] Setting up Phase 1...")
        model.freeze_backbone_layers(12) # freeze all PhoBERT
        
        loss_fn_p1 = ICDv5CombinedLoss(
            alpha_focal=0.0, # Disable classification loss
            lambda_kl=0.0,   # Disable LLM KD
            lambda_router=args.lambda_router,
            lambda_expert=args.lambda_expert,
            beta_rdrop=0.0
        )
        
        # Optimizer for Phase 1
        p1_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_p1 = torch.optim.AdamW(p1_params, lr=args.lr_p1, weight_decay=0.01)
        
        total_steps_p1 = (len(train_loader) // args.grad_accumulation) * args.phase1_epochs
        warmup_steps_p1 = int(total_steps_p1 * args.warmup_ratio)
        scheduler_p1 = get_cosine_schedule_with_warmup(optimizer_p1, warmup_steps_p1, total_steps_p1)
        
        _, best_model_p1 = train_phase(
            model, train_loader, val_loader, loss_fn_p1, optimizer_p1, scheduler_p1, scaler, device, args, phase=1
        )
        
        if best_model_p1 and os.path.exists(best_model_p1):
            model.load_state_dict(torch.load(best_model_p1, weights_only=True))
            print("[*] Loaded best Phase 1 model for Phase 2")
            
    # ---------------------------------------------------------
    # PHASE 2
    # ---------------------------------------------------------
    if args.phase2_epochs > 0:
        print("\n[*] Setting up Phase 2...")
        model.freeze_backbone_layers(args.freeze_layers) # Unfreeze top layers
        
        loss_fn_p2 = ICDv5CombinedLoss(
            alpha_focal=args.focal_alpha,
            gamma_focal=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            lambda_kl=args.lambda_kl,
            lambda_router=args.lambda_router,
            lambda_expert=args.lambda_expert,
            beta_rdrop=args.rdrop_alpha
        )
        
        param_groups_p2 = model.get_parameter_groups(base_lr=args.lr_p2, lr_decay=args.lr_decay)
        optimizer_p2 = torch.optim.AdamW(param_groups_p2)
        
        total_steps_p2 = (len(train_loader) // args.grad_accumulation) * args.phase2_epochs
        warmup_steps_p2 = int(total_steps_p2 * args.warmup_ratio)
        scheduler_p2 = get_cosine_schedule_with_warmup(optimizer_p2, warmup_steps_p2, total_steps_p2)
        
        _, best_model_p2 = train_phase(
            model, train_loader, val_loader, loss_fn_p2, optimizer_p2, scheduler_p2, scaler, device, args, phase=2
        )
        
    # ---------------------------------------------------------
    # TEST EVALUATION
    # ---------------------------------------------------------
    best_model_path = best_model_p2 if best_model_p2 else best_model_p1
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        
    loss_fn_eval = ICDv5CombinedLoss(alpha_focal=args.focal_alpha, lambda_kl=args.lambda_kl, lambda_router=args.lambda_router, lambda_expert=args.lambda_expert, beta_rdrop=0.0)
    
    print("\n" + "="*50)
    print("Testing Best Model...")
    test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds, test_probs = evaluate_v5(
        model, test_loader, loss_fn_eval, device, args.threshold
    )
    
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
    
    report = classification_report(test_labels, test_preds, target_names=["non-clickbait", "clickbait"])
    print("\nClassification Report:\n", report)
    
    metrics = {
        "test/loss": test_loss,
        "test/accuracy": test_acc,
        "test/precision": test_prec,
        "test/recall": test_rec,
        "test/f1": test_f1
    }
    mlflow.log_metrics(metrics)
    if not args.no_wandb:
        wandb.log(metrics)
        wandb.finish()
        
    mlflow.end_run()
    
    # Save outputs
    res_dir = EXP_DIR / args.experiment_name / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(res_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k.replace("test/", ""): v for k, v in metrics.items()}, f, indent=4)
        
    df_test_out = df_test.copy()
    if len(df_test_out) == len(test_preds):
        df_test_out["label"] = test_labels
        df_test_out["prob"] = test_probs
        df_test_out["pred"] = test_preds
        df_test_out.to_parquet(res_dir / "test_predictions.parquet", index=False)
        print(f"Saved predictions to {res_dir / 'test_predictions.parquet'}")
        
if __name__ == "__main__":
    main()
