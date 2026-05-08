"""
ICDv4 – Eval Baseline ICDv3 (Pha 5)
======================================
Đánh giá chi tiết ICDv3 baseline trên cleaned test set.
Sinh kết quả chuẩn để so sánh với ICDv4.

Output:
  src/experience/icdv3/results/test_metrics.json
  src/experience/icdv3/results/test_predictions.parquet
  src/experience/icdv3/results/calibration_curve.png

Chạy:
    conda run -n MLE python training/ICD/eval_baseline_v3.py
    conda run -n MLE python training/ICD/eval_baseline_v3.py --checkpoint path/to/model.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.ICD.ICD_Model_v3 import ClickbaitDetectorV3_1, FocalLossWithSmoothing
from training.ICD.train_ICD_v3 import ClickbaitPairDataset, load_and_preprocess_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = BASE_DIR / "data" / "processed" / "cleaned"
RESULTS_DIR = BASE_DIR / "src" / "experience" / "icdv3" / "results"
CKPT_DIR    = BASE_DIR / "src" / "experience" / "icdv3" / "checkpoints"
DEFAULT_CKPT = BASE_DIR / "result" / "ICD" / "checkpoints" / "best_model_v3.pth"


# ---------------------------------------------------------------------------
# Evaluation with probability output
# ---------------------------------------------------------------------------
def evaluate_with_probs(model, dataloader, loss_fn, device, threshold=0.5):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating v3 baseline", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features   = batch["aux_features"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, aux_features)
            loss   = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= threshold).int()

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(-1).cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    brier = brier_score_loss(all_labels, all_probs)

    return avg_loss, acc, precision, recall, f1, brier, all_labels, all_probs, all_preds


def plot_calibration_curve(labels, probs, output_path: Path, model_name="ICDv3"):
    """Vẽ reliability diagram (calibration curve)."""
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=1.5)
    ax.plot(prob_pred, prob_true, "o-", color="#4C9BE8", label=model_name, linewidth=2)
    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title(f"Calibration Curve – {model_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return prob_true, prob_pred


def plot_confusion_matrix(labels, preds, output_path: Path, model_name="ICDv3"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-CB", "Clickbait"],
                yticklabels=["Non-CB", "Clickbait"], ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=12)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ICDv4 – Eval baseline ICDv3")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_alpha", type=float, default=0.65)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # Load tokenizer
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sep_token_id = tokenizer.sep_token_id

    # Load test data
    print("[*] Loading & preprocessing test data (VnCoreNLP)...")
    test_titles, test_leads, test_labels, test_raw_t, test_raw_l = \
        load_and_preprocess_data(str(DATA_DIR / "test_clean.csv"))

    test_dataset = ClickbaitPairDataset(
        test_titles, test_leads, test_labels, test_raw_t, test_raw_l,
        tokenizer, max_len=args.max_len
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # Load model
    print(f"[*] Loading ICDv3 from: {args.checkpoint}")
    model = ClickbaitDetectorV3_1(
        model_name=model_name,
        sep_token_id=sep_token_id,
        dropout_rate=0.3
    )
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint không tồn tại: {ckpt_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    loss_fn = FocalLossWithSmoothing(
        alpha=args.focal_alpha, gamma=args.focal_gamma,
        smoothing=args.label_smoothing
    )

    # Evaluate
    print("[*] Running evaluation...")
    avg_loss, acc, prec, rec, f1, brier, labels, probs, preds = evaluate_with_probs(
        model, test_loader, loss_fn, device, threshold=args.threshold
    )

    # Print results
    print("\n" + "="*50)
    print("ICDv3 Baseline – Test Results")
    print("="*50)
    print(f"  Loss:      {avg_loss:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Brier:     {brier:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Non-Clickbait", "Clickbait"]))

    # Calibration curve data
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")

    # Save metrics JSON
    metrics = {
        "model": "ICDv3.1",
        "checkpoint": str(ckpt_path),
        "threshold": args.threshold,
        "test_loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier_score": brier,
        "calibration": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }
    metrics_path = RESULTS_DIR / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Metrics saved: {metrics_path}")

    # Save predictions parquet
    # Load IDs từ test split
    test_df_ids = pd.read_csv(DATA_DIR / "test_clean.csv")["id"].astype(str).tolist()
    ids_for_preds = test_df_ids[:len(labels)]

    predictions_df = pd.DataFrame({
        "id": ids_for_preds,
        "label": [int(l) for l in labels],
        "prob": probs,
        "pred": [int(p) for p in preds],
    })
    pred_path = RESULTS_DIR / "test_predictions.parquet"
    predictions_df.to_parquet(str(pred_path), index=False)
    print(f"[+] Predictions saved: {pred_path}")

    # Plot calibration curve
    calib_path = RESULTS_DIR / "calibration_curve.png"
    plot_calibration_curve(labels, probs, calib_path, model_name="ICDv3")
    print(f"[+] Calibration curve: {calib_path}")

    # Plot confusion matrix
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(labels, preds, cm_path, model_name="ICDv3")
    print(f"[+] Confusion matrix: {cm_path}")

    print("\n[✓] Evaluation complete!")


if __name__ == "__main__":
    main()
