"""
ICDv4 – Compare ICDv3 vs ICDv4 (Pha 6)
=========================================
So sánh kết quả chi tiết giữa ICDv3 baseline và ICDv4.

Output:
  src/experience/comparison/comparison_report.md
  src/experience/comparison/calibration_curves.png
  src/experience/comparison/confusion_matrices.png

Chạy:
    conda run -n MLE python training/ICD/compare_v3_v4.py
"""

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V3_METRICS  = BASE_DIR / "src" / "experience" / "ICDv3" / "results" / "test_metrics.json"
V4_METRICS  = BASE_DIR / "src" / "experience" / "icdv4" / "results" / "ICDv4_9FL" / "test_metrics_full.json"
V3_PREDS    = BASE_DIR / "src" / "experience" / "ICDv3" / "results" / "test_predictions.parquet"
V4_PREDS    = BASE_DIR / "src" / "experience" / "icdv4" / "results" / "ICDv4_9FL" / "test_predictions_full.parquet"
COMPARE_DIR = BASE_DIR / "src" / "experience" / "comparison"


# ---------------------------------------------------------------------------
# McNemar's Test (statistical significance)
# ---------------------------------------------------------------------------
def mcnemar_test(v3_preds: list, v4_preds: list, labels: list) -> dict:
    """
    McNemar's test để kiểm tra sự khác biệt có ý nghĩa thống kê.
    H0: ICDv3 và ICDv4 có cùng tỷ lệ lỗi.
    """
    labels  = np.array(labels)
    v3_arr  = np.array(v3_preds)
    v4_arr  = np.array(v4_preds)

    v3_correct = (v3_arr == labels)
    v4_correct = (v4_arr == labels)

    # Contingency table
    b = np.sum(v3_correct & ~v4_correct)   # v3 đúng, v4 sai
    c = np.sum(~v3_correct & v4_correct)   # v3 sai, v4 đúng

    if (b + c) == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0, "significant": False}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # với Yates' correction
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "b": int(b), "c": int(c),
        "chi2": float(chi2), "p_value": float(p_value),
        "significant": p_value < 0.05,
        "interpretation": (
            f"v4 cải thiện {c} cases mà v3 sai, v4 làm hỏng {b} cases mà v3 đúng."
        )
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_calibration_comparison(v3_labels, v3_probs, v4_labels, v4_probs, out_path: Path):
    """Vẽ calibration curves ICDv3 vs ICDv4 trên cùng figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration curves
    for model_name, labels, probs, color in [
        ("ICDv3", v3_labels, v3_probs, "#4C9BE8"),
        ("ICDv4", v4_labels, v4_probs, "#F4A261"),
    ]:
        prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")
        brier = brier_score_loss(labels, probs)
        ax1.plot(prob_pred, prob_true, "o-", label=f"{model_name} (Brier={brier:.3f})",
                 color=color, linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax1.set_xlabel("Mean predicted probability", fontsize=11)
    ax1.set_ylabel("Fraction of positives", fontsize=11)
    ax1.set_title("Calibration Curves (Reliability Diagram)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Probability distribution
    ax2.hist(v3_probs, bins=30, alpha=0.5, label="ICDv3", color="#4C9BE8", density=True)
    ax2.hist(v4_probs, bins=30, alpha=0.5, label="ICDv4", color="#F4A261", density=True)
    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Probability Distribution Comparison", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("ICDv3 vs ICDv4 – Calibration Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Calibration plot: {out_path}")


def plot_confusion_matrices(v3_labels, v3_preds, v4_labels, v4_preds, out_path: Path):
    """Side-by-side confusion matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    classes = ["Non-CB", "Clickbait"]

    for ax, labels, preds, title in [
        (ax1, v3_labels, v3_preds, "ICDv3 Confusion Matrix"),
        (ax2, v4_labels, v4_preds, "ICDv4 Confusion Matrix"),
    ]:
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.suptitle("ICDv3 vs ICDv4 – Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Confusion matrices: {out_path}")


def plot_metrics_bar(metrics_table: pd.DataFrame, out_path: Path):
    """Bar chart so sánh metrics."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics_table))
    width = 0.35

    rects1 = ax.bar(x - width/2, metrics_table["ICDv3"], width, label="ICDv3", color="#4C9BE8")
    rects2 = ax.bar(x + width/2, metrics_table["ICDv4"], width, label="ICDv4", color="#F4A261")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("ICDv3 vs ICDv4 – Metrics Comparison", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_table.index, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Metrics bar chart: {out_path}")


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def main():
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    # Load metrics JSONs
    def load_json(p: Path) -> Optional[dict]:
        if not p.exists():
            print(f"[WARN] Không tìm thấy: {p}")
            return None
        with open(str(p), "r", encoding="utf-8") as f:
            return json.load(f)

    v3_metrics = load_json(V3_METRICS)
    v4_metrics = load_json(V4_METRICS)

    if v3_metrics is None or v4_metrics is None:
        print("[ERROR] Chưa có đủ metrics. Chạy eval_baseline_v3.py và train_ICD_v4.py trước.")
        sys.exit(1)

    # Load predictions
    def load_preds(p: Path) -> Optional[pd.DataFrame]:
        if not p.exists():
            print(f"[WARN] Không tìm thấy predictions: {p}")
            return None
        return pd.read_parquet(str(p))

    v3_df = load_preds(V3_PREDS)
    v4_df = load_preds(V4_PREDS)

    # Align predictions theo ID
    if v3_df is not None and v4_df is not None:
        merged = v3_df.merge(v4_df, on="id", suffixes=("_v3", "_v4"))
        labels    = merged["label_v3"].tolist()
        v3_probs  = merged["prob_v3"].tolist()
        v4_probs  = merged["prob_v4"].tolist()
        v3_preds  = merged["pred_v3"].tolist()
        v4_preds  = merged["pred_v4"].tolist()
    else:
        labels = v3_probs = v4_probs = v3_preds = v4_preds = []

    # Calculate Brier from probs if missing (e.g. for ICDv4)
    v3_brier = v3_metrics.get("brier_score")
    if (v3_brier is None or v3_brier == 0) and labels:
        v3_brier = brier_score_loss(labels, v3_probs)
    
    v4_brier = v4_metrics.get("brier_score")
    if (v4_brier is None or v4_brier == 0) and labels:
        v4_brier = brier_score_loss(labels, v4_probs)

    # ── Global Metrics Table ──────────────────────────────────────────────
    metrics_table = pd.DataFrame({
        "ICDv3": {
            "Accuracy":  v3_metrics.get("accuracy", 0),
            "Precision": v3_metrics.get("precision", 0),
            "Recall":    v3_metrics.get("recall", 0),
            "F1":        v3_metrics.get("f1", 0),
            "Brier":     v3_brier if v3_brier is not None else 0,
        },
        "ICDv4": {
            "Accuracy":  v4_metrics.get("accuracy", 0),
            "Precision": v4_metrics.get("precision", 0),
            "Recall":    v4_metrics.get("recall", 0),
            "F1":        v4_metrics.get("f1", 0),
            "Brier":     v4_brier if v4_brier is not None else 0,
        },
    })
    metrics_table["Δ (v4-v3)"] = metrics_table["ICDv4"] - metrics_table["ICDv3"]
    # Brier: lower is better
    metrics_table.loc["Brier", "Δ (v4-v3)"] = -(metrics_table.loc["Brier", "ICDv4"] -
                                                  metrics_table.loc["Brier", "ICDv3"])

    print("\n" + "="*60)
    print("ICDv3 vs ICDv4 – Comparison Report")
    print("="*60)
    print(metrics_table.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Statistical Significance ──────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Statistical Significance (McNemar's Test):")
    if labels:
        mcnemar = mcnemar_test(v3_preds, v4_preds, labels)
        print(f"  b={mcnemar['b']}, c={mcnemar['c']}")
        print(f"  chi2={mcnemar['chi2']:.4f}, p={mcnemar['p_value']:.4f}")
        print(f"  Significant: {mcnemar['significant']}")
        print(f"  {mcnemar['interpretation']}")
    else:
        mcnemar = {}
        print("  (cần predictions để tính)")

    # ── Kết luận ─────────────────────────────────────────────────────────
    v3_f1 = v3_metrics.get("f1", 0)
    v4_f1 = v4_metrics.get("f1", 0)
    print(f"\n{'─'*50}")
    print("Kết luận:")
    if v4_f1 >= v3_f1:
        print(f"  ✅ ICDv4 cải thiện F1: {v3_f1:.4f} → {v4_f1:.4f} (+{v4_f1-v3_f1:.4f})")
    else:
        print(f"  ⚠️  ICDv4 F1 thấp hơn ICDv3: {v3_f1:.4f} → {v4_f1:.4f} ({v4_f1-v3_f1:.4f})")

    # ── Plots ─────────────────────────────────────────────────────────────
    if labels:
        plot_calibration_comparison(
            labels, v3_probs, labels, v4_probs,
            COMPARE_DIR / "calibration_curves.png"
        )
        plot_confusion_matrices(
            labels, v3_preds, labels, v4_preds,
            COMPARE_DIR / "confusion_matrices.png"
        )

    # Bar chart metrics (exclude Brier for readability)
    plot_data = metrics_table.drop(index=["Brier", "Δ (v4-v3)"] if "Δ (v4-v3)" in metrics_table.index else ["Brier"],
                                   errors="ignore")[["ICDv3", "ICDv4"]]
    plot_metrics_bar(plot_data, COMPARE_DIR / "metrics_bar.png")

    # ── Save Markdown Report ──────────────────────────────────────────────
    report_path = COMPARE_DIR / "comparison_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write("# ICDv3 vs ICDv4 – Comparison Report\n\n")
        f.write(f"**ICDv3 checkpoint**: `{v3_metrics.get('checkpoint','N/A')}`\n")
        f.write(f"**ICDv4 checkpoint**: `{v4_metrics.get('checkpoint','N/A')}`\n\n")
        f.write("## Global Metrics\n\n")
        f.write(metrics_table.to_markdown(floatfmt=".4f"))
        f.write("\n\n## McNemar's Test\n\n")
        if mcnemar:
            f.write(f"| b | c | chi2 | p-value | Significant |\n")
            f.write(f"|---|---|------|---------|-------------|\n")
            f.write(f"| {mcnemar.get('b','-')} | {mcnemar.get('c','-')} | "
                    f"{mcnemar.get('chi2',0):.4f} | {mcnemar.get('p_value',0):.4f} | "
                    f"{'Yes' if mcnemar.get('significant') else 'No'} |\n")
        f.write(f"\n\n## Conclusion\n\n")
        if v4_f1 >= v3_f1:
            f.write(f"✅ ICDv4 cải thiện F1 từ **{v3_f1:.4f}** lên **{v4_f1:.4f}** (+{v4_f1-v3_f1:.4f})\n")
        else:
            f.write(f"⚠️ ICDv4 F1 thấp hơn ICDv3: {v3_f1:.4f} → {v4_f1:.4f}\n")

    print(f"\n[+] Report: {report_path}")
    print("[✓] Comparison hoàn thành!")


if __name__ == "__main__":
    main()
