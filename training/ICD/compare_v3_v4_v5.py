"""
ICDv5 – Compare ICDv3 vs ICDv4 vs ICDv5
=========================================
So sánh kết quả chi tiết giữa ICDv3 baseline, ICDv4 và các biến thể của ICDv5.

Output:
  src/experience/comparison_v5/comparison_report.md
  src/experience/comparison_v5/calibration_curves.png
  src/experience/comparison_v5/confusion_matrices.png

Chạy:
    conda run -n MLE python training/ICD/compare_v3_v4_v5.py
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V3_METRICS  = BASE_DIR / "src" / "experience" / "ICDv3" / "results" / "test_metrics.json"
V4_METRICS  = BASE_DIR / "src" / "experience" / "icdv4" / "results" / "ICDv4_9FL" / "test_metrics_full.json"

V5_FULL_METRICS = BASE_DIR / "src" / "experience" / "icdv5" / "ICDv5_Full" / "results" / "test_metrics.json"
V5_NO_ROUTER_SUP_METRICS = BASE_DIR / "src" / "experience" / "icdv5" / "ICDv5_NoRouterSup" / "results" / "test_metrics.json"
V5_NO_ROUTER_METRICS = BASE_DIR / "src" / "experience" / "icdv5" / "ICDv5_NoRouter" / "results" / "test_metrics.json"

V3_PREDS    = BASE_DIR / "src" / "experience" / "ICDv3" / "results" / "test_predictions.parquet"
V4_PREDS    = BASE_DIR / "src" / "experience" / "icdv4" / "results" / "ICDv4_9FL" / "test_predictions_full.parquet"

V5_FULL_PREDS = BASE_DIR / "src" / "experience" / "icdv5" / "ICDv5_Full" / "results" / "test_predictions.parquet"
V5_NO_ROUTER_SUP_PREDS = BASE_DIR / "src" / "experience" / "icdv5" / "ICDv5_NoRouterSup" / "results" / "test_predictions.parquet"
V5_NO_ROUTER_PREDS = BASE_DIR / "src" / "experience" / "icdv5" / "ICDv5_NoRouter" / "results" / "test_predictions.parquet"

COMPARE_DIR = BASE_DIR / "src" / "experience" / "comparison_v5"

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        print(f"[WARN] Không tìm thấy: {p}")
        return None
    with open(str(p), "r", encoding="utf-8") as f:
        return json.load(f)

def load_preds(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        print(f"[WARN] Không tìm thấy predictions: {p}")
        return None
    return pd.read_parquet(str(p))

def plot_metrics_bar(metrics_table: pd.DataFrame, out_path: Path):
    """Bar chart so sánh metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # metrics_table có dạng: columns = ['ICDv3', 'ICDv4', 'ICDv5_Full', ...]
    models = list(metrics_table.columns)
    if "Δ (v5-v4)" in models: models.remove("Δ (v5-v4)")
    if "Δ (v5-v3)" in models: models.remove("Δ (v5-v3)")
    
    x = np.arange(len(metrics_table))
    width = 0.8 / len(models)
    
    colors = ["#4C9BE8", "#F4A261", "#E76F51", "#2A9D8F", "#E9C46A"]
    
    for i, model in enumerate(models):
        rects = ax.bar(x + i*width - 0.4 + width/2, metrics_table[model], width, label=model, color=colors[i % len(colors)])
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, rotation=90)
                
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("ICD Models Comparison", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_table.index, fontsize=10)
    ax.set_ylim(0, 1.15) # Leave space for labels
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Metrics bar chart: {out_path}")

def main():
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "ICDv3": load_json(V3_METRICS),
        "ICDv4": load_json(V4_METRICS),
        "ICDv5_Full": load_json(V5_FULL_METRICS),
        "ICDv5_NoRouterSup": load_json(V5_NO_ROUTER_SUP_METRICS),
        "ICDv5_NoRouter": load_json(V5_NO_ROUTER_METRICS)
    }
    
    # ── Global Metrics Table ──────────────────────────────────────────────
    table_data = {}
    for model_name, m in metrics.items():
        if m is None:
            table_data[model_name] = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0}
        else:
            table_data[model_name] = {
                "Accuracy":  m.get("accuracy", 0),
                "Precision": m.get("precision", 0),
                "Recall":    m.get("recall", 0),
                "F1":        m.get("f1", 0)
            }
            
    metrics_table = pd.DataFrame(table_data)
    
    # Tính delta (nếu v5_full tồn tại)
    if "ICDv5_Full" in metrics_table and "ICDv4" in metrics_table:
        metrics_table["Δ (v5-v4)"] = metrics_table["ICDv5_Full"] - metrics_table["ICDv4"]
    if "ICDv5_Full" in metrics_table and "ICDv3" in metrics_table:
        metrics_table["Δ (v5-v3)"] = metrics_table["ICDv5_Full"] - metrics_table["ICDv3"]

    print("\n" + "="*60)
    print("ICDv3 vs ICDv4 vs ICDv5 – Comparison Report")
    print("="*60)
    print(metrics_table.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_data = metrics_table.drop(columns=["Δ (v5-v4)", "Δ (v5-v3)"], errors="ignore")
    # Only plot available models
    plot_data = plot_data.loc[:, (plot_data != 0).any(axis=0)]
    
    if not plot_data.empty:
        plot_metrics_bar(plot_data, COMPARE_DIR / "metrics_bar.png")

    # ── Save Markdown Report ──────────────────────────────────────────────
    report_path = COMPARE_DIR / "comparison_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write("# ICDv3 vs ICDv4 vs ICDv5 – Comparison Report\n\n")
        f.write("## Global Metrics\n\n")
        f.write(metrics_table.to_markdown(floatfmt=".4f"))
        f.write("\n\n## Conclusion\n\n")
        
        v3_f1 = metrics_table.loc["F1", "ICDv3"] if "ICDv3" in metrics_table.columns else 0
        v4_f1 = metrics_table.loc["F1", "ICDv4"] if "ICDv4" in metrics_table.columns else 0
        v5_f1 = metrics_table.loc["F1", "ICDv5_Full"] if "ICDv5_Full" in metrics_table.columns else 0
        
        if v5_f1 > v4_f1:
            f.write(f"✅ ICDv5 cải thiện F1 so với ICDv4: {v4_f1:.4f} → **{v5_f1:.4f}** (+{v5_f1-v4_f1:.4f})\n")
        else:
            f.write(f"⚠️ ICDv5 F1 thấp hơn/bằng ICDv4: {v4_f1:.4f} → {v5_f1:.4f}\n")
            
        f.write(f"- So với baseline (ICDv3): {v3_f1:.4f} → **{v5_f1:.4f}** (+{v5_f1-v3_f1:.4f})\n")

    print(f"\n[+] Report: {report_path}")
    print("[✓] Comparison hoàn thành!")

if __name__ == "__main__":
    main()
