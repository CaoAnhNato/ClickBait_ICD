"""
ICDv4 – Validate Reasoning Dataset (Pha 2)
==========================================
Kiểm tra chất lượng reasoning sau khi sinh.

Chạy:
    conda run -n MLE python src/ICD/reasoning/validate_reasoning.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

from src.ICD.reasoning.prompts import ALPHA, BETA, GAMMA

REASONING_JSONL = BASE_DIR / "data" / "processed" / "icdv4" / "reasoning_all.jsonl"
REPORT_DIR = BASE_DIR / "data" / "processed" / "icdv4"


def load_reasoning(path: Path) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def validate_reasoning(df: pd.DataFrame) -> dict:
    """Kiểm tra các ràng buộc SORG."""
    total = len(df)
    
    # Status
    success = (df["reasoning_status"] == "success").sum()
    failed = total - success
    
    # Ràng buộc initial rating
    vi_in_range = ((df["initial_score"] >= ALPHA) & (df["initial_score"] <= (100 - ALPHA))).sum()
    
    # Ràng buộc agree
    va_polarity = (df["agree_score"] >= (50 + GAMMA)).sum()
    va_delta = ((df["agree_score"] - df["initial_score"]) >= BETA).sum()
    va_gt_vi = (df["agree_score"] > df["initial_score"]).sum()
    
    # Ràng buộc disagree
    vd_polarity = (df["disagree_score"] <= (50 - GAMMA)).sum()
    vd_delta = ((df["initial_score"] - df["disagree_score"]) >= BETA).sum()
    vd_lt_vi = (df["disagree_score"] < df["initial_score"]).sum()
    
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "success_rate": success / total * 100,
        "vi_in_range": vi_in_range,
        "vi_in_range_rate": vi_in_range / total * 100,
        "va_polarity_ok": va_polarity,
        "va_delta_ok": va_delta,
        "va_gt_vi": va_gt_vi,
        "vd_polarity_ok": vd_polarity,
        "vd_delta_ok": vd_delta,
        "vd_lt_vi": vd_lt_vi,
    }


def plot_distributions(df: pd.DataFrame, output_dir: Path):
    """Vẽ histogram phân phối scores."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(df["initial_score"], bins=20, color="#4C9BE8", edgecolor="white")
    axes[0].set_title("Initial Score (V_I) Distribution")
    axes[0].set_xlabel("Score (0-100)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(ALPHA, color="red", linestyle="--", label=f"α={ALPHA}")
    axes[0].axvline(100 - ALPHA, color="red", linestyle="--")
    axes[0].legend()

    axes[1].hist(df["agree_score"], bins=20, color="#F4A261", edgecolor="white")
    axes[1].set_title("Agree Score (V_A) Distribution")
    axes[1].set_xlabel("Score (0-100)")
    axes[1].axvline(50 + GAMMA, color="red", linestyle="--", label=f"50+γ={50+GAMMA}")
    axes[1].legend()

    axes[2].hist(df["disagree_score"], bins=20, color="#2EC4B6", edgecolor="white")
    axes[2].set_title("Disagree Score (V_D) Distribution")
    axes[2].set_xlabel("Score (0-100)")
    axes[2].axvline(50 - GAMMA, color="red", linestyle="--", label=f"50-γ={50-GAMMA}")
    axes[2].legend()

    plt.tight_layout()
    out_path = output_dir / "reasoning_score_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


def main():
    if not REASONING_JSONL.exists():
        print(f"[ERROR] File không tồn tại: {REASONING_JSONL}")
        print("Chạy generate_reasoning.py trước.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("ICDv4 – Reasoning Dataset Validation Report")
    print(f"{'='*60}")

    df = load_reasoning(REASONING_JSONL)
    print(f"\n[INFO] Đọc {len(df)} records từ {REASONING_JSONL.name}")

    # Validate
    stats = validate_reasoning(df)

    print(f"\n{'─'*40}")
    print("📊 Status Overview:")
    print(f"  Total samples:     {stats['total']}")
    print(f"  Success:           {stats['success']} ({stats['success_rate']:.1f}%)")
    print(f"  Failed:            {stats['failed']}")

    print(f"\n{'─'*40}")
    print("✅ SORG Constraint Checks:")
    print(f"  V_I ∈ [{ALPHA}, {100-ALPHA}]:     {stats['vi_in_range']}/{stats['total']} ({stats['vi_in_range_rate']:.1f}%)")
    print(f"  V_A ≥ {50+GAMMA} (polarity):  {stats['va_polarity_ok']}/{stats['total']}")
    print(f"  V_A - V_I ≥ {BETA}:       {stats['va_delta_ok']}/{stats['total']}")
    print(f"  V_A > V_I:          {stats['va_gt_vi']}/{stats['total']}")
    print(f"  V_D ≤ {50-GAMMA} (polarity):  {stats['vd_polarity_ok']}/{stats['total']}")
    print(f"  V_I - V_D ≥ {BETA}:       {stats['vd_delta_ok']}/{stats['total']}")
    print(f"  V_D < V_I:          {stats['vd_lt_vi']}/{stats['total']}")

    print(f"\n{'─'*40}")
    print("📈 Score Statistics:")
    for col in ["initial_score", "agree_score", "disagree_score"]:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}, "
                  f"min={df[col].min()}, max={df[col].max()}")

    print(f"\n{'─'*40}")
    print("🔍 Sample Examples (5 ngẫu nhiên):")
    for _, row in df.sample(min(5, len(df)), random_state=42).iterrows():
        print(f"\n  ID: {row['id']}")
        print(f"  V_I={row['initial_score']} | V_A={row['agree_score']} | V_D={row['disagree_score']}")
        print(f"  Initial: {str(row.get('initial_reason',''))[:80]}...")
        print(f"  Agree:   {str(row.get('agree_reason',''))[:80]}...")
        print(f"  Disagree:{str(row.get('disagree_reason',''))[:80]}...")

    # Plot
    print(f"\n{'─'*40}")
    print("📉 Tạo distribution plots...")
    plot_distributions(df, REPORT_DIR)

    print(f"\n{'='*60}")
    if stats["success_rate"] >= 95.0:
        print("✅ Kết luận: Dataset reasoning đạt chất lượng (≥95% success)")
    else:
        print(f"⚠️  Cảnh báo: Success rate thấp ({stats['success_rate']:.1f}%), cần kiểm tra lại")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
