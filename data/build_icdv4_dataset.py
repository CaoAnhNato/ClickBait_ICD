"""
ICDv4 – Build Dataset (Pha 3)
==============================
Merge CSV gốc với reasoning JSONL, tạo soft labels, lưu thành parquet.

Chạy:
    conda run -n MLE python data/build_icdv4_dataset.py
    conda run -n MLE python data/build_icdv4_dataset.py --verify
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Path setup (từ data/ folder lên root)
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.ICD.reasoning.prompts import SOFT_LABEL_LAMBDA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = BASE_DIR / "data" / "processed"
CLEANED_CSV = DATA_DIR / "cleaned" / "Cleaned_Clickbait_Dataset.csv"
TRAIN_CSV   = DATA_DIR / "cleaned" / "train_clean.csv"
VAL_CSV     = DATA_DIR / "cleaned" / "validate_clean.csv"
TEST_CSV    = DATA_DIR / "cleaned" / "test_clean.csv"

REASONING_JSONL = DATA_DIR / "icdv4" / "reasoning_all.jsonl"
OUTPUT_PARQUET  = DATA_DIR / "icdv4" / "Cleaned_Clickbait_with_reasoning.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_reasoning(path: Path) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_split_ids() -> dict[str, str]:
    """Trả về dict {id: split} cho tất cả samples."""
    split_map = {}
    for csv_path, split_name in [(TRAIN_CSV, "train"), (VAL_CSV, "validate"), (TEST_CSV, "test")]:
        df = pd.read_csv(csv_path, usecols=["id"])
        for sid in df["id"].astype(str):
            split_map[sid] = split_name
    return split_map


def compute_soft_labels(df: pd.DataFrame, lam: float = SOFT_LABEL_LAMBDA) -> pd.DataFrame:
    """
    Tạo các trường soft label theo ICDv4.md:
      - p_llm_initial = V_I / 100
      - p_llm_agree   = V_A / 100
      - p_llm_disagree = V_D / 100
      - p_llm_final:
          if label=1: (1-λ)*1.0 + λ*p_llm_initial
          if label=0: (1-λ)*0.0 + λ*p_llm_initial
    """
    df["p_llm_initial"]  = df["initial_score"] / 100.0
    df["p_llm_agree"]    = df["agree_score"]   / 100.0
    df["p_llm_disagree"] = df["disagree_score"] / 100.0

    # Binary label (0/1) – áp dụng label map nếu cần
    label_map = {"non-clickbait": 0, "clickbait": 1}
    if df["label"].dtype == object:
        df["label_bin"] = df["label"].map(label_map).fillna(df["label"]).astype(int)
    else:
        df["label_bin"] = df["label"].astype(int)

    df["p_llm_final"] = np.where(
        df["label_bin"] == 1,
        (1 - lam) * 1.0 + lam * df["p_llm_initial"],
        (1 - lam) * 0.0 + lam * df["p_llm_initial"],
    )

    # Clip to [0, 1]
    df["p_llm_final"]    = df["p_llm_final"].clip(0, 1)
    df["p_llm_initial"]  = df["p_llm_initial"].clip(0, 1)
    df["p_llm_agree"]    = df["p_llm_agree"].clip(0, 1)
    df["p_llm_disagree"] = df["p_llm_disagree"].clip(0, 1)

    return df


def build_dataset(verify_only: bool = False) -> pd.DataFrame:
    # 1. Load CSV gốc
    print(f"[1] Đọc CSV gốc: {CLEANED_CSV}")
    df_csv = pd.read_csv(CLEANED_CSV)
    df_csv["id"] = df_csv["id"].astype(str)
    # Map lead (alias cho lead_paragraph)
    df_csv["lead"] = df_csv["lead_paragraph"].fillna("")
    df_csv["title"] = df_csv["title"].fillna("")
    print(f"    {len(df_csv)} rows")

    # 2. Load reasoning
    print(f"[2] Đọc reasoning: {REASONING_JSONL}")
    df_reason = load_reasoning(REASONING_JSONL)
    df_reason["id"] = df_reason["id"].astype(str)
    print(f"    {len(df_reason)} reasoning records")

    # 3. Load split labels
    print(f"[3] Load split info...")
    split_map = load_split_ids()
    df_csv["split"] = df_csv["id"].map(split_map).fillna("unknown")
    split_counts = df_csv["split"].value_counts()
    print(f"    {dict(split_counts)}")

    # 4. Merge
    print(f"[4] Merge CSV + reasoning theo id...")
    df_merged = df_csv.merge(df_reason, on="id", how="inner")
    print(f"    Trước merge: {len(df_csv)} | Sau merge: {len(df_merged)}")

    missing = len(df_csv) - len(df_merged)
    if missing > 0:
        missing_ids = set(df_csv["id"]) - set(df_merged["id"])
        print(f"    ⚠️  {missing} samples thiếu reasoning: {list(missing_ids)[:10]}")

    # 5. Loại samples có reasoning_status = failed (nếu có)
    if "reasoning_status" in df_merged.columns:
        failed = df_merged["reasoning_status"] != "success"
        if failed.sum() > 0:
            print(f"    ⚠️  Bỏ {failed.sum()} samples với status failed")
            df_merged = df_merged[~failed].reset_index(drop=True)

    # 6. Tạo soft labels
    print(f"[5] Tạo soft labels (λ={SOFT_LABEL_LAMBDA})...")
    df_merged = compute_soft_labels(df_merged, lam=SOFT_LABEL_LAMBDA)

    # 7. Chọn columns cần thiết
    keep_cols = [
        "id", "title", "lead_paragraph", "lead", "label", "label_bin", "split",
        "initial_score", "initial_reason",
        "agree_reason", "agree_score",
        "disagree_reason", "disagree_score",
        "reasoning_status",
        "p_llm_initial", "p_llm_agree", "p_llm_disagree", "p_llm_final",
    ]
    # Thêm các cột phụ nếu có
    extra_cols = ["category", "source", "publish_datetime"]
    for col in extra_cols:
        if col in df_merged.columns:
            keep_cols.append(col)

    df_final = df_merged[[c for c in keep_cols if c in df_merged.columns]]

    if verify_only:
        return df_final

    # 8. Save parquet
    print(f"[6] Lưu parquet: {OUTPUT_PARQUET}")
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"    ✓ Lưu {len(df_final)} rows")

    return df_final


def verify_dataset(df: pd.DataFrame):
    """Kiểm tra tính hợp lệ của dataset đã build."""
    print(f"\n{'='*60}")
    print("ICDv4 Dataset Verification")
    print(f"{'='*60}")
    print(f"\nTotal rows: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")

    # Missing values
    print(f"\n[CHECK] Missing values:")
    missing = df.isnull().sum()
    for col, cnt in missing[missing > 0].items():
        print(f"  ⚠️  {col}: {cnt} missing")
    if missing.sum() == 0:
        print("  ✅ Không có missing values")

    # Soft label range
    print(f"\n[CHECK] Soft label ranges:")
    for col in ["p_llm_initial", "p_llm_agree", "p_llm_disagree", "p_llm_final"]:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
            print(f"  {col}: min={df[col].min():.3f}, max={df[col].max():.3f}"
                  f" | out-of-range: {out_of_range}")

    # Label distribution
    print(f"\n[CHECK] Label distribution:")
    if "label_bin" in df.columns:
        dist = df["label_bin"].value_counts()
        print(f"  0 (non-clickbait): {dist.get(0, 0)}")
        print(f"  1 (clickbait):     {dist.get(1, 0)}")

    # Split distribution
    print(f"\n[CHECK] Split distribution:")
    if "split" in df.columns:
        for sp, cnt in df["split"].value_counts().items():
            print(f"  {sp}: {cnt}")

    print(f"\n{'='*60}")
    print("✅ Verification complete")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="ICDv4 – Build Dataset")
    parser.add_argument("--verify", action="store_true",
                        help="Chỉ verify dataset đã build (không save lại)")
    args = parser.parse_args()

    if args.verify and OUTPUT_PARQUET.exists():
        print(f"Đọc dataset đã build: {OUTPUT_PARQUET}")
        df = pd.read_parquet(OUTPUT_PARQUET)
        verify_dataset(df)
    else:
        df = build_dataset(verify_only=False)
        verify_dataset(df)


if __name__ == "__main__":
    main()
