import os
import sys
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import py_vncorenlp

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

# Lazy segmenter
vncorenlp_path = os.path.expanduser('~/.vncorenlp_data')
_rdrsegmenter = None

def get_segmenter():
    global _rdrsegmenter
    if _rdrsegmenter is None:
        if not os.path.exists(vncorenlp_path):
            os.makedirs(vncorenlp_path, exist_ok=True)
            py_vncorenlp.download_model(save_dir=vncorenlp_path)
        orig_cwd = os.getcwd()
        try:
            _rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)
        finally:
            os.chdir(orig_cwd)
    return _rdrsegmenter

def preprocess_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    segmenter = get_segmenter()
    return " ".join(segmenter.word_segment(str(text))).strip()

def build_split(split, cat2id, src2id, icdv6_dir, icdv5_dir, cleaned_dir):
    print(f"\n[*] Building {split} split...")
    cleaned_file = cleaned_dir / f"{split}_best_cleaned.csv"
    if split == "valid":
        cleaned_file = cleaned_dir / "validate_best_cleaned.csv"
        
    df = pd.read_csv(cleaned_file)
    
    # Map category and source
    df["category_id"] = df["category"].map(cat2id).fillna(0).astype(int)
    df["source_id"] = df["source"].map(src2id).fillna(0).astype(int)
    
    # Load icdv5 patterns
    pattern_file = icdv5_dir / f"icdv5_{split}_patterns.parquet"
    if pattern_file.exists():
        df_patterns = pd.read_parquet(pattern_file)
        # Merge by id
        df = df.merge(df_patterns, on="id", how="left")
    else:
        print(f"Warning: Missing pattern tags for {split}. Filling with 0.")
        for tag in ["tag_shock", "tag_lifestyle", "tag_listicle", "tag_analysis", "tag_promo", "tag_hardnews"]:
            df[tag] = 0

    # Fill NaN tags with 0
    for tag in ["tag_shock", "tag_lifestyle", "tag_listicle", "tag_analysis", "tag_promo", "tag_hardnews"]:
        if tag in df.columns:
            df[tag] = df[tag].fillna(0).astype(int)
            
    # Segmentation
    tqdm.pandas(desc=f"Segmenting {split} titles")
    df["title_seg"] = df["title"].progress_apply(preprocess_text)
    
    tqdm.pandas(desc=f"Segmenting {split} leads")
    df["lead_seg"] = df["lead_paragraph"].progress_apply(preprocess_text)
    
    # Clean label
    label_map = {'non-clickbait': 0, 'clickbait': 1}
    df['label'] = df['label'].map(label_map).fillna(0).astype(int)
    
    out_file = icdv6_dir / f"icdv6_{split}.parquet"
    df.to_parquet(out_file, index=False)
    print(f"[+] Saved {out_file.name} - Shape: {df.shape}")

def main():
    cleaned_dir = BASE_DIR / "data/processed/cleaned"
    icdv5_dir = BASE_DIR / "data/processed/icdv5"
    icdv6_dir = BASE_DIR / "data/processed/icdv6"
    icdv6_dir.mkdir(parents=True, exist_ok=True)
    
    # Build vocabulary for category and source from all cleaned splits
    df_train = pd.read_csv(cleaned_dir / "train_best_cleaned.csv")
    df_val = pd.read_csv(cleaned_dir / "validate_best_cleaned.csv")
    df_test = pd.read_csv(cleaned_dir / "test_best_cleaned.csv")
    
    all_cats = pd.concat([df_train["category"], df_val["category"], df_test["category"]]).unique()
    all_srcs = pd.concat([df_train["source"], df_val["source"], df_test["source"]]).unique()
    
    cat2id = {c: i for i, c in enumerate(all_cats)}
    src2id = {s: i for i, s in enumerate(all_srcs)}
    
    with open(icdv6_dir / "cat2id.json", "w", encoding="utf-8") as f:
        json.dump(cat2id, f, ensure_ascii=False, indent=2)
    with open(icdv6_dir / "src2id.json", "w", encoding="utf-8") as f:
        json.dump(src2id, f, ensure_ascii=False, indent=2)
        
    print(f"[*] Found {len(cat2id)} categories and {len(src2id)} sources.")
    
    build_split("train", cat2id, src2id, icdv6_dir, icdv5_dir, cleaned_dir)
    build_split("valid", cat2id, src2id, icdv6_dir, icdv5_dir, cleaned_dir)
    build_split("test", cat2id, src2id, icdv6_dir, icdv5_dir, cleaned_dir)
    
    print("\n[*] ICDv6 Data Prep Complete!")

if __name__ == "__main__":
    main()
