import os
import sys
import pandas as pd
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[3]
ICDV5_DIR = BASE_DIR / "data" / "processed" / "icdv5"
ICDV4_REASONING_FILE = BASE_DIR / "data" / "processed" / "icdv4" / "Cleaned_Clickbait_with_reasoning.parquet"

def merge_datasets(split: str):
    print(f"--- Building {split} dataset ---")
    
    # 1. Load base
    base_file = ICDV5_DIR / f"icdv5_{split}_base.parquet"
    if not base_file.exists():
        print(f"Error: Base file not found: {base_file}")
        return
    df_base = pd.read_parquet(base_file)
    df_base['id'] = df_base['id'].astype(str)
    
    # 2. Load patterns
    patterns_file = ICDV5_DIR / f"icdv5_{split}_patterns.parquet"
    if not patterns_file.exists():
        print(f"Error: Patterns file not found: {patterns_file}. Please run generate_pattern_tags.py first.")
        return
    df_patterns = pd.read_parquet(patterns_file)
    df_patterns['id'] = df_patterns['id'].astype(str)
    
    # 3. Load reasoning (common for all splits)
    if not ICDV4_REASONING_FILE.exists():
        print(f"Error: ICDv4 reasoning file not found: {ICDV4_REASONING_FILE}")
        return
    df_reasoning = pd.read_parquet(ICDV4_REASONING_FILE)
    df_reasoning['id'] = df_reasoning['id'].astype(str)
    
    # Select reasoning columns and rename
    reasoning_cols = {
        'id': 'id',
        'agree_reason': 'reason_agree_vi',
        'disagree_reason': 'reason_disagree_vi',
        'initial_score': 'score_init',
        'agree_score': 'score_agree',
        'disagree_score': 'score_disagree'
    }
    df_reasoning = df_reasoning[list(reasoning_cols.keys())].rename(columns=reasoning_cols)
    
    # 4. Merge all
    print(f"Base shape: {df_base.shape}")
    print(f"Patterns shape: {df_patterns.shape}")
    
    df_merged = df_base.merge(df_patterns, on='id', how='left')
    df_merged = df_merged.merge(df_reasoning, on='id', how='left')
    
    # 5. Fill missing values
    tags = ['tag_shock', 'tag_lifestyle', 'tag_listicle', 'tag_analysis', 'tag_promo', 'tag_hardnews']
    for tag in tags:
        df_merged[tag] = df_merged[tag].fillna(0).astype(int)
        
    df_merged['reason_agree_vi'] = df_merged['reason_agree_vi'].fillna("")
    df_merged['reason_disagree_vi'] = df_merged['reason_disagree_vi'].fillna("")
    df_merged['score_init'] = df_merged['score_init'].fillna(50.0).astype(float)
    df_merged['score_agree'] = df_merged['score_agree'].fillna(50.0).astype(float)
    df_merged['score_disagree'] = df_merged['score_disagree'].fillna(50.0).astype(float)
    
    print(f"Merged shape: {df_merged.shape}")
    
    # 6. Save final dataset
    out_file = ICDV5_DIR / f"icdv5_{split}_full.parquet"
    df_merged.to_parquet(out_file, index=False)
    print(f"Saved to {out_file}\n")

def main():
    splits = ["train", "valid", "test"]
    for split in splits:
        merge_datasets(split)
        
    print("Dataset building completed.")

if __name__ == "__main__":
    main()
