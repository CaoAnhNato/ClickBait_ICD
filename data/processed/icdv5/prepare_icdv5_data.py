import os
import sys
import pandas as pd
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[3]
CLEANED_DIR = BASE_DIR / "data" / "processed" / "cleaned"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "icdv5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_and_save(input_filename, output_filename):
    input_path = CLEANED_DIR / input_filename
    output_path = OUTPUT_DIR / output_filename
    
    print(f"Processing {input_path}...")
    df = pd.read_csv(input_path)
    
    # Map labels: clickbait -> 1, non-clickbait -> 0
    if 'label' in df.columns:
        df['label'] = df['label'].map({'clickbait': 1, 'non-clickbait': 0})
    else:
        print(f"Warning: 'label' column not found in {input_filename}")
        
    # Select columns
    columns_to_keep = ['id', 'title', 'lead_paragraph', 'category', 'source', 'label']
    
    # Check if columns exist, otherwise fill with NA/empty string
    for col in columns_to_keep:
        if col not in df.columns:
            print(f"Warning: '{col}' column not found in {input_filename}, creating empty column.")
            df[col] = ""
            
    df = df[columns_to_keep]
    
    # Fill NA for text fields
    df['title'] = df['title'].fillna("")
    df['lead_paragraph'] = df['lead_paragraph'].fillna("")
    df['category'] = df['category'].fillna("Unknown")
    df['source'] = df['source'].fillna("Unknown")
    
    # Convert ID to string to ensure consistency
    df['id'] = df['id'].astype(str)
    
    print(f"Saving to {output_path} with shape {df.shape}")
    df.to_parquet(output_path, index=False)

def main():
    splits = [
        ("train_best_cleaned.csv", "icdv5_train_base.parquet"),
        ("validate_best_cleaned.csv", "icdv5_valid_base.parquet"),
        ("test_best_cleaned.csv", "icdv5_test_base.parquet")
    ]
    
    for in_file, out_file in splits:
        process_and_save(in_file, out_file)
        
    print("Done preparing ICDv5 base datasets.")

if __name__ == "__main__":
    main()
