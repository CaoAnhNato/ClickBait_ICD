import pandas as pd
import json
from pathlib import Path

def convert_parquet_to_json():
    # Define paths - use the directory where the script is located
    current_dir = Path(__file__).resolve().parent
    input_path = current_dir / "test_predictions_full.parquet"
    output_path = current_dir / "test_predictions_full.json"
    
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Optional: Convert types for better JSON serialization if needed
    # (id is likely int/str, labels are int, probs are float)
    
    print(f"Saving to {output_path}...")
    # orient='records' creates a list of dicts
    # force_ascii=False to preserve Vietnamese characters if any
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    
    print(f"[✓] Successfully converted to JSON: {output_path}")

if __name__ == "__main__":
    convert_parquet_to_json()
