import pandas as pd
import json
import os

def remap_data():
    # Đường dẫn file
    # Lấy 'id' từ file split ids, ưu tiên dùng best_split_ids.json theo yêu cầu mapping
    # Fallback về data/test_data/split_ids.json nếu người dùng truyền nhầm tên file
    split_ids_path = 'data/processed/best_split_ids.json'
    if not os.path.exists(split_ids_path):
        split_ids_path = 'data/test_data/split_ids.json'
        
    cleaned_csv_path = 'data/processed/cleaned/Cleaned_Clickbait_Dataset.csv'
    output_dir = 'data/processed/cleaned'

    print(f"Reading split IDs from {split_ids_path}...")
    with open(split_ids_path, 'r', encoding='utf-8') as f:
        split_ids = json.load(f)

    print(f"Reading cleaned dataset from {cleaned_csv_path}...")
    df_cleaned = pd.read_csv(cleaned_csv_path)

    for split_name in ['train', 'validate', 'test']:
        if split_name in split_ids:
            # Lấy list ids của split
            ids = split_ids[split_name]
            
            # Mapping ID (Những 'id' trong split_ids == 'id' trong Cleaned_Clickbait_Dataset.csv)
            df_split = df_cleaned[df_cleaned['id'].isin(ids)]
            
            # Đường dẫn file output
            output_path = os.path.join(output_dir, f'{split_name}_best_cleaned.csv')
            
            # Lưu lại thành file csv
            df_split.to_csv(output_path, index=False)
            print(f"Saved {split_name} split with {len(df_split)} rows to {output_path}")
        else:
            print(f"Warning: Split '{split_name}' not found in {split_ids_path}")

if __name__ == '__main__':
    remap_data()
