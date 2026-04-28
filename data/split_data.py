import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Đường dẫn file gốc
input_file = "data/raw/clickbait_dataset_vietnamese.csv"

# Đọc dữ liệu
df = pd.read_csv(input_file)

label_col = "label" 

# Tách train (80%) và phần còn lại (20%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df[label_col]
)

# Tách validate/test từ 20% còn lại thành 10%/10%
validate_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df[label_col]
)

# Tạo thư mục processed nếu chưa có
os.makedirs("data/processed", exist_ok=True)

# Lưu thành 3 file
train_df.to_csv("data/processed/train.csv", index=False, encoding="utf-8-sig")
validate_df.to_csv("data/processed/validate.csv", index=False, encoding="utf-8-sig")
test_df.to_csv("data/processed/test.csv", index=False, encoding="utf-8-sig")

print(f"Done: train={len(train_df)}, validate={len(validate_df)}, test={len(test_df)}")
print(f"Label column used: {label_col}")