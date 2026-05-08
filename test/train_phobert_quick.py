import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from tqdm import tqdm
import py_vncorenlp

import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Cấu hình đường dẫn
    dataset_path = 'data/processed/cleaned/Cleaned_Clickbait_Dataset.csv'
    json_path = 'data/test_data/split_ids.json'
    
    if not os.path.exists(json_path):
        print(f"Không tìm thấy {json_path}. Vui lòng chạy cell cuối trong check_distribute.ipynb trước.")
        return

    # Khởi tạo py_vncorenlp (Word Segmenter)
    vncorenlp_path = os.path.join(os.getcwd(), 'vncorenlp_data')
    if not os.path.exists(vncorenlp_path):
        os.makedirs(vncorenlp_path, exist_ok=True)
        py_vncorenlp.download_model(save_dir=vncorenlp_path)
    
    print("Loading VnCoreNLP...")
    orig_cwd = os.getcwd()
    segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)
    os.chdir(orig_cwd)

    def segment_text(text):
        if pd.isna(text): return ""
        return " ".join(segmenter.word_segment(str(text))).strip()

    # 2. Load dữ liệu
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    with open(json_path, 'r') as f:
        split_ids = json.load(f)

    # Nối Title và Lead Paragraph
    df['title'] = df['title'].fillna('')
    df['lead_paragraph'] = df['lead_paragraph'].fillna('')
    df['text_concat'] = df['title'] + " " + df['lead_paragraph']

    # Lọc train và validate
    train_df = df[df['id'].isin(split_ids['train'])].copy()
    val_df = df[df['id'].isin(split_ids['validate'])].copy()

    print("Tokenizing texts with py_vncorenlp...")
    # Chỉ segment những text thực sự dùng để tiết kiệm thời gian
    train_df['text_seg'] = train_df['text_concat'].apply(segment_text)
    val_df['text_seg'] = val_df['text_concat'].apply(segment_text)

    # Encode label
    label_map = {'non-clickbait': 0, 'clickbait': 1}
    train_df['label_id'] = train_df['label'].map(label_map)
    val_df['label_id'] = val_df['label'].map(label_map)

    # 3. Model setup (PhoBERT)
    model_name = "vinai/phobert-base-v2"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class ClickbaitDataset(Dataset):
        def __init__(self, texts, labels):
            self.encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
            self.labels = torch.tensor(labels)
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encoded.items()}
            item['labels'] = self.labels[idx]
            return item

    train_dataset = ClickbaitDataset(train_df['text_seg'].tolist(), train_df['label_id'].tolist())
    val_dataset = ClickbaitDataset(val_df['text_seg'].tolist(), val_df['label_id'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 4. Training loop
    epochs = 5
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_preds, train_labels = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # VALIDATE
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validate"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"\n=> Epoch {epoch+1} Summary: Train F1 (Macro) = {train_f1:.4f} | Val F1 (Macro) = {val_f1:.4f}\n")

if __name__ == '__main__':
    main()
