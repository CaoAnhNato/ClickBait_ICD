import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer
from underthesea import word_tokenize
from tqdm import tqdm
import json
import numpy as np
import sys

# Thêm đường dẫn root vào sys.path để có thể import từ src
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.ICD_Model import ClickbaitDetectionModel, JointLoss

class CharTokenizer:
    """Bộ tokenizer tự xây dựng cho cấp độ ký tự"""
    def __init__(self, max_length=150):
        self.max_length = max_length
        # Dùng ASCII và các ký tự phổ biến (kích thước vocab = 256)
        self.vocab_size = 256
        
    def encode(self, texts):
        char_ids = []
        char_masks = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            text_str = str(text)
            
            # Chuyển đổi ký tự thành ID, dùng ord(c) % 256 để giữ trong giới hạn vocab
            ids = [ord(c) % 256 for c in text_str]
            
            # Truncation
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                
            # Padding
            pad_len = self.max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [0] * pad_len # Padding bằng ID 0
            
            char_ids.append(ids)
            char_masks.append(mask)
            
        return torch.tensor(char_ids, dtype=torch.long), torch.tensor(char_masks, dtype=torch.float32)

class ClickbaitICDDataset(Dataset):
    def __init__(self, titles, leads, labels, tokenizer, char_tokenizer, max_len=256):
        self.titles = titles
        self.leads = leads
        self.labels = labels
        self.tokenizer = tokenizer
        self.char_tokenizer = char_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        lead = str(self.leads[idx])
        label = float(self.labels[idx])

        # Tokenize content riêng biệt
        title_encoding = self.tokenizer(title, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        lead_encoding = self.tokenizer(lead, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        
        # Lấy [0] do return_tensors="pt" sẽ bọc kết quả trong 1 chiều batch
        title_ids = title_encoding['input_ids'][0]
        title_mask = title_encoding['attention_mask'][0].float()
        
        lead_ids = lead_encoding['input_ids'][0]
        lead_mask = lead_encoding['attention_mask'][0].float()

        # Tokenize character cho title
        char_ids_batch, char_mask_batch = self.char_tokenizer.encode([title])
        char_ids = char_ids_batch[0]
        char_mask = char_mask_batch[0]

        return {
            "title_ids": title_ids,
            "title_mask": title_mask,
            "lead_ids": lead_ids,
            "lead_mask": lead_mask,
            "char_ids": char_ids,
            "char_mask": char_mask,
            "labels": torch.tensor([label], dtype=torch.float32)
        }

def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Gom nhóm các từ tiếng Việt lại theo chuẩn
    return word_tokenize(str(text), format="text")

def load_and_preprocess_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    df['title'] = df['title'].fillna("")
    df['lead_paragraph'] = df['lead_paragraph'].fillna("")
    
    # Gom từ (Word Segmentation) bằng underthesea
    df['title'] = df['title'].apply(preprocess_text)
    df['lead_paragraph'] = df['lead_paragraph'].apply(preprocess_text)
    
    # Lập bản đồ nhãn
    label_map = {'non-clickbait': 0, 'clickbait': 1}
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    
    return df['title'].tolist(), df['lead_paragraph'].tolist(), df['label'].tolist()

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            title_ids = batch["title_ids"].to(device)
            title_mask = batch["title_mask"].to(device)
            lead_ids = batch["lead_ids"].to(device)
            lead_mask = batch["lead_mask"].to(device)
            char_ids = batch["char_ids"].to(device)
            char_mask = batch["char_mask"].to(device)
            labels = batch["labels"].to(device)
            
            preds, e_title, e_lead = model(title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask)
            
            loss, _, _ = loss_fn(preds, labels, e_title, e_lead)
            total_loss += loss.item()
            
            # Convert xác suất sang nhãn nhị phân
            pred_labels = (preds >= 0.5).int().cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, precision, recall, f1

def save_metrics(results, output_dir, file_name="test_metrics.csv"):
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame([results])
    file_path = os.path.join(output_dir, file_name)
    results_df.to_csv(file_path, index=False)
    print(f"[+] Metrics saved to: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Attention ICD Model")
    parser.add_argument('--hw_profile', type=str, choices=['rtx3050', 'ada5000'], default='rtx3050', 
                        help="Hardware profile to automatically configure batch size and accumulation steps")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    # Hardware Configuration Profiles
    if args.hw_profile == 'rtx3050':
        BATCH_SIZE = 4
        GRAD_ACCUMULATION = 8
        NUM_WORKERS = 4
        USE_AMP = True
        print("[*] Using Hardware Profile: RTX 3050 (Low VRAM). Batch=4, Grad_Acc=8, AMP=True")
    elif args.hw_profile == 'ada5000':
        BATCH_SIZE = 32
        GRAD_ACCUMULATION = 1
        NUM_WORKERS = 16
        USE_AMP = True
        print("[*] Using Hardware Profile: RTX ADA 5000 (High VRAM). Batch=32, Grad_Acc=1, AMP=True")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device: {device}")

    # Paths
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'result', 'ICD')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Load Tokenizers
    print("[*] Preparing tokenizers...")
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    char_tokenizer = CharTokenizer(max_length=150)

    # 2. Load Datasets
    train_titles, train_leads, train_labels = load_and_preprocess_data(os.path.join(data_dir, 'train.csv'))
    val_titles, val_leads, val_labels = load_and_preprocess_data(os.path.join(data_dir, 'validate.csv'))
    test_titles, test_leads, test_labels = load_and_preprocess_data(os.path.join(data_dir, 'test.csv'))

    train_dataset = ClickbaitICDDataset(train_titles, train_leads, train_labels, tokenizer, char_tokenizer)
    val_dataset = ClickbaitICDDataset(val_titles, val_leads, val_labels, tokenizer, char_tokenizer)
    test_dataset = ClickbaitICDDataset(test_titles, test_leads, test_labels, tokenizer, char_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. Model & Loss
    print("[*] Initializing model & loss function...")
    model = ClickbaitDetectionModel(vocab_size=256, content_model_name=model_name, hidden_size=768, d_c=128)
    model.to(device)
    
    loss_fn = JointLoss(margin=1.0, lambda_weight=0.3)
    loss_fn.to(device)

    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # 5. Training Loop
    best_f1 = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            title_ids = batch["title_ids"].to(device)
            title_mask = batch["title_mask"].to(device)
            lead_ids = batch["lead_ids"].to(device)
            lead_mask = batch["lead_mask"].to(device)
            char_ids = batch["char_ids"].to(device)
            char_mask = batch["char_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                preds, e_title, e_lead = model(title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask)
                loss, bce_loss, L_CL = loss_fn(preds, labels, e_title, e_lead)
                
                # Normalize loss based on gradient accumulation steps
                loss = loss / GRAD_ACCUMULATION
                
            # Backward pass
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUMULATION == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_train_loss += loss.item() * GRAD_ACCUMULATION
            progress_bar.set_postfix({"Loss": f"{loss.item() * GRAD_ACCUMULATION:.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        
        # Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"  [!] New best F1-Score: {best_f1:.4f}. Saving checkpoint...")
            torch.save(model.state_dict(), best_model_path)
            
    # 6. Final Testing
    print("\n" + "="*50)
    print("Training Complete! Running inference on Test Set using Best Model...")
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("[WARNING] Best model checkpoint not found. Using the model from the last epoch.")
        
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, loss_fn, device)
    
    test_results = {
        "Test Loss": test_loss,
        "Accuracy": test_acc,
        "Precision": test_prec,
        "Recall": test_rec,
        "F1-Score": test_f1
    }
    
    print("Final Test Results:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")
        
    save_metrics(test_results, output_dir, file_name="test_metrics.csv")

if __name__ == "__main__":
    main()
