import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from cleanlab.filter import find_label_issues
from tqdm import tqdm

# Thêm đường dẫn root để import src và training
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.ICD_Model_v3 import ClickbaitDetectorV3_1
from training.ICD.train_ICD_v3 import ClickbaitPairDataset, load_and_preprocess_data, extract_aux_features

def train_and_predict_fold(fold_idx, train_idx, val_idx, full_data, device, tokenizer, model_name, sep_token_id):
    """Huấn luyện trên train_idx và predict trên val_idx."""
    titles_seg, leads_seg, labels, raw_titles, raw_leads = full_data
    
    # Tạo datasets
    train_dataset = ClickbaitPairDataset(
        [titles_seg[i] for i in train_idx],
        [leads_seg[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [raw_titles[i] for i in train_idx],
        [raw_leads[i] for i in train_idx],
        tokenizer
    )
    val_dataset = ClickbaitPairDataset(
        [titles_seg[i] for i in val_idx],
        [leads_seg[i] for i in val_idx],
        [labels[i] for i in val_idx],
        [raw_titles[i] for i in val_idx],
        [raw_leads[i] for i in val_idx],
        tokenizer
    )
    
    # Tối ưu hóa cấu hình DataLoader cho 28 Cores và 40GB RAM
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=24, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=24, pin_memory=True, prefetch_factor=2)
    
    # Init model
    model = ClickbaitDetectorV3_1(model_name=model_name, sep_token_id=sep_token_id)
    
    # Freeze lower layers to speed up since we just need a decent predictor for noise finding
    if hasattr(model, 'freeze_backbone_layers'):
        model.freeze_backbone_layers(freeze_until=8)
    
    model.to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # PyTorch AMP (Automatic Mixed Precision) cho RTX ADA 6000
    scaler = torch.amp.GradScaler('cuda')
    
    # Train nhanh (vài epochs là đủ để cleanlab bắt được noise)
    epochs = 4
    print(f"\n--- Training Fold {fold_idx+1} ---")
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            aux_features = batch["aux_features"].to(device, non_blocking=True)
            labels_batch = batch["labels"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, aux_features)
                loss = loss_fn(logits, labels_batch)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    # Predict
    model.eval()
    fold_probs = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            aux_features = batch["aux_features"].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, aux_features)
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            fold_probs.extend(probs)
            
    return np.array(fold_probs)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sep_token_id = tokenizer.sep_token_id
    
    # Load data (Gộp train và validate để tìm noise trên tập lớn hơn)
    data_dir = os.path.join(base_dir, 'data', 'processed')
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv = os.path.join(data_dir, 'validate.csv')
    
    print("Loading datasets...")
    t1, l1, lab1, rt1, rl1 = load_and_preprocess_data(train_csv)
    t2, l2, lab2, rt2, rl2 = load_and_preprocess_data(val_csv)
    
    # Merge
    full_titles_seg = t1 + t2
    full_leads_seg = l1 + l2
    full_labels = lab1 + lab2
    full_raw_titles = rt1 + rt2
    full_raw_leads = rl1 + rl2
    
    full_data = (full_titles_seg, full_leads_seg, full_labels, full_raw_titles, full_raw_leads)
    n_samples = len(full_labels)
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros(n_samples)
    
    indices = np.arange(n_samples)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        fold_probs = train_and_predict_fold(
            fold_idx, train_idx, val_idx, full_data, 
            device, tokenizer, model_name, sep_token_id
        )
        all_probs[val_idx] = fold_probs
        
    # Chuẩn bị input cho cleanlab
    # p_neg = 1 - p_pos, p_pos = p
    probs_out = np.stack([1 - all_probs, all_probs], axis=1)
    labels_arr = np.array(full_labels)
    
    print("\n--- Running Cleanlab to find label issues ---")
    label_issues_indices = find_label_issues(
        labels=labels_arr,
        pred_probs=probs_out,
        return_indices_ranked_by='self_confidence',
        n_jobs=24  # Tối ưu hóa cho 28 CPU Cores
    )
    
    print(f"Found {len(label_issues_indices)} potential label issues.")
    
    # Lưu kết quả
    output_results = []
    for idx in label_issues_indices:
        output_results.append({
            "index": int(idx),
            "original_label": int(labels_arr[idx]),
            "predicted_prob_clickbait": float(all_probs[idx]),
            "title": full_raw_titles[idx],
            "lead": full_raw_leads[idx]
        })
        
    output_path = os.path.join(base_dir, 'data', 'EDA', 'label_issues.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(output_results, f, ensure_ascii=False, indent=4)
        
    print(f"Label issues saved to {output_path}")

if __name__ == "__main__":
    main()
