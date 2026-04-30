import os
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

# Add project root to sys.path
# File location: src/experience/ICDv3/find_Lsmoothing_RDrop.py
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(base_dir)

from src.ICD.ICD_Model_v3 import ClickbaitDetectorV3_1, FocalLossWithSmoothing, rdrop_loss
from training.ICD.train_ICD_v3 import load_and_preprocess_data, ClickbaitPairDataset

def evaluate_custom_threshold(model, dataloader, device, threshold=0.45):
    """Đánh giá model trên tập validation với threshold tùy chỉnh."""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features = batch["aux_features"].to(device)
            labels = batch["labels"].numpy()
            
            logits = model(input_ids, attention_mask, aux_features)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.flatten())
            
    probs = 1 / (1 + np.exp(-np.array(all_logits)))
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(np.array(all_labels), preds, average='binary', zero_division=0)
    return f1

def train_and_eval(config, train_loader, val_loader, device, tokenizer):
    """Train model với 1 cấu hình cụ thể và trả về best F1 trên validation."""
    model_name = "vinai/phobert-base"
    model = ClickbaitDetectorV3_1(
        model_name=model_name, 
        sep_token_id=tokenizer.sep_token_id, 
        dropout_rate=0.3
    )
    # Tốc độ training nhanh hơn bằng cách đóng băng một số layer đầu
    model.freeze_backbone_layers(freeze_until=8) 
    model.to(device)
    
    loss_fn = FocalLossWithSmoothing(
        alpha=0.65, 
        gamma=2.0, 
        smoothing=config['label_smoothing']
    )
    
    param_groups = model.get_parameter_groups(lr=5e-5, lr_decay=0.98)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    epochs = 3
    grad_acc = 8
    total_steps = len(train_loader) * epochs // grad_acc
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.15), 
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    best_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features = batch["aux_features"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # R-Drop requires 2 forward passes
                logits1 = model(input_ids, attention_mask, aux_features)
                logits2 = model(input_ids, attention_mask, aux_features)
                
                loss1 = loss_fn(logits1, labels)
                loss2 = loss_fn(logits2, labels)
                task_loss = (loss1 + loss2) / 2
                
                kl_loss = rdrop_loss(logits1, logits2, alpha=config['rdrop_alpha'])
                loss = (task_loss + kl_loss) / grad_acc
                
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_acc == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
        # Evaluate at the end of each epoch
        f1 = evaluate_custom_threshold(model, val_loader, device, threshold=0.45)
        if f1 > best_f1:
            best_f1 = f1
            
    return best_f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    log_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(log_dir, exist_ok=True)
    
    print("[*] Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    train_titles, train_leads, train_labels, train_raw_t, train_raw_l = load_and_preprocess_data(os.path.join(data_dir, 'train.csv'))
    val_titles, val_leads, val_labels, val_raw_t, val_raw_l = load_and_preprocess_data(os.path.join(data_dir, 'validate.csv'))
    
    train_dataset = ClickbaitPairDataset(train_titles, train_leads, train_labels, train_raw_t, train_raw_l, tokenizer, max_len=256)
    val_dataset = ClickbaitPairDataset(val_titles, val_leads, val_labels, val_raw_t, val_raw_l, tokenizer, max_len=256)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    
    # GridSearch parameters
    param_grid = {
        'label_smoothing': [0.0, 0.05, 0.1],
        'rdrop_alpha': [0.5, 1.0, 2.0, 5.0]
    }
    
    grid = list(ParameterGrid(param_grid))
    print(f"[*] Starting Grid Search over {len(grid)} configurations...")
    print("[*] Evaluating using Threshold = 0.45")
    
    best_config = None
    best_overall_f1 = 0.0
    results = []
    
    for i, config in enumerate(grid):
        print(f"\n[{i+1}/{len(grid)}] Training with config: {config}")
        f1 = train_and_eval(config, train_loader, val_loader, device, tokenizer)
        print(f"[*] Result F1 (threshold=0.45): {f1:.4f}")
        
        results.append({**config, 'val_f1': f1})
        
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_config = config
            
    print("\n" + "="*50)
    print(f"[*] BEST CONFIGURATION FOUND:")
    print(f"[*] Label Smoothing: {best_config['label_smoothing']}")
    print(f"[*] R-Drop Alpha:    {best_config['rdrop_alpha']}")
    print(f"[*] Best Val F1:     {best_overall_f1:.4f}")
    print("="*50)
    
    # Save log
    log_file = os.path.join(log_dir, "best_lsmoothing_rdrop_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== GRID SEARCH RESULTS ===\n")
        f.write(f"Threshold used: 0.45\n\n")
        f.write(f"Best Configuration:\n")
        f.write(f"  Label Smoothing: {best_config['label_smoothing']}\n")
        f.write(f"  R-Drop Alpha:    {best_config['rdrop_alpha']}\n")
        f.write(f"  Validation F1:   {best_overall_f1:.4f}\n\n")
        
        f.write("All Results (Sorted by F1-Score):\n")
        for r in sorted(results, key=lambda x: x['val_f1'], reverse=True):
            f.write(f"  LS: {r['label_smoothing']}, R-Drop: {r['rdrop_alpha']} => F1: {r['val_f1']:.4f}\n")
            
    print(f"\n[+] Log saved to: {log_file}")

if __name__ == "__main__":
    main()
