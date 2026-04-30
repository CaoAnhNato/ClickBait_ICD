import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Add project root to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(base_dir)

from src.ICD.ICD_Model_v3 import ClickbaitDetectorV3_1
from training.ICD.train_ICD_v3 import load_and_preprocess_data, ClickbaitPairDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")
    
    # Paths
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'result', 'ICD')
    best_model_path = os.path.join(output_dir, 'checkpoints', "best_model_v3.pth")
    log_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(log_dir, exist_ok=True)
    
    if not os.path.exists(best_model_path):
        print(f"[-] Error: Best model not found at {best_model_path}")
        return
        
    print("[*] Loading tokenizer and validation data...")
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    val_titles, val_leads, val_labels, val_raw_t, val_raw_l = load_and_preprocess_data(os.path.join(data_dir, 'validate.csv'))
    
    val_dataset = ClickbaitPairDataset(
        val_titles, val_leads, val_labels, val_raw_t, val_raw_l,
        tokenizer, max_len=256
    )
    # Using larger batch size since we only do inference
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("[*] Loading model...")
    model = ClickbaitDetectorV3_1(
        model_name=model_name,
        sep_token_id=tokenizer.sep_token_id
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    print("[*] Getting logits from validation set...")
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features = batch["aux_features"].to(device)
            labels = batch["labels"].numpy()
            
            logits = model(input_ids, attention_mask, aux_features)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.flatten())
            
    logits_val = np.array(all_logits)
    labels_val = np.array(all_labels)
    
    print("[*] Sweeping thresholds from 0.1 to 0.9...")
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_t = 0.0, 0.5
    best_metrics = {}
    
    probs = 1 / (1 + np.exp(-logits_val))  # Sigmoid function
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels_val, preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_metrics = {
                "threshold": best_t,
                "f1_score": best_f1,
                "precision": precision_score(labels_val, preds, zero_division=0),
                "recall": recall_score(labels_val, preds, zero_division=0),
                "accuracy": accuracy_score(labels_val, preds)
            }
            
    print("\n" + "="*40)
    print(f"[*] BEST THRESHOLD: {best_t:.3f}")
    print(f"[*] BEST F1-SCORE:  {best_f1:.4f}")
    print("="*40)
    print("Detailed Metrics at Best Threshold:")
    for k, v in best_metrics.items():
        if k != "threshold" and k != "f1_score":
            print(f"  {k.capitalize()}: {v:.4f}")
            
    # Save log
    log_file = os.path.join(log_dir, "best_threshold_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== THRESHOLD TUNING RESULTS ===\n")
        f.write(f"Best Threshold: {best_metrics['threshold']:.3f}\n")
        f.write(f"F1-Score:       {best_metrics['f1_score']:.4f}\n")
        f.write(f"Precision:      {best_metrics['precision']:.4f}\n")
        f.write(f"Recall:         {best_metrics['recall']:.4f}\n")
        f.write(f"Accuracy:       {best_metrics['accuracy']:.4f}\n")
        f.write("================================\n")
        
    print(f"\n[+] Log saved to: {log_file}")

if __name__ == "__main__":
    main()
