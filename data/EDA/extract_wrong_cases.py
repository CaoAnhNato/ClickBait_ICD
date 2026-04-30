import os
import sys
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.ICD_Model_v3 import ClickbaitDetectorV3_1
from training.ICD.train_ICD_v3 import ClickbaitPairDataset, load_and_preprocess_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Grid Search Parameters
    THRESHOLD = 0.45
    LABEL_SMOOTHING = 0.05
    RDROP_ALPHA = 1.0
    print(f"Params: Threshold={THRESHOLD}, LS={LABEL_SMOOTHING}, R-Drop={RDROP_ALPHA}")
    
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sep_token_id = tokenizer.sep_token_id
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    val_path = os.path.join(data_dir, 'validate.csv')
    
    print("Loading validation data...")
    val_titles, val_leads, val_labels, val_raw_t, val_raw_l = load_and_preprocess_data(val_path)
    
    val_dataset = ClickbaitPairDataset(
        val_titles, val_leads, val_labels, val_raw_t, val_raw_l,
        tokenizer, max_len=256
    )
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print("Loading model...")
    model = ClickbaitDetectorV3_1(
        model_name=model_name,
        sep_token_id=sep_token_id,
        dropout_rate=0.3
    )
    
    best_model_path = os.path.join(base_dir, 'result', 'ICD', 'checkpoints', 'best_model_v3.pth')
    if not os.path.exists(best_model_path):
        print(f"Error: Model checkpoint not found at {best_model_path}")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features = batch["aux_features"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask, aux_features)
            preds = torch.sigmoid(logits).squeeze(-1)
            
            # Since shuffle is False, we can map back to index
            batch_size = input_ids.size(0)
            start_idx = i * 16
            
            for j in range(batch_size):
                idx = start_idx + j
                pred_prob = preds[j].item()
                label = labels[j].item()
                pred_class = 1 if pred_prob >= THRESHOLD else 0
                
                results.append({
                    "index": idx,
                    "title": val_raw_t[idx],
                    "lead": val_raw_l[idx],
                    "score": pred_prob,
                    "pred": pred_class,
                    "label": int(label)
                })
                
    # Chia 4 nhóm
    TP = [r for r in results if r['pred'] == 1 and r['label'] == 1]
    TN = [r for r in results if r['pred'] == 0 and r['label'] == 0]
    FP = [r for r in results if r['pred'] == 1 and r['label'] == 0]
    FN = [r for r in results if r['pred'] == 0 and r['label'] == 1]
    
    print(f"TP: {len(TP)}, TN: {len(TN)}, FP: {len(FP)}, FN: {len(FN)}")
    
    # Sort FP and FN
    # FP: model thinks clickbait (high score). The higher score, the more confident the model was.
    FP_sorted = sorted(FP, key=lambda x: x['score'], reverse=True)
    
    # FN: model thinks non-clickbait (low score). The lower score, the more confident the model was it's 0.
    FN_sorted = sorted(FN, key=lambda x: x['score'], reverse=False)
    
    output_data = {
        "FP": [{"index": r["index"], "title": r["title"], "lead": r["lead"], "score": r["score"]} for r in FP_sorted],
        "FN": [{"index": r["index"], "title": r["title"], "lead": r["lead"], "score": r["score"]} for r in FN_sorted]
    }
    
    output_path = os.path.join(base_dir, 'data', 'EDA', 'wrong_case.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
