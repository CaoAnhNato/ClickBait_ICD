import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import py_vncorenlp
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_vncorenlp() -> py_vncorenlp.VnCoreNLP:
    """Load VnCoreNLP, copying to a space-free path if necessary."""
    import shutil

    vncorenlp_path = os.path.join(os.getcwd(), "vncorenlp_data")
    if " " in vncorenlp_path:
        safe_path = os.path.expanduser("~/.cache/vncorenlp_data")
        if not os.path.exists(safe_path):
            os.makedirs(safe_path, exist_ok=True)
            if os.path.exists(vncorenlp_path):
                shutil.copytree(vncorenlp_path, safe_path, dirs_exist_ok=True)
            else:
                py_vncorenlp.download_model(save_dir=safe_path)
        vncorenlp_path = safe_path
    elif not os.path.exists(vncorenlp_path):
        os.makedirs(vncorenlp_path, exist_ok=True)
        py_vncorenlp.download_model(save_dir=vncorenlp_path)

    orig_cwd  = os.getcwd()
    segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)
    os.chdir(orig_cwd)
    return segmenter

def get_error_type(y_true, y_pred):
    if y_true == 1 and y_pred == 1:
        return 'TP'
    elif y_true == 0 and y_pred == 0:
        return 'TN'
    elif y_true == 0 and y_pred == 1:
        return 'FP'
    elif y_true == 1 and y_pred == 0:
        return 'FN'
    return 'UNKNOWN'

def evaluate_and_dump(df, segmenter, tokenizer, model, device, max_length=256, output_path="predictions.csv", batch_size=32):
    label_map = {"non-clickbait": 0, "clickbait": 1}
    
    if 'id' not in df.columns:
        df['id'] = df.index

    results = []
    model.eval()
    
    data = []
    for _, row in df.iterrows():
        id_val = row['id']
        title_raw = str(row['title']) if pd.notna(row['title']) else ""
        lead_raw = str(row['lead_paragraph']) if pd.notna(row['lead_paragraph']) else ""
        
        title_seg = " ".join(segmenter.word_segment(title_raw)).strip() if title_raw else ""
        lead_seg = " ".join(segmenter.word_segment(lead_raw)).strip() if lead_raw else ""
        
        text = f"{title_raw} {lead_raw}".strip()
        y_true = label_map.get(row['label'], 0) if 'label' in row else 0
        
        title_tokens = len(tokenizer.encode(title_seg, add_special_tokens=False)) if title_seg else 0
        lead_tokens = len(tokenizer.encode(lead_seg, add_special_tokens=False)) if lead_seg else 0
        
        data.append({
            "id": id_val,
            "title_raw": title_raw,
            "lead_raw": lead_raw,
            "title_seg": title_seg,
            "lead_seg": lead_seg,
            "text": text,
            "y_true": y_true,
            "title_tokens": title_tokens,
            "lead_tokens": lead_tokens
        })
        
    all_probs = []
    all_num_tokens = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        title_segs = [b['title_seg'] for b in batch]
        lead_segs = [b['lead_seg'] for b in batch]
        
        inputs = tokenizer(
            title_segs,
            lead_segs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        for j in range(len(batch)):
            input_ids = inputs['input_ids'][j]
            num_t = input_ids.shape[0] - (input_ids == tokenizer.pad_token_id).sum().item()
            all_num_tokens.append(num_t)
            
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            
    for i, item in enumerate(data):
        probs = all_probs[i]
        num_tokens = all_num_tokens[i]
        
        prob_non_clickbait = float(probs[0])
        prob_clickbait = float(probs[1])
        y_pred = int(np.argmax(probs))
        confidence = float(np.max(probs))
        margin = float(abs(prob_clickbait - prob_non_clickbait))
        ent = float(entropy(probs, base=2))
        y_true = item['y_true']
        is_correct = bool(y_true == y_pred)
        error_type = get_error_type(y_true, y_pred)
        
        results.append({
            "id": item['id'],
            "title": item['title_raw'],
            "lead_paragraph": item['lead_raw'],
            "text": item['text'],
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_non_clickbait": prob_non_clickbait,
            "prob_clickbait": prob_clickbait,
            "confidence": confidence,
            "margin": margin,
            "entropy": ent,
            "is_correct": is_correct,
            "error_type": error_type,
            "num_tokens": num_tokens,
            "title_tokens": item['title_tokens'],
            "lead_tokens": item['lead_tokens']
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved predictions to {output_path}")

def main():
    val_path = "data/processed/cleaned/validate_best_cleaned.csv"
    test_path = "data/processed/cleaned/test_best_cleaned.csv"
    model_dir = "result/results_phoBERT/phobert_base/best_model"
    output_dir = "EDA_erro_case/PhoBert_base"
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading VnCoreNLP...")
    segmenter = setup_vncorenlp()
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    
    print("Loading datasets...")
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    val_output = os.path.join(output_dir, "dev_predictions.csv")
    test_output = os.path.join(output_dir, "test_predictions.csv")
    
    print("Evaluating Validation Set...")
    evaluate_and_dump(val_df, segmenter, tokenizer, model, device, output_path=val_output)
    
    print("Evaluating Test Set...")
    evaluate_and_dump(test_df, segmenter, tokenizer, model, device, output_path=test_output)
    
    print("Done!")

if __name__ == "__main__":
    main()
