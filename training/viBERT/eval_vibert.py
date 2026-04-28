#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
from pyvi import ViTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset

class ClickbaitDataset(Dataset):
    def __init__(self, texts_a, texts_b, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts_a,
            texts_b,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    model_path = "./results_vibert/best_model"
    test_path = "test.csv"
    result_csv = "training/videberta/result.csv"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Không tìm thấy model tại '{model_path}'. Vui lòng train model trước!")
        return

    print(f"[INFO] Load test data from {test_path}...")
    df = pd.read_csv(test_path)
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["lead_paragraph"] = df["lead_paragraph"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    
    df = df[df["label"] != ""].reset_index(drop=True)
    df = df[(df["title"] != "") | (df["lead_paragraph"] != "")].reset_index(drop=True)
    
    label2id = {"non-clickbait": 0, "clickbait": 1}
    df["label_id"] = df["label"].map(label2id)

    print("[INFO] Pre-processing with Pyvi ViTokenizer...")
    titles = []
    leads = []
    for t, l in zip(df["title"], df["lead_paragraph"]):
        titles.append(ViTokenizer.tokenize(t) if t else "")
        leads.append(ViTokenizer.tokenize(l) if l else "")
    labels = df["label_id"].tolist()

    print(f"[INFO] Load model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    test_dataset = ClickbaitDataset(titles, leads, labels, tokenizer, max_len=256)

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    print("[INFO] Đang đánh giá (Evaluating)...")
    results = trainer.evaluate(test_dataset)
    
    acc = results['eval_accuracy']
    prec = results['eval_precision']
    rec = results['eval_recall']
    f1 = results['eval_f1']
    
    print(f"\n[RESULTS] Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1-Macro={f1:.4f}\n")
    
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    file_exists = os.path.isfile(result_csv)
    
    with open(result_csv, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("model,accuracy,precision,recall,f1_score\n")
        # Ghi kết quả vào file
        f.write(f"viBERT,{acc},{prec},{rec},{f1}\n")
        
    print(f"[INFO] Đã thêm kết quả vào {result_csv}")

if __name__ == "__main__":
    main()
