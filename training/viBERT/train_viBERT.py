#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.tensor([1.0 / 0.6891, 1.0 / 0.3109], dtype=torch.float32).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train.csv")
    parser.add_argument("--val_path", type=str, default="validate.csv")
    parser.add_argument("--test_path", type=str, default="test.csv")
    parser.add_argument("--model_name", type=str, default="FPTAI/vibert-base-cased")
    parser.add_argument("--output_dir", type=str, default="./results_vibert")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["lead_paragraph"] = df["lead_paragraph"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    
    df = df[df["label"] != ""].reset_index(drop=True)
    df = df[(df["title"] != "") | (df["lead_paragraph"] != "")].reset_index(drop=True)
    
    label2id = {"non-clickbait": 0, "clickbait": 1}
    id2label = {0: "non-clickbait", 1: "clickbait"}
    df["label_id"] = df["label"].map(label2id)
    
    return df, label2id, id2label

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    args = parse_args()
    set_seed(args.seed)

    print("[INFO] Đang xử lý data...")
    train_df, label2id, id2label = prepare_data(args.train_path)
    val_df, _, _ = prepare_data(args.val_path)
    test_df, _, _ = prepare_data(args.test_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def process_texts(df):
        titles = []
        leads = []
        for t, l in zip(df["title"], df["lead_paragraph"]):
            t_seg = ViTokenizer.tokenize(t) if t else ""
            l_seg = ViTokenizer.tokenize(l) if l else ""
            titles.append(t_seg)
            leads.append(l_seg)
        return titles, leads

    train_titles, train_leads = process_texts(train_df)
    train_labels = train_df["label_id"].tolist()
    
    val_titles, val_leads = process_texts(val_df)
    val_labels = val_df["label_id"].tolist()
    
    test_titles, test_leads = process_texts(test_df)
    test_labels = test_df["label_id"].tolist()

    train_dataset = ClickbaitDataset(train_titles, train_leads, train_labels, tokenizer, args.max_length)
    val_dataset = ClickbaitDataset(val_titles, val_leads, val_labels, tokenizer, args.max_length)
    test_dataset = ClickbaitDataset(test_titles, test_leads, test_labels, tokenizer, args.max_length)

    print("[INFO] Load model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=400,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(), 
        dataloader_num_workers=12, 
        seed=args.seed
    )

    model.float()
    torch.cuda.empty_cache()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("[INFO] Bắt đầu quá trình hội tụ...")
    trainer.train()

    print("[INFO] Đánh giá trên tập Test...")
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)

    best_model_dir = Path(args.output_dir) / "best_model"
    trainer.save_model(best_model_dir)
    
    with (best_model_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
