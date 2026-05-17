"""
Tune_LLRD.py
Fine-tune vinai/phobert-base-v2 for Vietnamese Clickbait Detection using Layer-wise Learning Rate Decay.

Output (per run) written to result/results_phobert_base_v2/llrd_base/:
  ├── best_model/                  # Best checkpoint weights + tokenizer
  ├── classification_report.csv    # Per-class + macro/weighted metrics
  └── config.json                  # Run hyperparameters + final metrics
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import py_vncorenlp

from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# VnCoreNLP helper
# ──────────────────────────────────────────────────────────────────────────────

def setup_vncorenlp() -> py_vncorenlp.VnCoreNLP:
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

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1     = f1_score(labels, predictions, average="macro")
    f1_cb  = f1_score(labels, predictions, average="binary", pos_label=1)
    acc    = accuracy_score(labels, predictions)
    prec   = precision_score(labels, predictions, average="macro")
    rec    = recall_score(labels, predictions, average="macro")
    
    return {
        "f1_macro": f1,
        "f1_clickbait": f1_cb,
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    }

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune vinai/phobert-base-v2 using LLRD."
    )
    parser.add_argument("-e",  "--epochs",       type=int,   default=10,  help="Number of training epochs.")
    parser.add_argument("-b",  "--batch-size",   type=int,   default=8,   help="Batch size per device.")
    parser.add_argument("-ga", "--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("-ld", "--layer-decay",  type=float, default=0.95, help="Layer-wise learning rate decay factor.")
    parser.add_argument("-m",  "--max-length",   type=int,   default=256, help="Maximum sequence length.")
    parser.add_argument("-fl", "--freeze-layers", type=int,  default=0,   help="Number of encoder layers to freeze.")
    parser.add_argument("-p",  "--patience",     type=int,   default=5,   help="Patience for early stopping.")
    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────────
    train_path = "data/processed/cleaned/train_best_cleaned.csv"
    val_path   = "data/processed/cleaned/validate_best_cleaned.csv"
    test_path  = "data/processed/cleaned/test_best_cleaned.csv"
    output_dir = "result/results_phobert_base_v2/llrd_base"
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # ── VnCoreNLP ──────────────────────────────────────────────────────────────
    print(">>> Loading VnCoreNLP...")
    segmenter = setup_vncorenlp()

    def segment_text(text: str) -> str:
        if pd.isna(text) or text == "":
            return ""
        return " ".join(segmenter.word_segment(str(text))).strip()

    # ── Data ───────────────────────────────────────────────────────────────────
    print(">>> Loading datasets...")
    label_map = {"non-clickbait": 0, "clickbait": 1}

    def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["title"]          = df["title"].fillna("").apply(segment_text)
        df["lead_paragraph"] = df["lead_paragraph"].fillna("").apply(segment_text)
        df["label"]          = df["label"].map(label_map)
        return df

    print(">>> Segmenting text (this may take a while)...")
    train_df = preprocess_df(pd.read_csv(train_path))
    val_df   = preprocess_df(pd.read_csv(val_path))
    test_df  = preprocess_df(pd.read_csv(test_path))
    
    cols = ["title", "lead_paragraph", "label"]
    train_ds = Dataset.from_pandas(train_df[cols].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[cols].reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df[cols].reset_index(drop=True))

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    model_name = "vinai/phobert-base-v2"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["title"],
            examples["lead_paragraph"],
            truncation=True,
            max_length=args.max_length,
        )

    print(">>> Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_fn, batched=True).rename_column("label", "labels")
    val_ds   = val_ds.map(tokenize_fn,   batched=True).rename_column("label", "labels")
    test_ds  = test_ds.map(tokenize_fn,  batched=True).rename_column("label", "labels")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f">>> Initializing AutoModelForSequenceClassification: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if args.freeze_layers > 0:
        print(f">>> Freezing embeddings and first {args.freeze_layers} encoder layers...")
        for name, param in model.named_parameters():
            if name.startswith("roberta.embeddings."):
                param.requires_grad = False
            elif name.startswith("roberta.encoder.layer."):
                layer_idx = int(name.split(".")[3])
                if layer_idx < args.freeze_layers:
                    param.requires_grad = False

    # ── Training Arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        remove_unused_columns=True,
    )

    # ── Optimizer with LLRD ────────────────────────────────────────────────────
    def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, layer_decay):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = []
        num_layers = getattr(model.config, "num_hidden_layers", 12)
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if name.startswith("roberta.embeddings"):
                depth = num_layers + 1
            elif name.startswith("roberta.encoder.layer."):
                layer_idx = int(name.split(".")[3])
                depth = num_layers - layer_idx
            else:
                depth = 0
                
            lr = learning_rate * (layer_decay ** depth)
            wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            
            optimizer_grouped_parameters.append({
                "params": [param],
                "lr": lr,
                "weight_decay": wd
            })
            
        return optimizer_grouped_parameters

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, 
        learning_rate=args.learning_rate, 
        weight_decay=training_args.weight_decay, 
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(">>> Starting training...")
    train_result = trainer.train()

    # ── Evaluate on Test Set ───────────────────────────────────────────────────
    print("\n>>> Evaluating on Test Set...")
    predictions_output = trainer.predict(test_ds)
    preds  = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    target_names = ["non-clickbait", "clickbait"]
    report_dict  = classification_report(
        labels, preds, target_names=target_names, digits=4, output_dict=True
    )
    report_str = classification_report(
        labels, preds, target_names=target_names, digits=4
    )
    print("\nClassification Report:\n", report_str)

    # Check for improvement before saving --------------------------------------
    config_path = os.path.join(output_dir, "config.json")
    new_f1 = report_dict.get("macro avg", {}).get("f1-score", 0.0)
    old_f1 = 0.0
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                old_config = json.load(f)
                old_f1 = old_config.get("test_metrics", {}).get("f1-score", 0.0)
        except Exception:
            pass

    if new_f1 > old_f1:
        print(f"\n>>> Improvement detected: {old_f1:.4f} -> {new_f1:.4f}. Saving results...")

        report_df = pd.DataFrame(report_dict).transpose()
        report_csv_path = os.path.join(output_dir, "classification_report.csv")
        report_df.to_csv(report_csv_path)
        print(f">>> Classification report saved → {report_csv_path}")

        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        print(f">>> Best model saved → {best_model_dir}")

        final_metrics = {
            k: float(v) for k, v in report_dict.get("macro avg", {}).items()
            if k in ("precision", "recall", "f1-score")
        }
        final_metrics["accuracy"] = float(report_dict.get("accuracy", 0.0))
        final_metrics["f1_clickbait"] = float(report_dict.get("clickbait", {}).get("f1-score", 0.0))
        
        config_data = {
            "model": model_name,
            "method": "LLRD",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "layer_decay": args.layer_decay,
            "max_length": args.max_length,
            "freeze_layers": args.freeze_layers,
            "early_stopping_patience": args.patience,
            "test_metrics": final_metrics,
            "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0), 2),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        print(f">>> Config saved → {config_path}")
    else:
        print(f"\n>>> No improvement detected: {new_f1:.4f} <= {old_f1:.4f}. Skipping save.")

    print("\n>>> Done.")

if __name__ == "__main__":
    main()
