import argparse
import os
import pandas as pd
from underthesea import word_tokenize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from torch.utils.data import Dataset
import numpy as np

class ClickbaitDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Underthesea word_tokenize for Vietnamese as per RULE
    return word_tokenize(str(text), format="text")

def load_and_preprocess_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Fill NaN values
    df['title'] = df['title'].fillna("")
    df['lead_paragraph'] = df['lead_paragraph'].fillna("")
    
    # Word segmentation
    df['title'] = df['title'].apply(preprocess_text)
    df['lead_paragraph'] = df['lead_paragraph'].apply(preprocess_text)
    
    # Encode label: 'non-clickbait' -> 0, 'clickbait' -> 1
    label_map = {'non-clickbait': 0, 'clickbait': 1}
    df['label'] = df['label'].map(label_map)
    
    # Filter out rows with unmapped labels if any (just in case)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    titles = df['title'].tolist()
    leads = df['lead_paragraph'].tolist()
    labels = df['label'].tolist()
    return titles, leads, labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT-large for Clickbait Detection.")
    parser.add_argument('-e', '--epochs', type=int, default=5, help="Number of training epochs. Default is 5.")
    parser.add_argument('-b', '--batch-size', type=int, default=4, help="Batch size per device. Default is 4 (optimized for memory).")
    parser.add_argument('-ga', '--gradient-accumulation', type=int, default=8, help="Number of gradient accumulation steps. Default is 8.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5, help="Learning rate. Default is 1e-5.")
    parser.add_argument('-m', '--max-length', type=int, default=256, help="Maximum sequence length. Default is 256.")
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.01, help="Weight decay. Default is 0.01.")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of workers for data loading. Default is 4.")
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'result', 'results_phoBERT')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading and preprocessing datasets...")
    train_titles, train_leads, train_labels = load_and_preprocess_data(os.path.join(data_dir, 'train.csv'))
    val_titles, val_leads, val_labels = load_and_preprocess_data(os.path.join(data_dir, 'validate.csv'))
    test_titles, test_leads, test_labels = load_and_preprocess_data(os.path.join(data_dir, 'test.csv'))

    model_name = "vinai/phobert-large"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Tokenizing datasets...")
    # Concatenating titles and leads via tokenizer(text_a, text_b)
    train_encodings = tokenizer(train_titles, train_leads, truncation=True, padding="max_length", max_length=args.max_length)
    val_encodings = tokenizer(val_titles, val_leads, truncation=True, padding="max_length", max_length=args.max_length)
    test_encodings = tokenizer(test_titles, test_leads, truncation=True, padding="max_length", max_length=args.max_length)

    train_dataset = ClickbaitDataset(train_encodings, train_labels)
    val_dataset = ClickbaitDataset(val_encodings, val_labels)
    test_dataset = ClickbaitDataset(test_encodings, test_labels)

    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    import transformers
    from packaging import version
    if version.parse(transformers.__version__) >= version.parse("4.41.0"):
        strategy_kwargs = {"eval_strategy": "epoch"}
    else:
        strategy_kwargs = {"evaluation_strategy": "epoch"}

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'checkpoints_phobert_large'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, 'logs_phobert_large'),
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=args.workers,
        **strategy_kwargs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Get raw predictions for classification report
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    
    # Save classification report
    report = classification_report(labels, preds, target_names=['non-clickbait', 'clickbait'], digits=4)
    report_path = os.path.join(output_dir, 'classification_report_phobert-large.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Save test results
    results_df = pd.DataFrame([test_results])
    results_path = os.path.join(output_dir, 'test_metrics_phobert-large.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")
    print(test_results)

    # Save final model
    trainer.save_model(os.path.join(output_dir, "best_model_phobert_large"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model_phobert_large"))
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    main()
