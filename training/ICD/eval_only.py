import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.ICD_Model_v3 import ClickbaitDetectorV3_1, FocalLossWithSmoothing, rdrop_loss
from training.ICD.train_ICD_v3 import load_and_preprocess_data, ClickbaitPairDataset

def evaluate_custom(model, dataloader, loss_fn, device, threshold=0.45, rdrop_alpha=1.0):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aux_features = batch["aux_features"].to(device)
            labels = batch["labels"].to(device)
            
            # Standard forward pass
            logits = model(input_ids, attention_mask, aux_features)
            loss = loss_fn(logits, labels)
            
            # Calculate R-Drop loss (will be 0 in eval mode since dropout is disabled, but included for completeness)
            rd_loss = rdrop_loss(logits, logits, alpha=rdrop_alpha)
            loss = loss + rd_loss
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits)
            pred_labels = (preds >= threshold).int().cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, precision, recall, f1, all_labels, all_preds

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'result', 'ICD')
    best_model_path = os.path.join(output_dir, 'checkpoints', "best_model_v3.pth")
    
    if not os.path.exists(best_model_path):
        print(f"[-] Error: Best model not found at {best_model_path}")
        return
        
    print("[*] Loading tokenizer and test data...")
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    test_titles, test_leads, test_labels, test_raw_t, test_raw_l = load_and_preprocess_data(os.path.join(data_dir, 'test.csv'))
    
    test_dataset = ClickbaitPairDataset(
        test_titles, test_leads, test_labels, test_raw_t, test_raw_l,
        tokenizer, max_len=256
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("[*] Loading model weights from:", best_model_path)
    model = ClickbaitDetectorV3_1(
        model_name=model_name,
        sep_token_id=tokenizer.sep_token_id
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    
    # Configuration
    threshold = 0.45
    label_smoothing = 0.05
    rdrop_alpha = 1.0
    
    print(f"[*] Settings - Threshold: {threshold}, Label Smoothing: {label_smoothing}, R-Drop Alpha: {rdrop_alpha}")
    loss_fn = FocalLossWithSmoothing(smoothing=label_smoothing)
    
    print("[*] Running evaluation...")
    test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds = evaluate_custom(
        model, test_loader, loss_fn, device, threshold=threshold, rdrop_alpha=rdrop_alpha
    )
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall: {test_rec:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    
    print("\nClassification Report:")
    class_report = classification_report(test_labels, test_preds, target_names=['non-clickbait', 'clickbait'])
    print(class_report)
    
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report:\n")
        f.write(class_report)
    print(f"[+] Classification report saved to: {report_path}")
        
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['non-clickbait', 'clickbait'], 
                yticklabels=['non-clickbait', 'clickbait'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    main()
