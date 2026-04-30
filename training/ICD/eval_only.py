import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.ICD_Model_v3 import ClickbaitDetectorV3_1
from training.ICD.train_ICD_v3 import load_and_preprocess_data, ClickbaitPairDataset, evaluate

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
    
    print("[*] Loading model...")
    model = ClickbaitDetectorV3_1(
        model_name=model_name,
        sep_token_id=tokenizer.sep_token_id
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    
    # Use BCEWithLogitsLoss as dummy for evaluate function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    print("[*] Running evaluation...")
    test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, loss_fn, device, return_preds=True
    )
    
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
