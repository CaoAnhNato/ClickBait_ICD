"""
Training Script cho ICD Model v3.1 - Iteration 3
==================================================
Frozen backbone + aggressive training for fast convergence

Key changes from iteration 2:
- REMOVED R-Drop (halved effective training)
- FROZEN bottom 8 PhoBERT layers → focus on top 4 + classifier
- Higher LR (5e-5) with 10x classifier multiplier
- No label smoothing (slowed convergence)
- Single forward pass per step (not double)

Chạy: conda run -n MLE python training/ICD/train_ICD_v3.py --hw_profile rtx3050 --epochs 3
"""

import argparse
import os
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import py_vncorenlp
from tqdm import tqdm
import numpy as np
import sys
import mlflow
import wandb

# Thêm đường dẫn root
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.ICD_Model_v3 import ClickbaitDetectorV3_1, FocalLossWithSmoothing, rdrop_loss


# ============================================================================
# Auxiliary Feature Extraction
# ============================================================================
def extract_aux_features(title: str, lead: str) -> list:
    """
    Trích xuất đặc trưng ngôn ngữ cho clickbait detection.
    Ref: "Click it or Leave it" (arXiv:2602.18171)
    """
    title = str(title) if not pd.isna(title) else ""
    lead = str(lead) if not pd.isna(lead) else ""
    
    exclamation_count = title.count('!')
    question_count = title.count('?')
    ellipsis_count = title.count('…') + title.count('...')
    
    alpha_chars = [c for c in title if c.isalpha()]
    uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
    
    lead_len = max(len(lead), 1)
    title_lead_ratio = len(title) / lead_len
    
    numeral_count = len(re.findall(r'\d+', title))
    
    return [exclamation_count, question_count, ellipsis_count, 
            uppercase_ratio, title_lead_ratio, numeral_count]


# ============================================================================
# Dataset
# ============================================================================
class ClickbaitPairDataset(Dataset):
    """Sentence-pair dataset cho PhoBERT."""
    def __init__(self, titles, leads, labels, raw_titles, raw_leads,
                 tokenizer, max_len=256):
        self.titles = titles
        self.leads = leads
        self.labels = labels
        self.raw_titles = raw_titles
        self.raw_leads = raw_leads
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        title = str(self.titles[idx])
        lead = str(self.leads[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            title, lead,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0).float()
        
        raw_title = str(self.raw_titles[idx])
        raw_lead = str(self.raw_leads[idx])
        aux_feats = extract_aux_features(raw_title, raw_lead)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "aux_features": torch.tensor(aux_feats, dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.float32)
        }


# ============================================================================
# Preprocessing
# ============================================================================
# Khởi tạo VnCoreNLP theo cơ chế Lazy Loading để tránh lỗi khi import trên các môi trường khác nhau
vncorenlp_path = os.path.join(os.getcwd(), 'vncorenlp_data')
_rdrsegmenter = None

def get_segmenter():
    global _rdrsegmenter
    if _rdrsegmenter is None:
        if not os.path.exists(vncorenlp_path):
            os.makedirs(vncorenlp_path, exist_ok=True)
            py_vncorenlp.download_model(save_dir=vncorenlp_path)
        _rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)
    return _rdrsegmenter

def format_vncorenlp(res):
    return " ".join(res).strip()

def preprocess_text(text):
    """Word segmentation bằng py_vncorenlp."""
    if pd.isna(text):
        return ""
    segmenter = get_segmenter()
    return format_vncorenlp(segmenter.word_segment(str(text)))


def load_and_preprocess_data(data_path):
    """Load CSV và apply word segmentation."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    df['title'] = df['title'].fillna("")
    df['lead_paragraph'] = df['lead_paragraph'].fillna("")
    
    raw_titles = df['title'].tolist()
    raw_leads = df['lead_paragraph'].tolist()
    
    df['title_seg'] = df['title'].apply(preprocess_text)
    df['lead_seg'] = df['lead_paragraph'].apply(preprocess_text)
    
    label_map = {'non-clickbait': 0, 'clickbait': 1}
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    
    return (df['title_seg'].tolist(), df['lead_seg'].tolist(), 
            df['label'].tolist(), raw_titles, raw_leads)


# ============================================================================
# Evaluation
# ============================================================================
def evaluate(model, dataloader, loss_fn, device, threshold=0.5, return_preds=False):
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
            
            logits = model(input_ids, attention_mask, aux_features)
            loss = loss_fn(logits, labels)
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
    
    if return_preds:
        return avg_loss, acc, precision, recall, f1, all_labels, all_preds
    return avg_loss, acc, precision, recall, f1


def save_metrics(results, output_dir, file_name="test_metrics_v3.csv"):
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame([results])
    file_path = os.path.join(output_dir, file_name)
    results_df.to_csv(file_path, index=False)
    print(f"[+] Metrics saved to: {file_path}")


# ============================================================================
# Main Training
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train ICD Model v3.1 - Iteration 3: Frozen Backbone")
    parser.add_argument('-hw', '--hw_profile', type=str, choices=['rtx3050', 'ada5000', 'rtxa4000'], 
                        default='rtx3050')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-lr', '--lr', type=float, default=5e-5)
    parser.add_argument('-ld', '--lr_decay', type=float, default=0.98)
    parser.add_argument('-wr', '--warmup_ratio', type=float, default=0.15)
    parser.add_argument('-fa', '--focal_alpha', type=float, default=0.65)
    parser.add_argument('-fg', '--focal_gamma', type=float, default=2.0)
    parser.add_argument('-fl', '--freeze_layers', type=int, default=8, help="Freeze bottom N PhoBERT layers")
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help="Classification threshold")
    parser.add_argument('-ls', '--label_smoothing', type=float, default=0.0, help="Label smoothing value")
    parser.add_argument('-ra', '--rdrop_alpha', type=float, default=0.0, help="R-Drop Alpha weight")
    parser.add_argument('-ml', '--max_len', type=int, default=256)
    parser.add_argument('-p', '--patience', type=int, default=5)
    parser.add_argument('-en', '--experiment_name', type=str, default="ICD-ClickbaitDetection")
    parser.add_argument('-rn', '--run_name', type=str, default=None)
    parser.add_argument('-wn', '--wandb_run_name', type=str, default=None)
    parser.add_argument('-nw', '--no_wandb', action='store_true')
    args = parser.parse_args()
    
    # Hardware profiles
    if args.hw_profile == 'rtx3050':
        BATCH_SIZE = 4
        GRAD_ACCUMULATION = 8
        NUM_WORKERS = 4
        USE_AMP = True
        print("[*] Hardware Profile: RTX 3050 (4GB). Batch=4, Grad_Acc=8, AMP=True")
    elif args.hw_profile == 'ada5000':
        BATCH_SIZE = 32
        GRAD_ACCUMULATION = 1
        NUM_WORKERS = 16
        USE_AMP = True
        print("[*] Hardware Profile: RTX ADA 5000 (32GB). Batch=32, Grad_Acc=1, AMP=True")
    elif args.hw_profile == 'rtxa4000':
        BATCH_SIZE = 16
        GRAD_ACCUMULATION = 2
        NUM_WORKERS = 16
        USE_AMP = True
        print("[*] Hardware Profile: RTX A4000 (16GB). Batch=16, Grad_Acc=2, AMP=True")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")
    
    # Paths
    data_dir = os.path.join(base_dir, 'data', 'processed', 'cleaned')
    output_dir = os.path.join(base_dir, 'result', 'ICD')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # MLflow setup
    mlflow_tracking_dir = os.path.join(base_dir, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
    mlflow.set_experiment(args.experiment_name)
    
    if args.run_name is None:
        args.run_name = f"ICD-v3.1_Frozen{args.freeze_layers}_lr{args.lr}"
    if args.wandb_run_name is None:
        args.wandb_run_name = args.run_name
    
    # ========================================================================
    # 1. Tokenizer
    # ========================================================================
    print("[*] Loading PhoBERT tokenizer...")
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sep_token_id = tokenizer.sep_token_id  # </s> token id
    print(f"[*] Separator token ID: {sep_token_id}")
    
    # ========================================================================
    # 2. Load & Preprocess Data
    # ========================================================================
    train_titles, train_leads, train_labels, train_raw_t, train_raw_l = \
        load_and_preprocess_data(os.path.join(data_dir, 'train_clean.csv'))
    val_titles, val_leads, val_labels, val_raw_t, val_raw_l = \
        load_and_preprocess_data(os.path.join(data_dir, 'validate_clean.csv'))
    test_titles, test_leads, test_labels, test_raw_t, test_raw_l = \
        load_and_preprocess_data(os.path.join(data_dir, 'test_clean.csv'))
    
    train_dataset = ClickbaitPairDataset(
        train_titles, train_leads, train_labels, train_raw_t, train_raw_l,
        tokenizer, max_len=args.max_len
    )
    val_dataset = ClickbaitPairDataset(
        val_titles, val_leads, val_labels, val_raw_t, val_raw_l,
        tokenizer, max_len=args.max_len
    )
    test_dataset = ClickbaitPairDataset(
        test_titles, test_leads, test_labels, test_raw_t, test_raw_l,
        tokenizer, max_len=args.max_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    # ========================================================================
    # 3. Model & Loss
    # ========================================================================
    print("[*] Initializing ClickbaitDetectorV3_1...")
    model = ClickbaitDetectorV3_1(
        model_name=model_name,
        sep_token_id=sep_token_id,
        dropout_rate=0.3
    )
    
    # Freeze bottom layers for faster convergence
    if args.freeze_layers > 0:
        model.freeze_backbone_layers(freeze_until=args.freeze_layers)
    
    model.to(device)
    
    loss_fn = FocalLossWithSmoothing(
        alpha=args.focal_alpha, 
        gamma=args.focal_gamma,
        smoothing=args.label_smoothing
    )
    
    # ========================================================================
    # 4. Optimizer with Layer-wise LR Decay
    # ========================================================================
    param_groups = model.get_parameter_groups(lr=args.lr, lr_decay=args.lr_decay)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    # Linear warmup + cosine decay
    total_steps = len(train_loader) * args.epochs // GRAD_ACCUMULATION
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"[*] Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Early Stopping
    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, "best_model_v3.pth")
    
    # ========================================================================
    # Config logging
    # ========================================================================
    config_dict = {
        "model_version": "v3.1-iter3",
        "architecture": "SegmentAwarePool-ESIM-FrozenBackbone",
        "backbone": model_name,
        "hidden_size": 768,
        "classifier_features": "CLS+title_pool+lead_pool+diff+prod+aux",
        "classifier_input_dim": 5 * 768 + 6,
        "freeze_layers": args.freeze_layers,
        "learning_rate": args.lr,
        "lr_decay": args.lr_decay,
        "warmup_ratio": args.warmup_ratio,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "grad_accumulation": GRAD_ACCUMULATION,
        "max_seq_len": args.max_len,
        "epochs": args.epochs,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "weight_decay": 0.01,
        "scheduler": "cosine_with_warmup",
        "hw_profile": args.hw_profile,
        "use_amp": USE_AMP,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }
    
    # WandB init (optional)
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = wandb.init(
                entity="caoanhdoan130605-ho-chi-minh-city-university-of-industry",
                project="ICD_Model",
                name=args.wandb_run_name,
                config=config_dict
            )
        except Exception as e:
            print(f"[WARNING] WandB init failed: {e}. Continuing without WandB.")
            use_wandb = False
    
    # ========================================================================
    # 5. Training Loop (single forward, no R-Drop)
    # ========================================================================
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(config_dict)
        mlflow.set_tags({
            "dataset": "ViClickbait",
            "task": "binary_classification",
            "framework": "PyTorch",
            "architecture": "ICD-SegmentAware-v3.1-iter3",
        })
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
            
            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                aux_features = batch["aux_features"].to(device)
                labels = batch["labels"].to(device)
                
                with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
                    if args.rdrop_alpha > 0.0:
                        logits1 = model(input_ids, attention_mask, aux_features)
                        logits2 = model(input_ids, attention_mask, aux_features)
                        loss_nll1 = loss_fn(logits1, labels)
                        loss_nll2 = loss_fn(logits2, labels)
                        loss_nll = 0.5 * (loss_nll1 + loss_nll2)
                        loss_rdrop = rdrop_loss(logits1, logits2, alpha=args.rdrop_alpha)
                        loss = loss_nll + loss_rdrop
                    else:
                        logits = model(input_ids, attention_mask, aux_features)
                        loss = loss_fn(logits, labels)
                    loss = loss / GRAD_ACCUMULATION
                
                # Backward
                scaler.scale(loss).backward()
                
                if (step + 1) % GRAD_ACCUMULATION == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                total_train_loss += loss.item() * GRAD_ACCUMULATION
                progress_bar.set_postfix({"Loss": f"{loss.item() * GRAD_ACCUMULATION:.4f}"})
            
            avg_train_loss = total_train_loss / len(train_loader)
            current_lr = optimizer.param_groups[-1]['lr']
            
            # Validation
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
                model, val_loader, loss_fn, device, threshold=args.threshold
            )
            
            print(f"\nEpoch {epoch}/{args.epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
            print(f"  Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | "
                  f"Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")
            
            # Log metrics
            metrics_dict = {
                "train/loss": avg_train_loss,
                "train/learning_rate": current_lr,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/precision": val_prec,
                "val/recall": val_rec,
                "val/f1_score": val_f1,
            }
            mlflow.log_metrics(metrics_dict, step=epoch)
            if use_wandb:
                wandb.log(metrics_dict, step=epoch)
            
            # Best model checkpoint
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                print(f"  [!] New best F1: {best_f1:.4f}. Saving checkpoint...")
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_metric("best_val_f1", best_f1, step=epoch)
            else:
                patience_counter += 1
                print(f"  [EarlyStopping] No improvement ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"\n[!] Early stopping at epoch {epoch}. Best F1: {best_f1:.4f}")
                    mlflow.log_metric("early_stopped_epoch", epoch)
                    if use_wandb:
                        wandb.log({"early_stopped_epoch": epoch}, step=epoch)
                    break
        
        # ====================================================================
        # 6. Final Testing
        # ====================================================================
        print("\n" + "=" * 50)
        print("Training Complete! Running inference on Test Set...")
        
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        else:
            print("[WARNING] Best model not found. Using last epoch model.")
        
        test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds = evaluate(
            model, test_loader, loss_fn, device, threshold=args.threshold, return_preds=True
        )
        
        test_results = {
            "Test Loss": test_loss,
            "Accuracy": test_acc,
            "Precision": test_prec,
            "Recall": test_rec,
            "F1-Score": test_f1
        }
        
        print("Final Test Results:")
        for k, v in test_results.items():
            print(f"  {k}: {v:.4f}")
            
        # Generate Classification Report
        print("\nClassification Report:")
        class_report = classification_report(test_labels, test_preds, target_names=['non-clickbait', 'clickbait'])
        print(class_report)
        
        # Save Classification Report to file
        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Classification Report:\n")
            f.write(class_report)
        print(f"[+] Classification report saved to: {report_path}")
            
        # Generate Confusion Matrix
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['non-clickbait', 'clickbait'], 
                    yticklabels=['non-clickbait', 'clickbait'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save Confusion Matrix to file
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Confusion matrix saved to: {cm_path}")
        
        test_metrics_dict = {
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "test/precision": test_prec,
            "test/recall": test_rec,
            "test/f1_score": test_f1,
        }
        mlflow.log_metrics(test_metrics_dict)
        mlflow.log_artifact(report_path, artifact_path="evaluation")
        mlflow.log_artifact(cm_path, artifact_path="evaluation")
        if use_wandb:
            wandb.log(test_metrics_dict)
        
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="model")
            if use_wandb and wandb_run:
                wandb_artifact = wandb.Artifact(f"best_model_v3.1_{wandb_run.id}", type="model")
                wandb_artifact.add_file(best_model_path)
                wandb_run.log_artifact(wandb_artifact)
        
        save_metrics(test_results, output_dir, file_name="test_metrics_v3.csv")
        
        print(f"\n[*] MLflow Run ID: {mlflow.active_run().info.run_id}")
        if use_wandb and wandb_run:
            print(f"[*] WandB Run URL: {wandb_run.url}")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
