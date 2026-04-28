import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer
from underthesea import word_tokenize
from tqdm import tqdm
import json
import numpy as np
import sys
import mlflow
import mlflow.pytorch
import wandb

# Thêm đường dẫn root vào sys.path để có thể import từ src
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.ICD_Model import ClickbaitDetectionModel, JointLoss

# ============================================================================
# [v2] Vietnamese-Aware CharTokenizer
# ============================================================================
class CharTokenizer:
    """
    Bộ tokenizer tự xây dựng cho cấp độ ký tự, hỗ trợ tiếng Việt.
    
    [v2] Khắc phục vấn đề encoding tiếng Việt:
    - v1 sử dụng ord(c) % 256 → gây collision giữa các ký tự Unicode
      (ví dụ: ord('ấ')=7845, ord('đ')=273, khi %256 → collision)
    - v2 xây dựng vocabulary đầy đủ cho các ký tự tiếng Việt phổ biến,
      bao gồm các diacritical marks và ký tự đặc biệt clickbait.
    """
    # Các ký tự tiếng Việt (nguyên âm có dấu + phụ âm đặc biệt)
    VIETNAMESE_CHARS = (
        "aàáảãạăằắẳẵặâầấẩẫậ"
        "eèéẻẽẹêềếểễệ"
        "iìíỉĩị"
        "oòóỏõọôồốổỗộơờớởỡợ"
        "uùúủũụưừứửữự"
        "yỳýỷỹỵ"
        "đ"
        "AÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
        "EÈÉẺẼẸÊỀẾỂỄỆ"
        "IÌÍỈĨỊ"
        "OÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
        "UÙÚỦŨỤƯỪỨỬỮỰ"
        "YỲÝỶỸỴ"
        "Đ"
    )
    
    # Ký tự đặc biệt thường thấy trong clickbait titles
    SPECIAL_CHARS = "!?\"'()[]{}…:;.,/-_@#$%^&*+=<>~`|\\0123456789"
    
    # Chữ cái Latin cơ bản (không dấu)
    LATIN_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def __init__(self, max_length=150):
        self.max_length = max_length
        
        # Xây dựng vocab: PAD=0, UNK=1, SPACE=2, rồi đến các chars
        self.char2id = {'<PAD>': 0, '<UNK>': 1, ' ': 2}
        idx = 3
        
        # Thêm Latin chars
        for c in self.LATIN_CHARS:
            if c not in self.char2id:
                self.char2id[c] = idx
                idx += 1
        
        # Thêm Vietnamese chars
        for c in self.VIETNAMESE_CHARS:
            if c not in self.char2id:
                self.char2id[c] = idx
                idx += 1
        
        # Thêm special chars
        for c in self.SPECIAL_CHARS:
            if c not in self.char2id:
                self.char2id[c] = idx
                idx += 1
        
        self.vocab_size = idx
        self.pad_id = 0
        self.unk_id = 1
        
    def encode(self, texts):
        char_ids = []
        char_masks = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            text_str = str(text)
            
            # [v2] Sử dụng vocabulary lookup thay vì ord() % 256
            ids = [self.char2id.get(c, self.unk_id) for c in text_str]
            
            # Truncation
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                
            # Padding
            pad_len = self.max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [self.pad_id] * pad_len
            
            char_ids.append(ids)
            char_masks.append(mask)
            
        return torch.tensor(char_ids, dtype=torch.long), torch.tensor(char_masks, dtype=torch.float32)

class ClickbaitICDDataset(Dataset):
    def __init__(self, titles, leads, labels, tokenizer, char_tokenizer, max_len=256):
        self.titles = titles
        self.leads = leads
        self.labels = labels
        self.tokenizer = tokenizer
        self.char_tokenizer = char_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        lead = str(self.leads[idx])
        label = float(self.labels[idx])

        # Tokenize content riêng biệt
        title_encoding = self.tokenizer(title, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        lead_encoding = self.tokenizer(lead, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        
        # Lấy [0] do return_tensors="pt" sẽ bọc kết quả trong 1 chiều batch
        title_ids = title_encoding['input_ids'][0]
        title_mask = title_encoding['attention_mask'][0].float()
        
        lead_ids = lead_encoding['input_ids'][0]
        lead_mask = lead_encoding['attention_mask'][0].float()

        # Tokenize character cho title
        char_ids_batch, char_mask_batch = self.char_tokenizer.encode([title])
        char_ids = char_ids_batch[0]
        char_mask = char_mask_batch[0]

        return {
            "title_ids": title_ids,
            "title_mask": title_mask,
            "lead_ids": lead_ids,
            "lead_mask": lead_mask,
            "char_ids": char_ids,
            "char_mask": char_mask,
            "labels": torch.tensor([label], dtype=torch.float32)
        }

def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Gom nhóm các từ tiếng Việt lại theo chuẩn
    return word_tokenize(str(text), format="text")

def load_and_preprocess_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    df['title'] = df['title'].fillna("")
    df['lead_paragraph'] = df['lead_paragraph'].fillna("")
    
    # Gom từ (Word Segmentation) bằng underthesea
    df['title'] = df['title'].apply(preprocess_text)
    df['lead_paragraph'] = df['lead_paragraph'].apply(preprocess_text)
    
    # Lập bản đồ nhãn
    label_map = {'non-clickbait': 0, 'clickbait': 1}
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    
    return df['title'].tolist(), df['lead_paragraph'].tolist(), df['label'].tolist()

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            title_ids = batch["title_ids"].to(device)
            title_mask = batch["title_mask"].to(device)
            lead_ids = batch["lead_ids"].to(device)
            lead_mask = batch["lead_mask"].to(device)
            char_ids = batch["char_ids"].to(device)
            char_mask = batch["char_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits, e_title, e_lead = model(title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask)
            
            loss, _, _ = loss_fn(logits, labels, e_title, e_lead)
            total_loss += loss.item()
            
            # Convert xác suất sang nhãn nhị phân
            preds = torch.sigmoid(logits)
            pred_labels = (preds >= 0.5).int().cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, precision, recall, f1

def save_metrics(results, output_dir, file_name="test_metrics.csv"):
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame([results])
    file_path = os.path.join(output_dir, file_name)
    results_df.to_csv(file_path, index=False)
    print(f"[+] Metrics saved to: {file_path}")

# ============================================================================
# [v2] Early Stopping
# ============================================================================
class EarlyStopping:
    """
    Early stopping để tránh overfitting.
    Dừng training khi val_f1 không cải thiện sau 'patience' epochs liên tiếp.
    """
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Attention ICD Model v2")
    parser.add_argument('--hw_profile', type=str, choices=['rtx3050', 'ada5000', 'rtxa4000'], default='rtx3050', 
                        help="Hardware profile to automatically configure batch size and accumulation steps")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--margin', type=float, default=0.5, help="Contrastive loss margin")
    parser.add_argument('--lambda_cl', type=float, default=0.3, help="Contrastive loss weight")
    parser.add_argument('--focal_alpha', type=float, default=0.6, help="Focal loss alpha (class weight)")
    parser.add_argument('--focal_gamma', type=float, default=2.0, help="Focal loss gamma (focusing param)")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--experiment_name', type=str, default="ICD-ClickbaitDetection", help="MLflow experiment name")
    parser.add_argument('--run_name', type=str, default=None, help="MLflow run name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="WandB run name (experiment name)")
    args = parser.parse_args()

    # Hardware Configuration Profiles
    if args.hw_profile == 'rtx3050':
        BATCH_SIZE = 4
        GRAD_ACCUMULATION = 8
        NUM_WORKERS = 4
        USE_AMP = True
        print("[*] Using Hardware Profile: RTX 3050 (Low VRAM). Batch=4, Grad_Acc=8, AMP=True")
    elif args.hw_profile == 'ada5000':
        BATCH_SIZE = 32
        GRAD_ACCUMULATION = 1
        NUM_WORKERS = 16
        USE_AMP = True
        print("[*] Using Hardware Profile: RTX ADA 5000 (High VRAM 32GB). Batch=32, Grad_Acc=1, AMP=True")
    elif args.hw_profile == 'rtxa4000':
        BATCH_SIZE = 16
        GRAD_ACCUMULATION = 2
        NUM_WORKERS = 16
        USE_AMP = True
        print("[*] Using Hardware Profile: RTX A4000 (Mid VRAM 16GB). Batch=16, Grad_Acc=2 (Eff_Batch=32), AMP=True")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device: {device}")

    # Paths
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'result', 'ICD')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================================================
    # [v2] MLflow Setup
    # ========================================================================
    mlflow_tracking_dir = os.path.join(base_dir, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
    mlflow.set_experiment(args.experiment_name)
    
    # Tạo run name tự động nếu không chỉ định
    if args.run_name is None:
        args.run_name = f"ICD-v2_lr{args.lr}_m{args.margin}_λ{args.lambda_cl}_α{args.focal_alpha}_γ{args.focal_gamma}"
    if args.wandb_run_name is None:
        args.wandb_run_name = args.run_name

    # 1. Load Tokenizers
    print("[*] Preparing tokenizers...")
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # [v2] Sử dụng Vietnamese-aware CharTokenizer
    char_tokenizer = CharTokenizer(max_length=150)
    print(f"[*] CharTokenizer vocab_size: {char_tokenizer.vocab_size}")

    # 2. Load Datasets
    train_titles, train_leads, train_labels = load_and_preprocess_data(os.path.join(data_dir, 'train.csv'))
    val_titles, val_leads, val_labels = load_and_preprocess_data(os.path.join(data_dir, 'validate.csv'))
    test_titles, test_leads, test_labels = load_and_preprocess_data(os.path.join(data_dir, 'test.csv'))

    train_dataset = ClickbaitICDDataset(train_titles, train_leads, train_labels, tokenizer, char_tokenizer)
    val_dataset = ClickbaitICDDataset(val_titles, val_leads, val_labels, tokenizer, char_tokenizer)
    test_dataset = ClickbaitICDDataset(test_titles, test_leads, test_labels, tokenizer, char_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. Model & Loss
    print("[*] Initializing model & loss function (v2)...")
    # [v2] d_c=64 (giảm từ 128), vocab_size từ CharTokenizer
    model = ClickbaitDetectionModel(
        vocab_size=char_tokenizer.vocab_size,
        content_model_name=model_name, 
        hidden_size=768, 
        d_c=64
    )
    model.to(device)
    
    # [v2] JointLoss với FocalLoss + improved Contrastive Loss
    loss_fn = JointLoss(
        margin=args.margin, 
        lambda_weight=args.lambda_cl,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    loss_fn.to(device)

    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # [v2] Learning Rate Scheduler: CosineAnnealingWarmRestarts
    # T_0=5: restart mỗi 5 epochs, T_mult=2: chu kỳ tiếp theo dài gấp đôi
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # [v2] Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

    # ========================================================================
    # 5. Training Loop with MLflow Tracking
    # ========================================================================
    best_f1 = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_model_v2.pth")
    
    config_dict = {
        "model_version": "v2",
        "backbone": model_name,
        "d_c": 64,
        "char_vocab_size": char_tokenizer.vocab_size,
        "hidden_size": 768,
        "classifier_input_dim": 7 * 768,
        "learning_rate": args.lr,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "grad_accumulation": GRAD_ACCUMULATION,
        "epochs": args.epochs,
        "margin": args.margin,
        "lambda_cl": args.lambda_cl,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "weight_decay": 0.01,
        "scheduler": "CosineAnnealingWarmRestarts",
        "scheduler_T0": 5,
        "early_stopping_patience": args.patience,
        "hw_profile": args.hw_profile,
        "use_amp": USE_AMP,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }

    # Initialize WandB
    wandb_run = wandb.init(
        entity="caoanhdoan130605-ho-chi-minh-city-university-of-industry",
        project="ICD_Model",
        name=args.wandb_run_name,
        config=config_dict
    )
    
    with mlflow.start_run(run_name=args.run_name):
        # Log hyperparameters
        mlflow.log_params(config_dict)
        
        # Log dataset info as tags
        mlflow.set_tags({
            "dataset": "ViClickbait",
            "task": "binary_classification",
            "framework": "PyTorch",
            "architecture": "ICD-MultiAttention-v2",
        })
    
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_train_loss = 0.0
            total_cls_loss = 0.0
            total_cl_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
            
            for step, batch in enumerate(progress_bar):
                # Move to device
                title_ids = batch["title_ids"].to(device)
                title_mask = batch["title_mask"].to(device)
                lead_ids = batch["lead_ids"].to(device)
                lead_mask = batch["lead_mask"].to(device)
                char_ids = batch["char_ids"].to(device)
                char_mask = batch["char_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass with AMP (using BFloat16 for better stability and avoiding overflow)
                with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16):
                    logits, e_title, e_lead = model(title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask)
                    loss, cls_loss, L_CL = loss_fn(logits, labels, e_title, e_lead)
                    
                    # Normalize loss based on gradient accumulation steps
                    loss = loss / GRAD_ACCUMULATION
                    
                # Backward pass
                scaler.scale(loss).backward()
                
                if (step + 1) % GRAD_ACCUMULATION == 0 or (step + 1) == len(train_loader):
                    # [v2] Gradient clipping để tránh exploding gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                total_train_loss += loss.item() * GRAD_ACCUMULATION
                total_cls_loss += cls_loss.item()
                total_cl_loss += L_CL.item()
                progress_bar.set_postfix({"Loss": f"{loss.item() * GRAD_ACCUMULATION:.4f}"})
            
            # [v2] Step scheduler after each epoch
            scheduler.step()
                
            avg_train_loss = total_train_loss / len(train_loader)
            avg_cls_loss = total_cls_loss / len(train_loader)
            avg_cl_loss = total_cl_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Validation
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, loss_fn, device)
            
            print(f"Epoch {epoch}/{args.epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} (Focal: {avg_cls_loss:.4f}, CL: {avg_cl_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
            print(f"  Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")
            
            # [v2] MLflow & WandB: Log metrics per epoch
            metrics_dict = {
                "train/total_loss": avg_train_loss,
                "train/focal_loss": avg_cls_loss,
                "train/contrastive_loss": avg_cl_loss,
                "train/learning_rate": current_lr,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/precision": val_prec,
                "val/recall": val_rec,
                "val/f1_score": val_f1,
            }
            mlflow.log_metrics(metrics_dict, step=epoch)
            wandb.log(metrics_dict, step=epoch)
            
            # Log contrastive loss temperature
            if hasattr(loss_fn, 'log_temperature'):
                temperature = torch.exp(loss_fn.log_temperature).item()
                mlflow.log_metric("train/cl_temperature", temperature, step=epoch)
                wandb.log({"train/cl_temperature": temperature}, step=epoch)
            
            # Checkpoint
            if val_f1 > best_f1:
                best_f1 = val_f1
                print(f"  [!] New best F1-Score: {best_f1:.4f}. Saving checkpoint...")
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_metric("best_val_f1", best_f1, step=epoch)
            
            # [v2] Early Stopping check
            early_stopping(val_f1)
            if early_stopping.should_stop:
                print(f"\n[!] Early stopping triggered at epoch {epoch}. Best F1: {best_f1:.4f}")
                mlflow.log_metric("early_stopped_epoch", epoch)
                wandb.log({"early_stopped_epoch": epoch}, step=epoch)
                break
                
        # 6. Final Testing
        print("\n" + "="*50)
        print("Training Complete! Running inference on Test Set using Best Model...")
        
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
        else:
            print("[WARNING] Best model checkpoint not found. Using the model from the last epoch.")
            
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, loss_fn, device)
        
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
        
        # [v2] MLflow & WandB: Log final test metrics
        test_metrics_dict = {
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "test/precision": test_prec,
            "test/recall": test_rec,
            "test/f1_score": test_f1,
        }
        mlflow.log_metrics(test_metrics_dict)
        wandb.log(test_metrics_dict)
        
        # [v2] MLflow: Log model artifact
        mlflow.log_artifact(best_model_path, artifact_path="model")
        
        # [v2] WandB: Log model artifact
        wandb_artifact = wandb.Artifact(f"best_model_{wandb_run.id}", type="model")
        wandb_artifact.add_file(best_model_path)
        wandb_run.log_artifact(wandb_artifact)
            
        save_metrics(test_results, output_dir, file_name="test_metrics_v2.csv")
        
        print(f"\n[*] MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"[*] View results: mlflow ui --backend-store-uri file://{mlflow_tracking_dir}")
        print(f"[*] WandB Run URL: {wandb_run.url}")

    wandb.finish()

if __name__ == "__main__":
    main()
