import argparse
import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import mlflow
import wandb

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.dataset_icdv6 import ClickbaitDatasetV6
from src.ICD.ICD_Model_v6 import ClickbaitDetectorV6
from src.ICD.losses_v6 import FocalLossWithSmoothing, rdrop_loss

def evaluate(model, dataloader, loss_fn, device, threshold=0.5, return_preds=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids_news"].to(device)
            attention_mask = batch["attention_mask_news"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask)
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

def save_metrics(results, output_dir, file_name="test_metrics_v6.csv"):
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame([results])
    file_path = os.path.join(output_dir, file_name)
    results_df.to_csv(file_path, index=False)
    print(f"[+] Metrics saved to: {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hw', '--hw_profile', type=str, choices=['rtx3050', 'ada5000'], default='rtx3050')
    parser.add_argument('-v', '--variant', type=str, choices=['simple', 'esim'], default='simple')
    parser.add_argument('-e', '--epochs', type=int, default=15)
    parser.add_argument('-lr', '--lr', type=float, default=2e-5)
    parser.add_argument('-ld', '--lr_decay', type=float, default=0.95)
    parser.add_argument('-wr', '--warmup_ratio', type=float, default=0.1)
    parser.add_argument('-fa', '--focal_alpha', type=float, default=0.65)
    parser.add_argument('-fg', '--focal_gamma', type=float, default=2.0)
    parser.add_argument('-ls', '--label_smoothing', type=float, default=0.1)
    parser.add_argument('-ra', '--rdrop_alpha', type=float, default=1.0)
    parser.add_argument('-ml', '--max_len', type=int, default=256)
    parser.add_argument('-p', '--patience', type=int, default=5)
    parser.add_argument('-en', '--experiment_name', type=str, default="ICDv6-Phase0")
    parser.add_argument('-rn', '--run_name', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true', help="Run 1 step per epoch for debugging")
    args = parser.parse_args()

    # Hardware profile
    if args.hw_profile == 'rtx3050':
        BATCH_SIZE = 8
        GRAD_ACCUMULATION = 4
        NUM_WORKERS = 4
        USE_AMP = True
        print("[*] Profile: RTX 3050 (Batch=8, Grad_Acc=4, AMP=True)")
    elif args.hw_profile == 'ada5000':
        BATCH_SIZE = 32
        GRAD_ACCUMULATION = 1
        NUM_WORKERS = 8
        USE_AMP = True
        print("[*] Profile: ADA 5000 (Batch=32, Grad_Acc=1, AMP=True)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    # Paths
    data_dir = os.path.join(base_dir, 'data', 'processed', 'icdv6')
    output_dir = os.path.join(base_dir, 'src', 'experience', 'icdv6')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    mlflow_tracking_dir = os.path.join(base_dir, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
    mlflow.set_experiment(args.experiment_name)
    
    if args.run_name is None:
        args.run_name = f"ICDv6_{args.variant}_RDrop{args.rdrop_alpha}_LS{args.label_smoothing}"

    # Tokenizer
    model_name = "vinai/phobert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("[*] Loading Datasets...")
    train_df = pd.read_parquet(os.path.join(data_dir, 'icdv6_train.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'icdv6_valid.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'icdv6_test.parquet'))
    
    if args.dry_run:
        train_df = train_df.head(100)
        val_df = val_df.head(100)
        test_df = test_df.head(100)

    train_dataset = ClickbaitDatasetV6(train_df, tokenizer, max_len_news=args.max_len)
    val_dataset = ClickbaitDatasetV6(val_df, tokenizer, max_len_news=args.max_len)
    test_dataset = ClickbaitDatasetV6(test_df, tokenizer, max_len_news=args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"[*] Initializing ICDv6 Model (Variant: {args.variant})...")
    model = ClickbaitDetectorV6(model_name=model_name, sep_token_id=tokenizer.eos_token_id, variant=args.variant)
    model.to(device)

    loss_fn = FocalLossWithSmoothing(alpha=args.focal_alpha, gamma=args.focal_gamma, smoothing=args.label_smoothing)

    param_groups = model.get_parameter_groups(lr=args.lr, lr_decay=args.lr_decay)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    total_steps = len(train_loader) * args.epochs // GRAD_ACCUMULATION
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, f"best_model_v6_{args.variant}.pth")

    config_dict = {
        "model_version": f"v6-phase0-{args.variant}",
        "learning_rate": args.lr,
        "batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "rdrop_alpha": args.rdrop_alpha,
        "label_smoothing": args.label_smoothing,
    }

    try:
        wandb_run = wandb.init(
            entity="caoanhdoan130605-ho-chi-minh-city-university-of-industry",
            project="ICD_Model",
            name=args.run_name,
            config=config_dict
        )
        use_wandb = True
    except:
        use_wandb = False
        print("[WARNING] Wandb init failed.")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(config_dict)
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
            
            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids_news"].to(device)
                attention_mask = batch["attention_mask_news"].to(device)
                labels = batch["label"].to(device)
                
                with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
                    if args.rdrop_alpha > 0.0:
                        logits1 = model(input_ids, attention_mask)
                        logits2 = model(input_ids, attention_mask)
                        loss_nll = 0.5 * (loss_fn(logits1, labels) + loss_fn(logits2, labels))
                        loss_rdrop = rdrop_loss(logits1, logits2, alpha=args.rdrop_alpha)
                        loss = loss_nll + loss_rdrop
                    else:
                        logits = model(input_ids, attention_mask)
                        loss = loss_fn(logits, labels)
                    loss = loss / GRAD_ACCUMULATION

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
                
                if args.dry_run and step >= 2:
                    break
                    
            avg_train_loss = total_train_loss / (step + 1)
            current_lr = optimizer.param_groups[-1]['lr']
            
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, loss_fn, device)
            
            print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            metrics = {
                "train/loss": avg_train_loss, "val/loss": val_loss,
                "val/f1": val_f1, "val/acc": val_acc, "lr": current_lr
            }
            mlflow.log_metrics(metrics, step=epoch)
            if use_wandb: wandb.log(metrics, step=epoch)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_metric("best_val_f1", best_f1, step=epoch)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping!")
                    break

        # Testing
        print("\nTraining Complete! Running Test Set...")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            
        test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds = evaluate(
            model, test_loader, loss_fn, device, return_preds=True
        )
        
        print(f"Test F1: {test_f1:.4f} | Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f}")
        
        report_path = os.path.join(output_dir, f"report_{args.variant}.txt")
        with open(report_path, "w") as f:
            f.write(classification_report(test_labels, test_preds, target_names=['non-clickbait', 'clickbait']))
        
        mlflow.log_artifact(report_path, artifact_path="evaluation")
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="model")
            
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
