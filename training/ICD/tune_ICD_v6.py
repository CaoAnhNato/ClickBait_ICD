import argparse
import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import optuna
import mlflow
import wandb
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.dataset_icdv6 import ClickbaitDatasetV6
from src.ICD.ICD_Model_v6 import ClickbaitDetectorV6
from src.ICD.losses_v6 import WeightedFocalLossV6, rdrop_loss

def evaluate(model, dataloader, loss_fn, device, use_pattern_tags=True):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids_news"].to(device)
            attention_mask = batch["attention_mask_news"].to(device)
            labels = batch["label"].to(device)
            pattern_tags = batch["pattern_tags"].to(device)
            
            logits = model(input_ids, attention_mask, pattern_tags=pattern_tags)
            loss = loss_fn(logits, labels, pattern_tags)
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits)
            pred_labels = (preds >= 0.5).int().cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(labels.cpu().numpy())
            
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    return total_loss / len(dataloader), f1

def objective(trial, args, train_df, val_df, tokenizer, device):
    # Hyperparameters to tune
    w_hardnews = trial.suggest_float("w_hardnews", 1.0, 3.0)
    w_shock = trial.suggest_float("w_shock", 1.0, 2.0)
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    rdrop_alpha = trial.suggest_float("rdrop_alpha", 0.5, 2.0)
    
    # Simple setup for HPO
    batch_size = 16 if args.hw_profile == 'ada5000' else 4
    grad_acc = 2 if args.hw_profile == 'ada5000' else 8
    
    train_dataset = ClickbaitDatasetV6(train_df, tokenizer, max_len_news=args.max_len, use_pattern_tags=True)
    val_dataset = ClickbaitDatasetV6(val_df, tokenizer, max_len_news=args.max_len, use_pattern_tags=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = ClickbaitDetectorV6(variant=args.variant, use_residual=args.use_pattern_residual).to(device)
    if args.freeze_layers > 0:
        model.freeze_backbone_layers(freeze_until=args.freeze_layers)
        
    loss_fn = WeightedFocalLossV6(hardnews_weight=w_hardnews, shock_weight=w_shock)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_val_f1 = 0.0
    
    # Trial logging
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        
        for epoch in range(args.num_epochs):
            model.train()
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids_news"].to(device)
                attention_mask = batch["attention_mask_news"].to(device)
                labels = batch["label"].to(device)
                pattern_tags = batch["pattern_tags"].to(device)
                
                logits1 = model(input_ids, attention_mask, pattern_tags=pattern_tags)
                logits2 = model(input_ids, attention_mask, pattern_tags=pattern_tags)
                
                loss_nll = 0.5 * (loss_fn(logits1, labels, pattern_tags) + loss_fn(logits2, labels, pattern_tags))
                loss_rdrop = rdrop_loss(logits1, logits2, alpha=rdrop_alpha)
                loss = (loss_nll + loss_rdrop) / grad_acc
                
                loss.backward()
                if (step + 1) % grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if args.dry_run and step >= 2: break
            
            val_loss, val_f1 = evaluate(model, val_loader, loss_fn, device)
            best_val_f1 = max(best_val_f1, val_f1)
            
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            if args.dry_run: break
            
    return best_val_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hw', '--hw_profile', type=str, choices=['rtx3050', 'ada5000'], default='ada5000')
    parser.add_argument('-v', '--variant', type=str, default='simple')
    parser.add_argument('-n', '--n_trials', type=int, default=20)
    parser.add_argument('-e', '--num_epochs', type=int, default=5) # Giảm epoch cho HPO
    parser.add_argument('-ml', '--max_len', type=int, default=256)
    parser.add_argument('-fl', '--freeze_layers', type=int, default=8)
    parser.add_argument('--use_pattern_residual', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    
    data_dir = os.path.join(base_dir, 'data', 'processed', 'icdv6')
    train_df = pd.read_parquet(os.path.join(data_dir, 'icdv6_train.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'icdv6_valid.parquet'))
    
    if args.dry_run:
        train_df = train_df.head(100)
        val_df = val_df.head(100)

    mlflow.set_experiment(f"ICDv6_HPO_{args.variant}")
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, args, train_df, val_df, tokenizer, device), n_trials=args.n_trials)

    print("\n[+] Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
