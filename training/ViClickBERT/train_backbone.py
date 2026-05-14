"""
train_backbone.py — ViClickBERT Domain-Adaptive Pre-Training (DAPT)
====================================================================
Pipeline:
  1. Load Nato1306/ICDv7 (title + sapo) → word-segment via VnCoreNLP → tokenize
  2. Split: 10k held-out | 10k valid | rest → train

  ── Sequential Training (LoRA DAPT) ──────────────────────────────
  3. Phase 1: MLM 15% dynamic masking + SOP, LoRA r=32/α=64
     LR=2e-4, cosine decay, no warmup, 15 epochs, effective-BS=512
     Eval PPL+MLM-Acc mỗi 2000 steps → save best by eval_loss
  4. merge_and_unload() → merged backbone
  5. Phase 2: tiếp tục tune merged model, 8 epochs (train+valid)

  ── Baseline (Full Fine-Tune, không LoRA) ────────────────────────
  6. Load vanilla PhoBERT-base-v2 → MLM full fine-tune 8 epochs
     Cùng data, cùng hypers → để so sánh công bằng

  ── Intrinsic Comparison ─────────────────────────────────────────
  7. Evaluate cả 2 model trên held-out 10k
     In bảng so sánh PPL + MLM-Acc: Sequential DAPT vs Baseline

Hardware target: 20-core CPU | 48 GB RAM | ADA-6000 48 GB GPU | 150 GB disk
Env: MLE  |  Labels: 0=Non-Clickbait  1=Clickbait
"""

import os, math, random, shutil, logging
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import py_vncorenlp
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
MODEL_NAME        = "vinai/phobert-base-v2"
HF_DATASET        = "Nato1306/ICDv7"
OUTPUT_DIR        = "result/ViClickBERT"
PHASE1_DIR        = os.path.join(OUTPUT_DIR, "phase1_dapt")
PHASE2_DIR        = os.path.join(OUTPUT_DIR, "phase2_dapt")
MERGED_DIR        = os.path.join(OUTPUT_DIR, "merged_backbone")
BASELINE_DIR      = os.path.join(OUTPUT_DIR, "baseline_vanilla")

# Cache for word-segmented dataset (skip re-segmentation on subsequent runs)
SEGMENTED_CACHE   = "data/processed/ViClickBERT/segmented"

HELD_OUT_SIZE     = 10_000
VALID_SIZE        = 10_000
TRAIN_LIMIT       = 500_000       # cap train split (None = use all remaining rows)
SEED              = 42
MAX_LENGTH        = 192           # 256→192: saves ~25% attention VRAM (title+sapo fits well)

# Phase-1 hypers
P1_EPOCHS         = 15
P1_LR             = 2e-4
P1_BATCH          = 48            # 64→48 per_device: saves ~25% activation VRAM
P1_GRAD_ACC       = 11            # eff. BS = 48×11 = 528 ≈ 512 (unchanged semantics)
P1_EVAL_STEPS     = 2000
P1_MLM_PROB       = 0.15
P1_SOP_RATIO      = 0.5          # fraction of pairs where sapo is swapped

# Phase-2 hypers (fine-tune backbone on same corpus, 8 epochs)
P2_EPOCHS         = 8
P2_LR             = 1e-4
P2_BATCH          = 48            # 64→48, same rationale as Phase-1
P2_GRAD_ACC       = 11            # eff. BS = 48×11 = 528 ≈ 512
P2_EVAL_STEPS     = 2000

# LoRA (Phase-1)
LORA_R            = 32
LORA_ALPHA        = 64
LORA_DROPOUT      = 0.05
LORA_TARGET       = ["query", "key", "value", "dense",
                      "intermediate.dense", "output.dense"]
MODULES_TO_SAVE   = ["lm_head"]

# ─────────────────────────────────────────────────────────
# 1. VnCoreNLP helper
# ─────────────────────────────────────────────────────────

def _get_safe_vncorenlp_path() -> str:
    raw = os.path.join(os.getcwd(), "vncorenlp_data")
    if " " in raw:
        safe = os.path.expanduser("~/.cache/vncorenlp_data")
        if not os.path.exists(safe):
            os.makedirs(safe, exist_ok=True)
            if os.path.exists(raw):
                log.info("Path has spaces → copying vncorenlp_data to %s", safe)
                shutil.copytree(raw, safe, dirs_exist_ok=True)
            else:
                log.info("Downloading VnCoreNLP to %s …", safe)
                py_vncorenlp.download_model(save_dir=safe)
        return safe
    if not os.path.exists(raw):
        os.makedirs(raw, exist_ok=True)
        py_vncorenlp.download_model(save_dir=raw)
    return raw


def load_segmenter():
    path = _get_safe_vncorenlp_path()
    orig_cwd = os.getcwd()
    log.info("Loading VnCoreNLP from %s …", path)
    seg = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=path)
    os.chdir(orig_cwd)
    return seg


def segment(seg, text: str) -> str:
    if not text or (isinstance(text, float) and math.isnan(text)):
        return ""
    try:
        return " ".join(seg.word_segment(str(text).strip())).strip()
    except Exception:
        return str(text).strip()

# ─────────────────────────────────────────────────────────
# 2. Dataset loading & splitting
# ─────────────────────────────────────────────────────────

def load_and_split(seg) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ICDv7, word-segment (VnCoreNLP), split into train/valid/held-out.

    Cache mechanism
    ───────────────
    After the first run the fully-segmented dataset is saved to
    `SEGMENTED_CACHE` (HuggingFace Arrow format).  Subsequent runs
    detect the cache folder and skip the expensive segment step.
    Delete the folder manually if you want a fresh re-segmentation.
    """

    # ── 1. Try loading from cache ─────────────────────────────────────────────
    if os.path.isdir(SEGMENTED_CACHE):
        log.info("[CACHE HIT] Loading pre-segmented dataset from '%s' …", SEGMENTED_CACHE)
        ds = Dataset.load_from_disk(SEGMENTED_CACHE)
        log.info("Cached dataset: %d rows — columns: %s", len(ds), ds.column_names)
    else:
        # ── 2. Download raw dataset ───────────────────────────────────────────
        log.info("Loading %s from HuggingFace …", HF_DATASET)
        ds = load_dataset(HF_DATASET, split="train")
        log.info("Dataset size: %d rows — columns: %s", len(ds), ds.column_names)

        # ── 3. Word-segment via pandas in main process ────────────────────────
        # VnCoreNLP uses jnius (Java bridge): jnius objects are NOT picklable.
        # datasets.map() always spawns subprocesses → dill tries to serialize
        # the closure capturing `seg` → TypeError: no default __reduce__.
        # Fix: pandas.progress_apply() runs entirely in the main process.
        # ─────────────────────────────────────────────────────────────────────
        log.info("Word-segmenting with VnCoreNLP in main process (no subprocess) …")
        df = ds.to_pandas()

        from tqdm import tqdm
        tqdm.pandas(desc="Segmenting title")
        df["title_seg"] = df["title"].progress_apply(lambda t: segment(seg, t))

        tqdm.pandas(desc="Segmenting sapo ")
        df["sapo_seg"]  = df["sapo"].progress_apply(lambda s: segment(seg, s))

        # Keep only segmented columns to save RAM
        ds = Dataset.from_pandas(df[["title_seg", "sapo_seg"]], preserve_index=False)
        del df
        log.info("Word-segmentation complete.")

        # ── 4. Save to cache ──────────────────────────────────────────────────
        os.makedirs(SEGMENTED_CACHE, exist_ok=True)
        ds.save_to_disk(SEGMENTED_CACHE)
        log.info("[CACHE SAVED] Segmented dataset saved to '%s'.", SEGMENTED_CACHE)
        log.info("Next run will skip word-segmentation and load from cache directly.")

    # ── 5. Shuffle + split ────────────────────────────────────────────────────
    ds = ds.shuffle(seed=SEED)
    held_out_ds = ds.select(range(HELD_OUT_SIZE))
    valid_ds    = ds.select(range(HELD_OUT_SIZE, HELD_OUT_SIZE + VALID_SIZE))
    train_ds    = ds.select(range(HELD_OUT_SIZE + VALID_SIZE, len(ds)))

    # ── 6. Cap train split at TRAIN_LIMIT ────────────────────────────────────
    if TRAIN_LIMIT is not None and len(train_ds) > TRAIN_LIMIT:
        log.info(
            "Capping train split: %d → %d rows (TRAIN_LIMIT=%d)",
            len(train_ds), TRAIN_LIMIT, TRAIN_LIMIT,
        )
        train_ds = train_ds.select(range(TRAIN_LIMIT))

    log.info(
        "Split → train=%d | valid=%d | held_out=%d",
        len(train_ds), len(valid_ds), len(held_out_ds),
    )
    return train_ds, valid_ds, held_out_ds


# ─────────────────────────────────────────────────────────
# 3. Tokenisation
# ─────────────────────────────────────────────────────────

def tokenize_ds(ds: Dataset, tokenizer) -> Dataset:
    """Tokenize title_seg + sapo_seg as (text_a, text_b) pair."""
    def tok(batch):
        enc = tokenizer(
            batch["title_seg"],
            batch["sapo_seg"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        enc.pop("token_type_ids", None)  # PhoBERT=RoBERTa, type_vocab_size=1
        return enc
    return ds.map(
        tok,
        batched=True,
        batch_size=1024,
        num_proc=4,
        remove_columns=ds.column_names,
        desc="Tokenize",
    )

# ─────────────────────────────────────────────────────────
# 4. SOP data collator
# ─────────────────────────────────────────────────────────

@dataclass
class DataCollatorForMLM_SOP:
    """
    Custom collator combining:
      - Dynamic MLM (15 % masking)
      - Sentence Order Prediction: label=0 correct order, label=1 swapped
    
    The SOP task uses token_type_ids to distinguish title (0) and sapo (1).
    We swap sapo segments with 50 % probability to create negative SOP examples.
    """
    tokenizer: Any
    mlm_probability: float = 0.15
    sop_ratio: float = 0.5

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        import copy, random as rnd

        batch_size = len(features)
        sop_labels = []
        processed  = []

        for i, feat in enumerate(features):
            feat = dict(feat)
            feat.pop("token_type_ids", None)  # PhoBERT=RoBERTa has no tti support
            ids = feat["input_ids"]

            # Find SEP positions to locate the sapo segment boundary
            sep_id  = self.tokenizer.sep_token_id
            sep_pos = [j for j, t in enumerate(ids) if t == sep_id]

            if batch_size > 1 and len(sep_pos) >= 2 and rnd.random() < self.sop_ratio:
                # Swap sapo with another sample's sapo to create SOP negative
                partner = (i + rnd.randint(1, batch_size - 1)) % batch_size
                p_feat  = features[partner]
                p_ids   = list(p_feat["input_ids"])
                p_sep   = [j for j, t in enumerate(p_ids) if t == sep_id]

                if len(p_sep) >= 2:
                    title_part = ids[:sep_pos[0] + 1]
                    sapo_part  = p_ids[p_sep[0] + 1:]
                    new_ids = (title_part + sapo_part)[:MAX_LENGTH]
                    feat["input_ids"]      = new_ids
                    feat["attention_mask"] = [1] * len(new_ids)
                    sop_labels.append(1)   # swapped = wrong order
                else:
                    sop_labels.append(0)
            else:
                sop_labels.append(0)       # correct order

            processed.append(feat)

        # Pad to same length -- no token_type_ids in processed features
        batch = self.tokenizer.pad(
            processed,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)  # defensive

        # MLM masking on the padded input_ids
        labels = batch["input_ids"].clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Do not mask special tokens
        special_mask = [
            self.tokenizer.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True)
            for ids in labels
        ]
        special_mask = torch.tensor(special_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # only compute loss on masked tokens

        # 80% MASK, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        batch["input_ids"][indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        batch["input_ids"][indices_random] = random_words[indices_random]

        batch["labels"]     = labels
        batch["sop_labels"] = torch.tensor(sop_labels, dtype=torch.long)
        return batch

# ─────────────────────────────────────────────────────────
# 5. Model with SOP head
# ─────────────────────────────────────────────────────────

class PhoBERTForMLM_SOP(torch.nn.Module):
    """
    Wraps AutoModelForMaskedLM and adds a 2-class SOP head.
    Forward returns a combined loss = mlm_loss + sop_loss.
    """
    def __init__(self, base_model, hidden_size: int = 768):
        super().__init__()
        self.bert   = base_model
        self.sop_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, 2),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,   # accepted but NOT forwarded (PhoBERT=RoBERTa, type_vocab_size=1)
        labels=None,           # MLM labels
        sop_labels=None,       # SOP labels
        **kwargs,
    ):
        # token_type_ids intentionally dropped - PhoBERT only accepts 0
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        mlm_loss = outputs.loss  # None if labels not provided

        # [CLS] representation for SOP
        cls_rep  = outputs.hidden_states[-1][:, 0, :]  # (B, H)
        sop_logits = self.sop_head(cls_rep)

        total_loss = None
        if mlm_loss is not None and sop_labels is not None:
            sop_loss   = torch.nn.functional.cross_entropy(sop_logits, sop_labels)
            total_loss = mlm_loss + sop_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss

        # Return a simple namespace so Trainer can pick up .loss
        from transformers.modeling_outputs import MaskedLMOutput
        return MaskedLMOutput(
            loss=total_loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ── Delegate save / config so Trainer & merge_and_unload work ────────────
    def save_pretrained(self, path, **kw):
        self.bert.save_pretrained(path, **kw)

    @property
    def config(self):
        return self.bert.config

    # ── Gradient-checkpointing delegates ─────────────────────────────────────
    # Trainer calls model.gradient_checkpointing_enable() when
    # TrainingArguments(gradient_checkpointing=True).  This method only exists
    # on PreTrainedModel, not on plain nn.Module wrappers like ours.
    # We delegate to the inner PEFT/HF model so the call succeeds.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is not None:
                self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                self.bert.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.bert, "gradient_checkpointing_disable"):
            self.bert.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        if hasattr(self.bert, "is_gradient_checkpointing"):
            return self.bert.is_gradient_checkpointing
        return False

# ─────────────────────────────────────────────────────────
# 6. Metrics helper
# ─────────────────────────────────────────────────────────

def preprocess_logits_for_metrics(logits, labels):
    """Strip hidden_states; return only vocab logits. Trainer expects only logits."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def compute_ppl_mlm_acc(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    if not isinstance(logits, torch.Tensor):
        logits = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    # only masked positions
    mask    = labels != -100
    if mask.sum() == 0:
        return {"eval_mlm_acc": 0.0, "eval_ppl": float("inf")}

    loss_fn  = torch.nn.CrossEntropyLoss()
    flat_log = logits.view(-1, logits.size(-1))
    flat_lab = labels.view(-1)
    loss     = loss_fn(flat_log, flat_lab)
    ppl      = math.exp(min(loss.item(), 300))

    preds   = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).float().mean().item()
    return {"eval_mlm_acc": round(correct, 4), "eval_ppl": round(ppl, 2)}


def print_trainable_params(model, tag: str):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "[%s] Trainable params: %s / %s (%.2f %%)",
        tag,
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total if total else 0,
    )

# ─────────────────────────────────────────────────────────
# 7. Phase-1: DAPT with LoRA + MLM + SOP
# ─────────────────────────────────────────────────────────

def phase1_dapt(train_tok: Dataset, valid_tok: Dataset, tokenizer):
    log.info("=== PHASE 1: DAPT (MLM + SOP) with LoRA ===")

    base_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    hidden   = base_mlm.config.hidden_size          # 768 for base-v2

    lora_cfg = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGET,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.FEATURE_EXTRACTION,  # MLM pretraining
        modules_to_save= MODULES_TO_SAVE,
    )
    base_mlm = get_peft_model(base_mlm, lora_cfg)
    model    = PhoBERTForMLM_SOP(base_mlm, hidden_size=hidden)

    print_trainable_params(model, "Phase-1 LoRA+SOP")

    collator = DataCollatorForMLM_SOP(
        tokenizer       = tokenizer,
        mlm_probability = P1_MLM_PROB,
        sop_ratio       = P1_SOP_RATIO,
    )

    total_steps = (len(train_tok) // (P1_BATCH * P1_GRAD_ACC)) * P1_EPOCHS

    args = TrainingArguments(
        output_dir                  = PHASE1_DIR,
        num_train_epochs            = P1_EPOCHS,
        per_device_train_batch_size = P1_BATCH,
        per_device_eval_batch_size  = P1_BATCH,
        gradient_accumulation_steps = P1_GRAD_ACC,
        learning_rate               = P1_LR,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 0,           # no warmup for continued PT
        weight_decay                = 0.01,
        bf16                        = True,
        eval_strategy               = "steps",
        eval_steps                  = P1_EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = P1_EVAL_STEPS,
        logging_steps               = 500,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 1,           # only best checkpoint
        dataloader_num_workers          = 8,
        dataloader_pin_memory           = True,
        dataloader_persistent_workers   = True,   # keep workers alive between epochs
        report_to                       = "none",
        seed                            = SEED,
        # ADA-6000 / large-memory optimisations
        optim                           = "adamw_torch_fused",
        gradient_checkpointing          = True,
        label_names                     = ["labels"],  # only MLM labels, not sop_labels
    )

    trainer = Trainer(
        model                         = model,
        args                          = args,
        train_dataset                 = train_tok,
        eval_dataset                  = valid_tok,
        data_collator                 = collator,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        callbacks                     = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info("Starting Phase-1 training … (total steps ≈ %d)", total_steps)
    trainer.train()
    log.info("Phase-1 complete. Best checkpoint at %s", PHASE1_DIR)
    return model

# ─────────────────────────────────────────────────────────
# 8. Merge LoRA
# ─────────────────────────────────────────────────────────

def merge_lora(model: PhoBERTForMLM_SOP) -> AutoModelForMaskedLM:
    log.info("=== Merging LoRA weights ===")
    # model.bert is the PeftModel
    merged_model = model.bert.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR)
    log.info("Merged model saved to %s", MERGED_DIR)
    return merged_model

# ─────────────────────────────────────────────────────────
# 9. Phase-2: Fine-tune merged model (train + valid, 8 epochs)
# ─────────────────────────────────────────────────────────

def phase2_finetune(
    merged_model,
    train_tok: Dataset,
    valid_tok: Dataset,
    tokenizer,
):
    log.info("=== PHASE 2: Fine-tune merged backbone (8 epochs) ===")

    # LoRA again on the merged model for Phase-2
    lora_cfg2 = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGET,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.FEATURE_EXTRACTION,
        modules_to_save= MODULES_TO_SAVE,
    )
    model2 = get_peft_model(merged_model, lora_cfg2)
    print_trainable_params(model2, "Phase-2 LoRA")

    # Combine train + valid for phase-2
    combined = Dataset.from_dict({
        k: train_tok[k] + valid_tok[k]
        for k in train_tok.column_names
    })
    log.info("Phase-2 combined size: %d", len(combined))

    collator2 = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=P1_MLM_PROB
    )

    args2 = TrainingArguments(
        output_dir                  = PHASE2_DIR,
        num_train_epochs            = P2_EPOCHS,
        per_device_train_batch_size = P2_BATCH,
        per_device_eval_batch_size  = P2_BATCH,
        gradient_accumulation_steps = P2_GRAD_ACC,
        learning_rate               = P2_LR,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 0,
        weight_decay                = 0.01,
        bf16                        = True,
        eval_strategy               = "steps",
        eval_steps                  = P2_EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = P2_EVAL_STEPS,
        logging_steps               = 500,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 1,
        dataloader_num_workers          = 8,
        dataloader_pin_memory           = True,
        dataloader_persistent_workers   = True,
        report_to                       = "none",
        seed                            = SEED,
        optim                           = "adamw_torch_fused",
        gradient_checkpointing          = True,
    )

    trainer2 = Trainer(
        model                         = model2,
        args                          = args2,
        train_dataset                 = combined,
        eval_dataset                  = valid_tok,
        data_collator                 = collator2,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        callbacks                     = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info("Starting Phase-2 training …")
    trainer2.train()
    log.info("Phase-2 complete. Best checkpoint at %s", PHASE2_DIR)
    return model2

# ─────────────────────────────────────────────────────────
# 10. Baseline: PhoBERT-base-v2 full fine-tune (no LoRA)
# ─────────────────────────────────────────────────────────

def baseline_finetune(train_tok: Dataset, valid_tok: Dataset, tokenizer):
    """
    Baseline: vanilla PhoBERT-base-v2 fine-tuned với MLM thuần túy,
    không dùng LoRA, không SOP. Cùng số epoch (8) và data với Phase-2
    để so sánh công bằng.
    """
    log.info("=== BASELINE: PhoBERT-base-v2 Full MLM Fine-Tune (no LoRA) ===")

    baseline_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    print_trainable_params(baseline_model, "Baseline (Full Fine-Tune)")

    # Combine train + valid — giống Phase-2
    combined = Dataset.from_dict({
        k: train_tok[k] + valid_tok[k]
        for k in train_tok.column_names
    })
    log.info("Baseline combined dataset size: %d", len(combined))

    collator_bl = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=P1_MLM_PROB
    )

    args_bl = TrainingArguments(
        output_dir                  = BASELINE_DIR,
        num_train_epochs            = P2_EPOCHS,          # 8 epochs — same as Phase-2
        per_device_train_batch_size = P2_BATCH,
        per_device_eval_batch_size  = P2_BATCH,
        gradient_accumulation_steps = P2_GRAD_ACC,
        learning_rate               = P2_LR,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 0,
        weight_decay                = 0.01,
        bf16                        = True,
        eval_strategy               = "steps",
        eval_steps                  = P2_EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = P2_EVAL_STEPS,
        logging_steps               = 500,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 1,
        dataloader_num_workers          = 8,
        dataloader_pin_memory           = True,
        dataloader_persistent_workers   = True,
        report_to                       = "none",
        seed                            = SEED,
        optim                           = "adamw_torch_fused",
        gradient_checkpointing          = True,
    )

    trainer_bl = Trainer(
        model                         = baseline_model,
        args                          = args_bl,
        train_dataset                 = combined,
        eval_dataset                  = valid_tok,
        data_collator                 = collator_bl,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        callbacks                     = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info("Starting Baseline training …")
    trainer_bl.train()
    log.info("Baseline complete. Best checkpoint at %s", BASELINE_DIR)
    return baseline_model


# ─────────────────────────────────────────────────────────
# 11. Intrinsic comparison: Sequential DAPT vs Baseline
# ─────────────────────────────────────────────────────────

def _eval_one(model, held_out_tok: Dataset, tokenizer, tag: str) -> dict:
    """Evaluate một model trên held-out, trả về dict kết quả."""
    collator_eval = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=P1_MLM_PROB
    )
    eval_args = TrainingArguments(
        output_dir                 = os.path.join(OUTPUT_DIR, f"eval_tmp_{tag}"),
        per_device_eval_batch_size = P1_BATCH,
        bf16                       = True,
        dataloader_num_workers     = 8,
        report_to                  = "none",
        seed                       = SEED,
    )
    evaluator = Trainer(
        model                         = model,
        args                          = eval_args,
        eval_dataset                  = held_out_tok,
        data_collator                 = collator_eval,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )
    return evaluator.evaluate()


def compare_intrinsic(
    dapt_model,
    baseline_model,
    held_out_tok: Dataset,
    tokenizer,
):
    """
    Chạy Intrinsic check trên cả 2 model và in bảng so sánh.
    """
    log.info("=== Intrinsic Comparison on Held-Out (10k) ===")

    log.info("Evaluating Sequential DAPT model …")
    dapt_res = _eval_one(dapt_model, held_out_tok, tokenizer, "dapt")

    log.info("Evaluating Baseline model …")
    bl_res   = _eval_one(baseline_model, held_out_tok, tokenizer, "baseline")

    dapt_ppl  = dapt_res.get("eval_ppl",     dapt_res.get("eval_loss", "N/A"))
    dapt_acc  = dapt_res.get("eval_mlm_acc", "N/A")
    bl_ppl    = bl_res.get("eval_ppl",       bl_res.get("eval_loss", "N/A"))
    bl_acc    = bl_res.get("eval_mlm_acc",   "N/A")

    # Delta (DAPT tốt hơn baseline nghĩa là PPL thấp hơn, Acc cao hơn)
    try:
        delta_ppl = round(float(bl_ppl) - float(dapt_ppl), 2)
        delta_acc = round(float(dapt_acc) - float(bl_acc), 4)
        ppl_arrow = "▼ (better)" if delta_ppl > 0 else ("▲ (worse)" if delta_ppl < 0 else "==")
        acc_arrow = "▲ (better)" if delta_acc > 0 else ("▼ (worse)" if delta_acc < 0 else "==")
    except (ValueError, TypeError):
        delta_ppl, delta_acc = "N/A", "N/A"
        ppl_arrow, acc_arrow = "", ""

    sep = "=" * 70
    print(f"\n{sep}")
    print("  ViClickBERT — Held-Out Intrinsic Check (10k samples)")
    print(sep)
    print(f"  {'Metric':<22} {'Sequential DAPT':>20} {'Baseline (vanilla)':>20}")
    print("-" * 70)
    print(f"  {'PPL (Perplexity)':<22} {str(dapt_ppl):>20} {str(bl_ppl):>20}")
    print(f"  {'MLM Accuracy':<22} {str(dapt_acc):>20} {str(bl_acc):>20}")
    print("-" * 70)
    print(f"  {'ΔPPL  (Baseline−DAPT)':<22} {str(delta_ppl):>20}  {ppl_arrow}")
    print(f"  {'ΔAcc  (DAPT−Baseline)':<22} {str(delta_acc):>20}  {acc_arrow}")
    print(sep)
    print("  Note: Sequential DAPT = Phase1 (LoRA 15ep) → merge → Phase2 (LoRA 8ep)")
    print("        Baseline        = PhoBERT-base-v2 full MLM fine-tune (8ep, no LoRA)")
    print(f"{sep}\n")

    return {"dapt": dapt_res, "baseline": bl_res}


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Segmenter
    seg = load_segmenter()

    # ── Step 2: Load & split dataset
    train_ds, valid_ds, held_out_ds = load_and_split(seg)

    # ── Step 3: Tokeniser
    log.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    log.info("Tokenizing splits …")
    train_tok    = tokenize_ds(train_ds,    tokenizer)
    valid_tok    = tokenize_ds(valid_ds,    tokenizer)
    held_out_tok = tokenize_ds(held_out_ds, tokenizer)

    # ── Step 4: Phase-1 DAPT (LoRA + MLM + SOP, 15 epochs)
    p1_model = phase1_dapt(train_tok, valid_tok, tokenizer)

    # ── Step 5: Merge LoRA into backbone
    merged = merge_lora(p1_model)

    # ── Step 6: Phase-2 Sequential fine-tune (LoRA on merged, 8 epochs)
    p2_model = phase2_finetune(merged, train_tok, valid_tok, tokenizer)

    # ── Step 7: Baseline — vanilla PhoBERT-base-v2 (no LoRA, 8 epochs)
    bl_model = baseline_finetune(train_tok, valid_tok, tokenizer)

    # ── Step 8: Intrinsic comparison on held-out 10k
    compare_intrinsic(p2_model, bl_model, held_out_tok, tokenizer)

    log.info("All done. DAPT backbone → %s | Baseline → %s", MERGED_DIR, BASELINE_DIR)


if __name__ == "__main__":
    main()
