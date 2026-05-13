"""
Smoke-test for train_backbone.py
Chạy pipeline đầy đủ với dữ liệu giả định để phát hiện tất cả bug TRƯỚC khi train thật.

Cách dùng:
    conda activate MLE
    python training/ViClickBERT/test_smoke.py
"""
import os, sys, math, logging, warnings
warnings.filterwarnings("ignore")

# ── Thêm root vào sys.path ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.modeling_outputs import MaskedLMOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke_test")

# ── Constants (giống train_backbone.py) ───────────────────────────────────
MODEL_NAME  = "vinai/phobert-base-v2"
MAX_LENGTH  = 256
SEED        = 42
LORA_R, LORA_ALPHA, LORA_DROPOUT = 4, 8, 0.05   # nhỏ hơn để test nhanh
LORA_TARGET  = ["query", "key", "value", "dense"]
MODULES_TO_SAVE = ["lm_head"]

N_FAKE   = 50   # số sample giả
N_STEPS  = 3    # số steps mỗi phase để test nhanh

# ── Import các class từ train_backbone ─────────────────────────────────────
log.info("Importing from train_backbone.py ...")
from training.ViClickBERT.train_backbone import (
    PhoBERTForMLM_SOP,
    DataCollatorForMLM_SOP,
    compute_ppl_mlm_acc,
    preprocess_logits_for_metrics,
)
log.info("Import OK")

# ═══════════════════════════════════════════════════════════════════════════
# 1. Tạo dữ liệu giả
# ═══════════════════════════════════════════════════════════════════════════
def make_fake_segmented(n: int) -> Dataset:
    titles = [f"tiêu_đề bài viết số {i} với nội_dung giả" for i in range(n)]
    sapos  = [f"sapo bài viết số {i} mô_tả thêm thông_tin" for i in range(n)]
    return Dataset.from_dict({"title_seg": titles, "sapo_seg": sapos})

# ═══════════════════════════════════════════════════════════════════════════
# 2. Tokenization
# ═══════════════════════════════════════════════════════════════════════════
def tokenize_ds(ds: Dataset, tokenizer) -> Dataset:
    def tok(batch):
        enc = tokenizer(
            batch["title_seg"],
            batch["sapo_seg"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        # Drop token_type_ids: PhoBERT (RoBERTa) only accepts 0 for all tokens.
        # Passing values of 1 for the second segment causes CUDA index OOB.
        enc.pop("token_type_ids", None)
        return enc
    return ds.map(tok, batched=True, batch_size=64,
                  remove_columns=ds.column_names, desc="Tokenize")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Test DataCollatorForMLM_SOP
# ═══════════════════════════════════════════════════════════════════════════
def test_collator(tok_ds: Dataset, tokenizer):
    log.info("[TEST] DataCollatorForMLM_SOP ...")
    collator = DataCollatorForMLM_SOP(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        sop_ratio=0.5,
    )
    features = [tok_ds[i] for i in range(min(8, len(tok_ds)))]
    batch = collator(features)
    assert "input_ids"  in batch, "Missing input_ids"
    assert "labels"     in batch, "Missing labels"
    assert "sop_labels" in batch, "Missing sop_labels"
    assert batch["input_ids"].shape[0] == len(features)
    log.info("  ✓ collator output keys: %s", list(batch.keys()))
    log.info("  ✓ input_ids shape: %s", tuple(batch["input_ids"].shape))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Test PhoBERTForMLM_SOP forward pass
# ═══════════════════════════════════════════════════════════════════════════
def test_forward(tok_ds: Dataset, tokenizer, model):
    log.info("[TEST] PhoBERTForMLM_SOP forward pass ...")
    collator = DataCollatorForMLM_SOP(tokenizer=tokenizer)
    features = [tok_ds[i] for i in range(4)]
    batch = collator(features)
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=batch["labels"],
            sop_labels=batch["sop_labels"],
        )
    assert out.loss is not None, "loss is None!"
    assert not torch.isnan(out.loss), "loss is NaN!"
    log.info("  ✓ loss = %.4f", out.loss.item())

# ═══════════════════════════════════════════════════════════════════════════
# 5. Test gradient_checkpointing_enable delegate
# ═══════════════════════════════════════════════════════════════════════════
def test_gradient_checkpointing(model):
    log.info("[TEST] gradient_checkpointing_enable delegate ...")
    try:
        model.gradient_checkpointing_enable()
        log.info("  ✓ gradient_checkpointing_enable OK")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
        log.info("  ✓ gradient_checkpointing_enable(kwargs={}) OK")
        model.gradient_checkpointing_disable()
        log.info("  ✓ gradient_checkpointing_disable OK")
    except Exception as e:
        log.error("  ✗ gradient_checkpointing error: %s", e)
        raise

# ═══════════════════════════════════════════════════════════════════════════
# 6. Test Trainer mini-run (Phase-1 style, 3 steps)
# ═══════════════════════════════════════════════════════════════════════════
def test_trainer_phase1(tok_ds: Dataset, tokenizer, model, tmp_dir: str):
    log.info("[TEST] Trainer mini-run (Phase-1) ...")
    collator = DataCollatorForMLM_SOP(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir                = tmp_dir,
        num_train_epochs          = 1,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 4,
        max_steps                 = N_STEPS,
        eval_strategy             = "steps",
        eval_steps                = N_STEPS,
        save_strategy             = "no",
        logging_steps             = 1,
        load_best_model_at_end    = False,
        report_to                 = "none",
        bf16                      = torch.cuda.is_bf16_supported(),
        gradient_checkpointing    = True,
        optim                     = "adamw_torch",
        dataloader_num_workers    = 0,
        # Phase-1 collator emits 'labels' (MLM) and 'sop_labels'.
        # label_names tells Trainer to only cache 'labels' in eval_pred.
        label_names               = ["labels"],
    )
    split = int(len(tok_ds) * 0.8)
    trainer = Trainer(
        model                         = model,
        args                          = args,
        train_dataset                 = tok_ds.select(range(split)),
        eval_dataset                  = tok_ds.select(range(split, len(tok_ds))),
        data_collator                 = collator,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    log.info("  ✓ Phase-1 eval: %s", metrics)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Test DataCollatorForLanguageModeling (Phase-2 / Baseline style)
# ═══════════════════════════════════════════════════════════════════════════
def test_trainer_phase2(tok_ds: Dataset, tokenizer, base_model, tmp_dir: str):
    log.info("[TEST] Trainer mini-run (Phase-2 / MLM only) ...")
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    args = TrainingArguments(
        output_dir                  = tmp_dir,
        num_train_epochs            = 1,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 4,
        max_steps                   = N_STEPS,
        eval_strategy               = "steps",
        eval_steps                  = N_STEPS,
        save_strategy               = "no",
        logging_steps               = 1,
        load_best_model_at_end      = False,
        report_to                   = "none",
        bf16                        = torch.cuda.is_bf16_supported(),
        gradient_checkpointing      = True,
        optim                       = "adamw_torch",
        dataloader_num_workers      = 0,
    )
    split = int(len(tok_ds) * 0.8)
    trainer = Trainer(
        model                         = base_model,
        args                          = args,
        train_dataset                 = tok_ds.select(range(split)),
        eval_dataset                  = tok_ds.select(range(split, len(tok_ds))),
        data_collator                 = collator,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    log.info("  ✓ Phase-2 eval: %s", metrics)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("SMOKE TEST — train_backbone.py")
    log.info("=" * 60)

    # Load tokenizer
    log.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. Fake data
    log.info("[1/7] Creating fake segmented data (%d rows) ...", N_FAKE)
    fake_ds = make_fake_segmented(N_FAKE)

    # 2. Tokenize
    log.info("[2/7] Tokenizing ...")
    tok_ds = tokenize_ds(fake_ds, tokenizer)
    log.info("  ✓ columns after tokenize: %s", tok_ds.column_names)
    assert "token_type_ids" not in tok_ds.column_names, \
        "FAIL: token_type_ids should NOT be in tokenized dataset (PhoBERT=RoBERTa)!"
    log.info("  ✓ token_type_ids correctly absent")

    # 3. Build LoRA model
    log.info("[3/7] Building PhoBERTForMLM_SOP with LoRA ...")
    base_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    hidden   = base_mlm.config.hidden_size
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET,
        lora_dropout=LORA_DROPOUT, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        modules_to_save=MODULES_TO_SAVE,
    )
    peft_model = get_peft_model(base_mlm, lora_cfg)
    model = PhoBERTForMLM_SOP(peft_model, hidden_size=hidden)
    log.info("  ✓ model built")

    # 4. Collator
    log.info("[4/7] Testing collator ...")
    test_collator(tok_ds, tokenizer)

    # 5. Forward
    log.info("[5/7] Testing forward pass ...")
    test_forward(tok_ds, tokenizer, model)

    # 6. Gradient checkpointing
    log.info("[6/7] Testing gradient_checkpointing_enable ...")
    test_gradient_checkpointing(model)

    # 7. Trainer mini-runs
    log.info("[7/7] Testing Trainer mini-runs ...")
    tmp_p1 = "result/ViClickBERT/_smoke_p1"
    tmp_p2 = "result/ViClickBERT/_smoke_p2"

    test_trainer_phase1(tok_ds, tokenizer, model, tmp_p1)

    # Build a fresh base model for phase-2 test
    base_mlm2 = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    test_trainer_phase2(tok_ds, tokenizer, base_mlm2, tmp_p2)

    # Cleanup smoke dirs
    import shutil
    for d in [tmp_p1, tmp_p2]:
        if os.path.isdir(d):
            shutil.rmtree(d)

    log.info("=" * 60)
    log.info("ALL SMOKE TESTS PASSED ✓")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
