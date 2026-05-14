"""
Smoke-test for train_backbone.py
=================================
Chạy TOÀN BỘ pipeline với dữ liệu giả định để phát hiện bug TRƯỚC khi train thật.
Bỏ qua word-segment (VnCoreNLP) — dữ liệu giả đã "pre-segmented".

Pipeline được test:
  [1]  Fake data                  – make_fake_segmented()
  [2]  Tokenization               – tokenize_ds()
  [3]  Build Phase-1 model        – LoRA + PhoBERTForMLM_SOP
  [4]  DataCollatorForMLM_SOP     – test keys / shapes
  [5]  Forward pass               – loss, no NaN
  [6]  Gradient-checkpointing     – delegate OK
  [7]  Phase-1 Trainer mini-run   – MLM + SOP, LoRA
  [8]  merge_lora                 – merge_and_unload(), save_pretrained()
  [9]  Phase-2 Trainer mini-run   – LoRA on merged, plain MLM
  [10] Baseline Trainer mini-run  – full fine-tune, plain MLM
  [11] compare_intrinsic          – eval both models, print table

Cách dùng:
    conda activate MLE
    python training/ViClickBERT/test_smoke.py
"""
import os, sys, math, copy, logging, shutil, warnings
warnings.filterwarnings("ignore")

# ── Thêm root vào sys.path ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from typing import Any, Dict, List
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke_test")

# ── Constants ─────────────────────────────────────────────────────────────
MODEL_NAME  = "vinai/phobert-base-v2"
MAX_LENGTH  = 256
SEED        = 42
LORA_R, LORA_ALPHA, LORA_DROPOUT = 4, 8, 0.05   # nhỏ hơn để test nhanh
LORA_TARGET     = ["query", "key", "value", "dense"]
MODULES_TO_SAVE = ["lm_head"]

N_FAKE  = 20   # số sample giả — đủ nhỏ để chạy nhanh
N_STEPS = 2    # số steps mỗi phase

# ── Import từ train_backbone ───────────────────────────────────────────────
log.info("Importing from train_backbone.py …")
from training.ViClickBERT.train_backbone import (
    PhoBERTForMLM_SOP,
    DataCollatorForMLM_SOP,
    compute_ppl_mlm_acc,
    preprocess_logits_for_metrics,
)
log.info("Import OK")

# ══════════════════════════════════════════════════════════════════════════
# Helper: build TrainingArguments for smoke (minimal, no checkpoint save)
# ══════════════════════════════════════════════════════════════════════════
def _smoke_args(output_dir: str, extra: dict | None = None) -> TrainingArguments:
    """Shared TrainingArguments for all smoke Trainer mini-runs."""
    base = dict(
        output_dir                  = output_dir,
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
        gradient_checkpointing      = False,   # already tested standalone
        optim                       = "adamw_torch",
        dataloader_num_workers      = 0,
        seed                        = SEED,
    )
    if extra:
        base.update(extra)
    return TrainingArguments(**base)


def _split(tok_ds: Dataset):
    """Return (train_slice, eval_slice) from tok_ds."""
    n = len(tok_ds)
    cut = max(4, int(n * 0.75))
    return tok_ds.select(range(cut)), tok_ds.select(range(cut, n))


# ══════════════════════════════════════════════════════════════════════════
# [1]  Fake data
# ══════════════════════════════════════════════════════════════════════════
def make_fake_segmented(n: int) -> Dataset:
    titles = [f"tiêu_đề bài_viết số {i} với nội_dung giả" for i in range(n)]
    sapos  = [f"sapo bài_viết số {i} mô_tả thêm thông_tin" for i in range(n)]
    return Dataset.from_dict({"title_seg": titles, "sapo_seg": sapos})


# ══════════════════════════════════════════════════════════════════════════
# [2]  Tokenization
# ══════════════════════════════════════════════════════════════════════════
def tokenize_ds(ds: Dataset, tokenizer) -> Dataset:
    def tok(batch):
        enc = tokenizer(
            batch["title_seg"],
            batch["sapo_seg"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        enc.pop("token_type_ids", None)   # PhoBERT = RoBERTa, type_vocab_size=1
        return enc
    return ds.map(
        tok, batched=True, batch_size=64,
        remove_columns=ds.column_names, desc="Tokenize",
    )


# ══════════════════════════════════════════════════════════════════════════
# [4]  DataCollatorForMLM_SOP
# ══════════════════════════════════════════════════════════════════════════
def test_collator(tok_ds: Dataset, tokenizer):
    log.info("[TEST 4] DataCollatorForMLM_SOP …")
    collator = DataCollatorForMLM_SOP(
        tokenizer=tokenizer, mlm_probability=0.15, sop_ratio=0.5,
    )
    features = [tok_ds[i] for i in range(min(8, len(tok_ds)))]
    batch = collator(features)
    assert "input_ids"  in batch, "Missing input_ids"
    assert "labels"     in batch, "Missing labels"
    assert "sop_labels" in batch, "Missing sop_labels"
    assert batch["input_ids"].shape[0] == len(features)
    log.info("  ✓ collator keys: %s  |  input_ids shape: %s",
             list(batch.keys()), tuple(batch["input_ids"].shape))


# ══════════════════════════════════════════════════════════════════════════
# [5]  Forward pass
# ══════════════════════════════════════════════════════════════════════════
def test_forward(tok_ds: Dataset, tokenizer, model):
    log.info("[TEST 5] PhoBERTForMLM_SOP forward pass …")
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


# ══════════════════════════════════════════════════════════════════════════
# [6]  Gradient-checkpointing delegate
# ══════════════════════════════════════════════════════════════════════════
def test_gradient_checkpointing(model):
    log.info("[TEST 6] gradient_checkpointing_enable delegate …")
    try:
        model.gradient_checkpointing_enable()
        log.info("  ✓ gradient_checkpointing_enable() OK")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
        log.info("  ✓ gradient_checkpointing_enable(kwargs={}) OK")
        model.gradient_checkpointing_disable()
        log.info("  ✓ gradient_checkpointing_disable() OK")
    except Exception as e:
        log.error("  ✗ error: %s", e)
        raise


# ══════════════════════════════════════════════════════════════════════════
# [7]  Phase-1 Trainer mini-run  (MLM + SOP, LoRA)
# ══════════════════════════════════════════════════════════════════════════
def test_trainer_phase1(tok_ds: Dataset, tokenizer, model, tmp_dir: str):
    log.info("[TEST 7] Trainer mini-run — Phase-1 (MLM + SOP, LoRA) …")
    collator = DataCollatorForMLM_SOP(tokenizer=tokenizer)
    train_ds, eval_ds = _split(tok_ds)

    args = _smoke_args(tmp_dir, {"label_names": ["labels"]})
    trainer = Trainer(
        model                         = model,
        args                          = args,
        train_dataset                 = train_ds,
        eval_dataset                  = eval_ds,
        data_collator                 = collator,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    log.info("  ✓ Phase-1 eval: ppl=%.2f  mlm_acc=%.4f",
             metrics.get("eval_ppl", float("nan")),
             metrics.get("eval_mlm_acc", float("nan")))


# ══════════════════════════════════════════════════════════════════════════
# [8]  merge_lora  (merge_and_unload + save_pretrained)
# ══════════════════════════════════════════════════════════════════════════
def test_merge_lora(model: PhoBERTForMLM_SOP, merged_dir: str):
    """
    Mirror of train_backbone.merge_lora():
      - model.bert is a PeftModel
      - merge_and_unload() folds LoRA into base weights → plain AutoModelForMaskedLM
      - save_pretrained() writes to disk (cleaned up after test)
    """
    log.info("[TEST 8] merge_lora (merge_and_unload + save_pretrained) …")
    merged_model = model.bert.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    # Sanity check
    assert os.path.isfile(os.path.join(merged_dir, "config.json")), "config.json missing after save"
    assert hasattr(merged_model, "forward"), "merged model has no forward()"
    log.info("  ✓ merge_and_unload OK  |  save_pretrained → %s", merged_dir)
    return merged_model   # AutoModelForMaskedLM with merged weights


# ══════════════════════════════════════════════════════════════════════════
# [9]  Phase-2 Trainer mini-run  (LoRA on merged model, plain MLM)
#      Mirrors phase2_finetune() in train_backbone.py
# ══════════════════════════════════════════════════════════════════════════
def test_trainer_phase2(tok_ds: Dataset, tokenizer, merged_model, tmp_dir: str):
    """
    Mirrors phase2_finetune():
      - Apply LoRA to the merged backbone
      - Combine train + valid (here: use full tok_ds as combined)
      - Train with DataCollatorForLanguageModeling (plain MLM, no SOP)
    """
    log.info("[TEST 9] Trainer mini-run — Phase-2 (LoRA on merged, plain MLM) …")

    lora_cfg2 = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET,
        lora_dropout=LORA_DROPOUT, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        modules_to_save=MODULES_TO_SAVE,
    )
    model2 = get_peft_model(merged_model, lora_cfg2)

    # Mirror: combine train + valid into one corpus for phase-2
    combined = tok_ds   # full fake dataset acts as "combined"
    train_ds, eval_ds = _split(combined)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    args = _smoke_args(tmp_dir)
    trainer = Trainer(
        model                         = model2,
        args                          = args,
        train_dataset                 = train_ds,
        eval_dataset                  = eval_ds,
        data_collator                 = collator,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    log.info("  ✓ Phase-2 eval: ppl=%.2f  mlm_acc=%.4f",
             metrics.get("eval_ppl", float("nan")),
             metrics.get("eval_mlm_acc", float("nan")))
    return model2   # PeftModel — used for compare_intrinsic


# ══════════════════════════════════════════════════════════════════════════
# [10] Baseline Trainer mini-run  (full fine-tune, no LoRA)
#      Mirrors baseline_finetune() in train_backbone.py
# ══════════════════════════════════════════════════════════════════════════
def test_trainer_baseline(tok_ds: Dataset, tokenizer, baseline_model, tmp_dir: str):
    """
    Mirrors baseline_finetune():
      - vanilla AutoModelForMaskedLM (no LoRA)
      - combine train + valid
      - plain MLM only
    Note: smoke reuses the merged_model as baseline_model to avoid a
    second HF download — architecturally identical to fresh PhoBERT.
    """
    log.info("[TEST 10] Trainer mini-run — Baseline (full fine-tune, plain MLM) …")

    combined = tok_ds
    train_ds, eval_ds = _split(combined)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    args = _smoke_args(tmp_dir)
    trainer = Trainer(
        model                         = baseline_model,
        args                          = args,
        train_dataset                 = train_ds,
        eval_dataset                  = eval_ds,
        data_collator                 = collator,
        compute_metrics               = compute_ppl_mlm_acc,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    log.info("  ✓ Baseline eval: ppl=%.2f  mlm_acc=%.4f",
             metrics.get("eval_ppl", float("nan")),
             metrics.get("eval_mlm_acc", float("nan")))
    return baseline_model


# ══════════════════════════════════════════════════════════════════════════
# [11] compare_intrinsic  (eval-only on held-out, print comparison table)
#      Mirrors compare_intrinsic() + _eval_one() in train_backbone.py
# ══════════════════════════════════════════════════════════════════════════
def test_compare_intrinsic(
    tok_ds: Dataset, tokenizer,
    dapt_model, baseline_model,
    tmp_dir: str,
):
    """
    Mirrors compare_intrinsic():
      - creates a Trainer for eval-only
      - evaluates DAPT model and Baseline on the same held-out slice
      - prints the comparison table
    """
    log.info("[TEST 11] compare_intrinsic (eval-only on held-out slice) …")

    _, held_out = _split(tok_ds)   # use eval portion as "held-out"

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )

    def _eval_one(model, tag: str) -> dict:
        eval_args = TrainingArguments(
            output_dir             = os.path.join(tmp_dir, f"eval_{tag}"),
            per_device_eval_batch_size = 4,
            bf16                   = torch.cuda.is_bf16_supported(),
            dataloader_num_workers = 0,
            report_to              = "none",
            seed                   = SEED,
        )
        evaluator = Trainer(
            model                         = model,
            args                          = eval_args,
            eval_dataset                  = held_out,
            data_collator                 = collator,
            compute_metrics               = compute_ppl_mlm_acc,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        )
        return evaluator.evaluate()

    dapt_res = _eval_one(dapt_model,     "dapt")
    bl_res   = _eval_one(baseline_model, "baseline")

    dapt_ppl = dapt_res.get("eval_ppl",     dapt_res.get("eval_loss", "N/A"))
    dapt_acc = dapt_res.get("eval_mlm_acc", "N/A")
    bl_ppl   = bl_res.get("eval_ppl",       bl_res.get("eval_loss",   "N/A"))
    bl_acc   = bl_res.get("eval_mlm_acc",   "N/A")

    try:
        delta_ppl = round(float(bl_ppl)  - float(dapt_ppl), 2)
        delta_acc = round(float(dapt_acc) - float(bl_acc),   4)
        ppl_arrow = "▼ (better)" if delta_ppl > 0 else ("▲ (worse)" if delta_ppl < 0 else "==")
        acc_arrow = "▲ (better)" if delta_acc > 0 else ("▼ (worse)" if delta_acc < 0 else "==")
    except (ValueError, TypeError):
        delta_ppl, delta_acc = "N/A", "N/A"
        ppl_arrow, acc_arrow = "", ""

    sep = "=" * 70
    print(f"\n{sep}")
    print("  [SMOKE] ViClickBERT — Intrinsic Check (fake held-out)")
    print(sep)
    print(f"  {'Metric':<22} {'Sequential DAPT':>20} {'Baseline (vanilla)':>20}")
    print("-" * 70)
    print(f"  {'PPL (Perplexity)':<22} {str(dapt_ppl):>20} {str(bl_ppl):>20}")
    print(f"  {'MLM Accuracy':<22} {str(dapt_acc):>20} {str(bl_acc):>20}")
    print("-" * 70)
    print(f"  {'ΔPPL  (BL−DAPT)':<22} {str(delta_ppl):>20}  {ppl_arrow}")
    print(f"  {'ΔAcc  (DAPT−BL)':<22} {str(delta_acc):>20}  {acc_arrow}")
    print(sep)
    print("  Note: table logic from compare_intrinsic() — smoke uses tiny fake data")
    print(f"{sep}\n")

    log.info("  ✓ compare_intrinsic table printed OK")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("SMOKE TEST — train_backbone.py  (full pipeline)")
    log.info("=" * 60)

    # Load tokenizer (shared across all tests)
    log.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── [1] Fake data ─────────────────────────────────────────────────────
    log.info("[1/11] Creating fake segmented data (%d rows) …", N_FAKE)
    fake_ds = make_fake_segmented(N_FAKE)

    # ── [2] Tokenize ─────────────────────────────────────────────────────
    log.info("[2/11] Tokenizing …")
    tok_ds = tokenize_ds(fake_ds, tokenizer)
    log.info("  ✓ columns: %s", tok_ds.column_names)
    assert "token_type_ids" not in tok_ds.column_names, \
        "FAIL: token_type_ids must NOT be in tokenized dataset (PhoBERT=RoBERTa)!"
    log.info("  ✓ token_type_ids correctly absent")

    # ── [3] Build Phase-1 model ──────────────────────────────────────────
    log.info("[3/11] Building PhoBERTForMLM_SOP + LoRA …")
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
    log.info("  ✓ model built  (hidden=%d)", hidden)

    # ── [4] Collator ─────────────────────────────────────────────────────
    log.info("[4/11] Testing DataCollatorForMLM_SOP …")
    test_collator(tok_ds, tokenizer)

    # ── [5] Forward ──────────────────────────────────────────────────────
    log.info("[5/11] Testing forward pass …")
    test_forward(tok_ds, tokenizer, model)

    # ── [6] Gradient checkpointing ───────────────────────────────────────
    log.info("[6/11] Testing gradient_checkpointing_enable delegate …")
    test_gradient_checkpointing(model)

    # ── [7] Phase-1 Trainer ──────────────────────────────────────────────
    log.info("[7/11] Phase-1 Trainer mini-run …")
    tmp_p1 = "result/ViClickBERT/_smoke_p1"
    test_trainer_phase1(tok_ds, tokenizer, model, tmp_p1)

    # ── [8] Merge LoRA ───────────────────────────────────────────────────
    log.info("[8/11] merge_lora …")
    tmp_merged = "result/ViClickBERT/_smoke_merged"
    merged_model = test_merge_lora(model, tmp_merged)

    # ── [9] Phase-2 Trainer (LoRA on merged) ─────────────────────────────
    log.info("[9/11] Phase-2 Trainer mini-run …")
    tmp_p2 = "result/ViClickBERT/_smoke_p2"
    p2_model = test_trainer_phase2(tok_ds, tokenizer, merged_model, tmp_p2)

    # ── [10] Baseline Trainer ─────────────────────────────────────────────
    # Speed opt: fresh copy of merged_model (identical weights to vanilla PhoBERT
    # after merge — saves ~10s HF download during smoke). Real baseline_finetune()
    # calls AutoModelForMaskedLM.from_pretrained() which we tested in step [3].
    log.info("[10/11] Baseline Trainer mini-run …")
    tmp_bl = "result/ViClickBERT/_smoke_baseline"
    import copy as _copy
    baseline_model = _copy.deepcopy(merged_model)   # fresh weights, same arch
    bl_model = test_trainer_baseline(tok_ds, tokenizer, baseline_model, tmp_bl)

    # ── [11] compare_intrinsic ────────────────────────────────────────────
    log.info("[11/11] compare_intrinsic …")
    tmp_cmp = "result/ViClickBERT/_smoke_compare"
    test_compare_intrinsic(tok_ds, tokenizer, p2_model, bl_model, tmp_cmp)

    # ── Cleanup smoke dirs ────────────────────────────────────────────────
    for d in [tmp_p1, tmp_p2, tmp_merged, tmp_bl, tmp_cmp]:
        if os.path.isdir(d):
            shutil.rmtree(d)
    log.info("Smoke dirs cleaned up.")

    log.info("=" * 60)
    log.info("ALL SMOKE TESTS PASSED ✓")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
