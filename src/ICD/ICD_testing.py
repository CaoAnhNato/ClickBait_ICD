"""
Unit Tests cho ICD Model v2
============================
Kiểm tra tính đúng đắn của kiến trúc model sau khi refactor:
- Shape các layer
- Masking hoạt động đúng
- Loss function tính toán hợp lý
- Logits scale không bị explosion
- Vietnamese char encoding

Chạy: conda run -n MLE python src/ICD/ICD_testing.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Thêm đường dẫn root vào sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.ICD.ICD_Model import (
    WordLevelAttention, ContentModelingModule, 
    StyleModelingModule, InteractionModelingModule,
    ClickbaitDetectionModel, JointLoss, FocalLoss,
    CharacterLevelAttention
)

# ============================================================================
# Test Configuration
# ============================================================================
BATCH_SIZE = 2
SEQ_LEN_TITLE = 32
SEQ_LEN_LEAD = 64
HIDDEN_SIZE = 768
D_C = 64          # [v2] Giảm từ 128 → 64
MAX_CHAR_LEN = 150
VOCAB_SIZE = 250   # [v2] Vietnamese-aware vocab

def test_word_level_attention():
    """Test WordLevelAttention: shape output và masking"""
    print("Test 1: WordLevelAttention...", end=" ")
    
    attn = WordLevelAttention(HIDDEN_SIZE)
    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN_TITLE, HIDDEN_SIZE)
    
    # Tạo mask: sample 0 có 20 tokens thật, sample 1 có 10 tokens thật
    mask = torch.zeros(BATCH_SIZE, SEQ_LEN_TITLE)
    mask[0, :20] = 1.0
    mask[1, :10] = 1.0
    
    e_out, alpha = attn(hidden_states, mask)
    
    assert e_out.shape == (BATCH_SIZE, HIDDEN_SIZE), f"Expected ({BATCH_SIZE}, {HIDDEN_SIZE}), got {e_out.shape}"
    assert alpha.shape == (BATCH_SIZE, SEQ_LEN_TITLE), f"Expected ({BATCH_SIZE}, {SEQ_LEN_TITLE}), got {alpha.shape}"
    
    # Kiểm tra alpha trọng số tại padding position ≈ 0
    assert alpha[0, 25].item() < 1e-3, "Alpha at padding position should be near 0"
    assert alpha[1, 15].item() < 1e-3, "Alpha at padding position should be near 0"
    
    # Kiểm tra alpha sum ≈ 1 trên valid tokens
    assert abs(alpha[0].sum().item() - 1.0) < 1e-4, "Alpha sum should be ~1.0"
    
    print("PASSED ✓")

def test_style_module():
    """Test StyleModelingModule: shape và d_c=64"""
    print("Test 2: StyleModelingModule (d_c=64)...", end=" ")
    
    style = StyleModelingModule(vocab_size=VOCAB_SIZE, d_c=D_C, nhead=4, num_layers=2)
    char_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_CHAR_LEN))
    char_mask = torch.ones(BATCH_SIZE, MAX_CHAR_LEN)
    char_mask[0, 80:] = 0  # Padding
    char_mask[1, 50:] = 0
    
    e_style = style(char_ids, char_mask)
    
    assert e_style.shape == (BATCH_SIZE, D_C), f"Expected ({BATCH_SIZE}, {D_C}), got {e_style.shape}"
    assert not torch.isnan(e_style).any(), "Style output contains NaN"
    assert not torch.isinf(e_style).any(), "Style output contains Inf"
    
    print("PASSED ✓")

def test_interaction_module():
    """Test InteractionModelingModule: shape, masking, và projection output"""
    print("Test 3: InteractionModelingModule (with projection)...", end=" ")
    
    interaction = InteractionModelingModule(hidden_size=HIDDEN_SIZE)
    
    H_title = torch.randn(BATCH_SIZE, SEQ_LEN_TITLE, HIDDEN_SIZE)
    H_lead = torch.randn(BATCH_SIZE, SEQ_LEN_LEAD, HIDDEN_SIZE)
    
    title_mask = torch.ones(BATCH_SIZE, SEQ_LEN_TITLE)
    title_mask[0, 20:] = 0
    title_mask[1, 10:] = 0
    
    lead_mask = torch.ones(BATCH_SIZE, SEQ_LEN_LEAD)
    lead_mask[0, 40:] = 0
    lead_mask[1, 30:] = 0
    
    r_title, r_lead = interaction(H_title, title_mask, H_lead, lead_mask)
    
    # [v2] Output shape phải là (B, hidden_size) thay vì (B, 2*hidden_size)
    assert r_title.shape == (BATCH_SIZE, HIDDEN_SIZE), \
        f"Expected ({BATCH_SIZE}, {HIDDEN_SIZE}), got {r_title.shape}"
    assert r_lead.shape == (BATCH_SIZE, HIDDEN_SIZE), \
        f"Expected ({BATCH_SIZE}, {HIDDEN_SIZE}), got {r_lead.shape}"
    
    # Kiểm tra co-attention masking
    A_T = interaction._A_T_temp  # (B, N, P)
    A_L = interaction._A_L_temp  # (B, P, N)
    
    # A_T attention tại padding lead positions phải ≈ 0
    assert A_T[0, 0, 50].item() < 1e-3, "A_T at lead padding should be ~0"
    assert A_T[1, 0, 35].item() < 1e-3, "A_T at lead padding should be ~0"
    
    # A_L attention tại padding title positions phải ≈ 0
    assert A_L[0, 0, 25].item() < 1e-3, "A_L at title padding should be ~0"
    assert A_L[1, 0, 15].item() < 1e-3, "A_L at title padding should be ~0"
    
    # Kiểm tra không có NaN/Inf
    assert not torch.isnan(r_title).any(), "r_title contains NaN"
    assert not torch.isnan(r_lead).any(), "r_lead contains NaN"
    
    print("PASSED ✓")

def test_focal_loss():
    """Test FocalLoss: tính toán đúng và xử lý class imbalance"""
    print("Test 4: FocalLoss...", end=" ")
    
    focal = FocalLoss(alpha=0.6, gamma=2.0)
    
    # Test case 1: Easy correct predictions (high confidence) → loss should be LOW
    logits_easy = torch.tensor([[5.0], [-5.0]])  # Predict 1 and 0 confidently
    labels_easy = torch.tensor([[1.0], [0.0]])
    loss_easy = focal(logits_easy, labels_easy)
    
    # Test case 2: Hard/wrong predictions → loss should be HIGHER
    logits_hard = torch.tensor([[-2.0], [2.0]])  # Predict opposite
    labels_hard = torch.tensor([[1.0], [0.0]])
    loss_hard = focal(logits_hard, labels_hard)
    
    assert loss_hard > loss_easy, f"Hard examples should have higher loss: {loss_hard:.4f} vs {loss_easy:.4f}"
    assert loss_easy.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss_easy), "Loss should not be NaN"
    
    # Test case 3: Focal loss phải nhỏ hơn BCE cho easy examples (gamma effect)
    bce = nn.BCEWithLogitsLoss()
    bce_loss_easy = bce(logits_easy, labels_easy)
    # Focal loss giảm weight cho easy examples → loss thấp hơn BCE
    assert loss_easy < bce_loss_easy, \
        f"Focal loss should be < BCE for easy examples: {loss_easy:.6f} vs {bce_loss_easy:.6f}"
    
    print("PASSED ✓")

def test_joint_loss_v2():
    """Test JointLoss v2: Focal + Contrastive with temperature"""
    print("Test 5: JointLoss v2 (Focal + Contrastive)...", end=" ")
    
    loss_fn = JointLoss(margin=0.5, lambda_weight=0.3, focal_alpha=0.6, focal_gamma=2.0)
    
    logits = torch.randn(BATCH_SIZE, 1)
    labels = torch.tensor([[1.0], [0.0]])
    e_title = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
    e_lead = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
    
    total_loss, cls_loss, L_CL = loss_fn(logits, labels, e_title, e_lead)
    
    assert total_loss.shape == (), f"Total loss should be scalar, got {total_loss.shape}"
    assert cls_loss.shape == (), f"Cls loss should be scalar, got {cls_loss.shape}"
    assert L_CL.shape == (), f"CL loss should be scalar, got {L_CL.shape}"
    
    assert total_loss.item() >= 0, "Total loss should be non-negative"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"
    
    # Kiểm tra temperature parameter tồn tại và trong khoảng hợp lý
    temp = torch.exp(loss_fn.log_temperature).item()
    assert 0.01 <= temp <= 1.0, f"Temperature should be in [0.01, 1.0], got {temp}"
    
    # Kiểm tra contrastive loss logic:
    # Với non-clickbait (label=0): similar title-lead → low D_cos → low loss
    e_similar = torch.randn(1, HIDDEN_SIZE)
    e_same = e_similar.clone()  # Exact same → cosine_sim=1, D_cos=0
    labels_nonclick = torch.tensor([[0.0]])
    logits_dummy = torch.zeros(1, 1)
    
    _, _, L_CL_similar = loss_fn(logits_dummy, labels_nonclick, e_similar, e_same)
    
    # D_cos should be ~0 for identical vectors → loss_0 ≈ 0
    assert L_CL_similar.item() < 0.1, \
        f"CL loss for identical vectors (non-clickbait) should be ~0, got {L_CL_similar:.4f}"
    
    print("PASSED ✓")

def test_logits_scale():
    """
    [v2] Test Critical: Kiểm tra logits scale không bị explosion
    
    Đây là test quan trọng nhất - xác nhận fix cho bug dot product explosion.
    v1: logits có thể lên tới hàng nghìn do sum(r_title * r_lead, dim=-1) với dim=1536
    v2: logits phải nằm trong khoảng hợp lý [-20, 20] cho binary classification
    """
    print("Test 6: Logits Scale (no explosion)...", end=" ")
    
    model = ClickbaitDetectionModel(
        vocab_size=VOCAB_SIZE, 
        content_model_name="vinai/phobert-base", 
        hidden_size=HIDDEN_SIZE, 
        d_c=D_C
    )
    model.eval()
    
    # Tạo input giả
    title_ids = torch.randint(0, 64000, (BATCH_SIZE, SEQ_LEN_TITLE))
    title_mask = torch.ones(BATCH_SIZE, SEQ_LEN_TITLE)
    lead_ids = torch.randint(0, 64000, (BATCH_SIZE, SEQ_LEN_LEAD))
    lead_mask = torch.ones(BATCH_SIZE, SEQ_LEN_LEAD)
    char_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_CHAR_LEN))
    char_mask = torch.ones(BATCH_SIZE, MAX_CHAR_LEN)
    
    with torch.no_grad():
        logits, e_title, e_lead = model(title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask)
    
    # Kiểm tra shape
    assert logits.shape == (BATCH_SIZE, 1), f"Expected logits shape ({BATCH_SIZE}, 1), got {logits.shape}"
    assert e_title.shape == (BATCH_SIZE, HIDDEN_SIZE), f"Expected e_title shape ({BATCH_SIZE}, {HIDDEN_SIZE})"
    assert e_lead.shape == (BATCH_SIZE, HIDDEN_SIZE), f"Expected e_lead shape ({BATCH_SIZE}, {HIDDEN_SIZE})"
    
    # [v2] CRITICAL: Kiểm tra logits scale
    max_logit = logits.abs().max().item()
    assert max_logit < 50.0, \
        f"LOGITS EXPLOSION DETECTED! Max |logit|={max_logit:.1f}. Should be < 50.0"
    
    # Kiểm tra không có NaN/Inf
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert not torch.isinf(logits).any(), "Logits contain Inf"
    
    print(f"PASSED ✓ (max |logit|={max_logit:.2f})")

def test_vietnamese_char_encoding():
    """
    [v2] Test Vietnamese char encoding trong CharTokenizer
    Đảm bảo các ký tự tiếng Việt được encode đúng (không collision)
    """
    print("Test 7: Vietnamese CharTokenizer...", end=" ")
    
    # Import CharTokenizer từ training script
    sys.path.insert(0, os.path.join(base_dir, 'training', 'ICD'))
    from train_ICD import CharTokenizer
    
    tokenizer = CharTokenizer(max_length=30)
    
    # Test 1: Vietnamese text encoding
    vn_text = "Ấn độ phát hiện"
    ids, mask = tokenizer.encode([vn_text])
    
    assert ids.shape == (1, 30), f"Expected shape (1, 30), got {ids.shape}"
    assert mask[0, :len(vn_text)].sum() == len(vn_text), "Mask should mark all chars as valid"
    assert mask[0, len(vn_text):].sum() == 0, "Mask should be 0 for padding"
    
    # Test 2: Không có collision cho các ký tự khác nhau
    chars_to_test = ['a', 'ấ', 'đ', 'ệ', 'ư', 'ợ']
    encoded_ids = [tokenizer.char2id.get(c, tokenizer.unk_id) for c in chars_to_test]
    # Tất cả IDs phải khác nhau (không collision)
    assert len(set(encoded_ids)) == len(chars_to_test), \
        f"Character collision detected! IDs: {dict(zip(chars_to_test, encoded_ids))}"
    
    # Test 3: Special clickbait chars được encode riêng
    special_chars = ['?', '!', '…', '"']
    special_ids = [tokenizer.char2id.get(c, tokenizer.unk_id) for c in special_chars]
    assert all(id != tokenizer.unk_id for id in special_ids), \
        f"Special chars should not map to UNK: {dict(zip(special_chars, special_ids))}"
    
    # Test 4: Vocab size phải > 200 để bao phủ đủ ký tự
    assert tokenizer.vocab_size >= 200, \
        f"Vocab size too small: {tokenizer.vocab_size}, expected >= 200"
    
    print(f"PASSED ✓ (vocab_size={tokenizer.vocab_size})")

def test_full_forward_backward():
    """Test full forward + backward pass: đảm bảo gradients flow correctly"""
    print("Test 8: Full Forward-Backward Pass...", end=" ")
    
    model = ClickbaitDetectionModel(
        vocab_size=VOCAB_SIZE,
        content_model_name="vinai/phobert-base",
        hidden_size=HIDDEN_SIZE,
        d_c=D_C
    )
    loss_fn = JointLoss(margin=0.5, lambda_weight=0.3)
    
    # Tạo input
    title_ids = torch.randint(0, 64000, (BATCH_SIZE, SEQ_LEN_TITLE))
    title_mask = torch.ones(BATCH_SIZE, SEQ_LEN_TITLE)
    lead_ids = torch.randint(0, 64000, (BATCH_SIZE, SEQ_LEN_LEAD))
    lead_mask = torch.ones(BATCH_SIZE, SEQ_LEN_LEAD)
    char_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_CHAR_LEN))
    char_mask = torch.ones(BATCH_SIZE, MAX_CHAR_LEN)
    labels = torch.tensor([[1.0], [0.0]])
    
    # Forward
    logits, e_title, e_lead = model(title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask)
    total_loss, _, _ = loss_fn(logits, labels, e_title, e_lead)
    
    # Backward
    total_loss.backward()
    
    # Kiểm tra gradients tồn tại cho các key parameters
    # Classifier MLP
    classifier_has_grad = False
    for name, param in model.classifier.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            classifier_has_grad = True
            break
    assert classifier_has_grad, "Classifier should have non-zero gradients"
    
    # Style projection
    assert model.style_projection.weight.grad is not None, "Style projection should have gradients"
    
    # Interaction projection
    for name, param in model.interaction_module.title_projection.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Interaction projection ({name}) should have gradients"
            break
    
    # Temperature parameter in loss
    assert loss_fn.log_temperature.grad is not None, "Temperature parameter should have gradients"
    
    print("PASSED ✓")

# ============================================================================
# Run All Tests
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ICD Model v2 - Unit Tests")
    print("=" * 60)
    
    test_word_level_attention()
    test_style_module()
    test_interaction_module()
    test_focal_loss()
    test_joint_loss_v2()
    test_logits_scale()
    test_vietnamese_char_encoding()
    test_full_forward_backward()
    
    print("\n" + "=" * 60)
    print("ALL 8 TESTS PASSED ✓")
    print("=" * 60)