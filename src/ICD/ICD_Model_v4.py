"""
ICD Model v4 – LLM-Assisted Opposing-Stance Reasoning
=======================================================
Mở rộng từ ICDv3.1 với:
1. 4 reasoning encoders (TF-agree, TF-disagree, TA-agree, TA-disagree)
2. Shared PhoBERT backbone (tiết kiệm VRAM)
3. Fused classifier head: z_all = [z_T; z_A; z_D]
4. Attention Pooling cho reasoning texts
5. Loss: Focal + Soft-label KL + Contrastive + R-Drop

Architecture:
  Input: (title, lead) + agree_reason + disagree_reason + p_llm_final
  
  Encoders:
    Encoder_T:    PhoBERT → SegmentAwarePool + WeightedLayerPool → z_T (3846)
    Encoder_A_TF: PhoBERT → AttentionPool → h_A_tf (768)
    Encoder_D_TF: PhoBERT → AttentionPool → h_D_tf (768)
    Encoder_A_TA: PhoBERT(title, R_A) → CLS → h_A_ta (768)
    Encoder_D_TA: PhoBERT(title, R_D) → CLS → h_D_ta (768)
  
  Fusion:
    z_A = concat(h_A_tf, h_A_ta)   → 1536
    z_D = concat(h_D_tf, h_D_ta)   → 1536
    z_all = concat(z_T, z_A, z_D)  → 6918
  
  Classifier: LayerNorm → MLP(6918→768→256→1)

Refs:
  - ORCD (Zhang et al., WWW'26): arXiv:2601.12019
  - PhoBERT (Nguyen & Nguyen, 2020)
  - ESIM (Chen et al., ACL'17)
  - R-Drop (Liang et al., NeurIPS'21)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


# ===========================================================================
# Pooling Modules (kế thừa từ v3.1)
# ===========================================================================
class SegmentAwarePool(nn.Module):
    """
    Tách title/lead tokens từ PhoBERT output, pool riêng từng phần.
    Format: <s> title </s></s> lead </s> [PAD]...
    """
    def __init__(self, sep_token_id: int):
        super().__init__()
        self.sep_token_id = sep_token_id

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        batch_size = hidden_states.size(0)
        title_pool_list, lead_pool_list = [], []

        for i in range(batch_size):
            sep_positions = (input_ids[i] == self.sep_token_id).nonzero(as_tuple=True)[0]

            if len(sep_positions) >= 2:
                title_end  = sep_positions[0].item()
                lead_start = sep_positions[1].item() + 1
                lead_end   = sep_positions[2].item() if len(sep_positions) >= 3 \
                             else attention_mask[i].sum().int().item()

                title_tokens = hidden_states[i, 1:title_end, :]
                title_pool = title_tokens.mean(dim=0) if title_tokens.size(0) > 0 \
                             else hidden_states[i, 0, :]

                lead_tokens = hidden_states[i, lead_start:lead_end, :]
                lead_pool = lead_tokens.mean(dim=0) if lead_tokens.size(0) > 0 \
                            else hidden_states[i, 0, :]
            else:
                title_pool = hidden_states[i, 0, :]
                lead_pool  = hidden_states[i, 0, :]

            title_pool_list.append(title_pool)
            lead_pool_list.append(lead_pool)

        return (torch.stack(title_pool_list, dim=0),
                torch.stack(lead_pool_list,  dim=0))


class WeightedLayerPool(nn.Module):
    """Learnable weighted sum của last K hidden states."""
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, hidden_states_list: list) -> torch.Tensor:
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states_list, dim=0)
        return (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)


class AttentionPool(nn.Module):
    """
    Learnable attention pooling cho reasoning texts.
    Học để focus vào các tokens quan trọng nhất trong reasoning.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, L, H)
        attention_mask: (B, L)
        Returns: (B, H)
        """
        # Scores
        scores = self.attention_vector(hidden_states).squeeze(-1)  # (B, L)
        # Mask padding tokens
        mask = (attention_mask == 0)
        scores = scores.masked_fill(mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)      # (B, L, 1)
        pooled = (hidden_states * weights).sum(dim=1)              # (B, H)
        return pooled


# ===========================================================================
# Auxiliary features (giống v3.1)
# ===========================================================================
NUM_AUX_FEATURES = 6


# ===========================================================================
# ICDv4 Main Model
# ===========================================================================
class ClickbaitDetectorV4(nn.Module):
    """
    ICDv4 – LLM-Assisted Opposing-Stance Reasoning Clickbait Detector

    Shared backbone PhoBERT với 5 loại encoding:
    1. News pair (title, lead) → SegmentAware + ESIM → z_T (3846)
    2. TF-Agree: agree_reason alone → AttentionPool → h_A_tf (768)
    3. TF-Disagree: disagree_reason alone → AttentionPool → h_D_tf (768)
    4. TA-Agree: (title, agree_reason) pair → CLS → h_A_ta (768)
    5. TA-Disagree: (title, disagree_reason) pair → CLS → h_D_ta (768)

    Total input to classifier: 3846 + 1536 + 1536 = 6918
    """

    NEWS_INPUT_DIM = 5 * 768 + NUM_AUX_FEATURES   # 3846
    REASON_TF_DIM  = 768
    REASON_TA_DIM  = 768
    FUSED_Z_A_DIM  = REASON_TF_DIM + REASON_TA_DIM  # 1536
    FUSED_Z_D_DIM  = REASON_TF_DIM + REASON_TA_DIM  # 1536
    CLASSIFIER_INPUT_DIM = NEWS_INPUT_DIM + FUSED_Z_A_DIM + FUSED_Z_D_DIM  # 6918

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        sep_token_id: int = 2,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # ── 1. Shared PhoBERT backbone ────────────────────────────────────
        self.config   = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.phobert  = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size  # 768

        # ── 2. News encoder components ────────────────────────────────────
        self.segment_pool      = SegmentAwarePool(sep_token_id=sep_token_id)
        self.weighted_layer_pool = WeightedLayerPool(num_layers=4)

        # ── 3. Reasoning encoder components ──────────────────────────────
        # Title-Free: Attention pooling trên reasoning texts
        self.attn_pool_agree    = AttentionPool(self.hidden_size)
        self.attn_pool_disagree = AttentionPool(self.hidden_size)

        # ── 4. Dropout ────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout_rate)

        # ── 5. Classifier ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.CLASSIFIER_INPUT_DIM),
            nn.Dropout(dropout_rate),
            nn.Linear(self.CLASSIFIER_INPUT_DIM, 768),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        # ── 6. Contrastive Projection (Shared space cho Cosine Similarity) ──
        self.proj_news   = nn.Linear(self.NEWS_INPUT_DIM, 768)
        self.proj_reason = nn.Linear(self.FUSED_Z_A_DIM, 768)

    # ── Forward helpers ───────────────────────────────────────────────────
    def _encode_with_phobert(self, input_ids: torch.Tensor,
                              attention_mask: torch.Tensor):
        """Chạy PhoBERT, trả về (last_hidden, all_hidden_states)."""
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.hidden_states

    def encode_news(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    aux_features: torch.Tensor) -> torch.Tensor:
        """
        Encode (title, lead) sentence pair → z_T (B, 3846).
        Giống ICDv3.1: SegmentAware + ESIM.
        """
        last_hidden, all_hidden = self._encode_with_phobert(input_ids, attention_mask)

        # CLS từ weighted layer pool
        last_4 = list(all_hidden[-4:])
        weighted = self.weighted_layer_pool(last_4)
        cls_output = weighted[:, 0, :]  # (B, 768)

        # Segment-aware dual pool
        title_pool, lead_pool = self.segment_pool(last_hidden, input_ids, attention_mask)

        # ESIM comparison
        diff = torch.abs(title_pool - lead_pool)
        prod = title_pool * lead_pool

        z_T = torch.cat([cls_output, title_pool, lead_pool, diff, prod, aux_features], dim=-1)
        return self.dropout(z_T)  # (B, 3846)

    def encode_reasoning_tf(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                             which: str = "agree") -> torch.Tensor:
        """
        Title-Free encoding: encode chỉ reasoning text → (B, 768).
        which: "agree" hoặc "disagree" để dùng đúng attention pool.
        """
        last_hidden, _ = self._encode_with_phobert(input_ids, attention_mask)
        pool_fn = self.attn_pool_agree if which == "agree" else self.attn_pool_disagree
        return self.dropout(pool_fn(last_hidden, attention_mask))  # (B, 768)

    def encode_reasoning_ta(self, input_ids: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Title-Aware encoding: encode (title, reasoning) sentence pair → CLS (B, 768).
        """
        last_hidden, all_hidden = self._encode_with_phobert(input_ids, attention_mask)
        last_4 = list(all_hidden[-4:])
        weighted = self.weighted_layer_pool(last_4)
        cls_output = weighted[:, 0, :]  # (B, 768)
        return self.dropout(cls_output)

    # ── Main forward ──────────────────────────────────────────────────────
    def forward(
        self,
        # News pair
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aux_features: torch.Tensor,
        # Title-Free reasoning
        input_ids_agree: torch.Tensor,
        attention_mask_agree: torch.Tensor,
        input_ids_disagree: torch.Tensor,
        attention_mask_disagree: torch.Tensor,
        # Title-Aware reasoning
        input_ids_ta_agree: torch.Tensor,
        attention_mask_ta_agree: torch.Tensor,
        input_ids_ta_disagree: torch.Tensor,
        attention_mask_ta_disagree: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:     (B, 1) – classification logit
            z_T:        (B, NEWS_INPUT_DIM) – news representation
            z_agree:    (B, FUSED_Z_A_DIM) – agree reasoning representation
            z_disagree: (B, FUSED_Z_D_DIM) – disagree reasoning representation
        """
        # 1. News encoding
        z_T = self.encode_news(input_ids, attention_mask, aux_features)

        # 2. Title-Free reasoning encodings
        h_A_tf = self.encode_reasoning_tf(input_ids_agree, attention_mask_agree, "agree")
        h_D_tf = self.encode_reasoning_tf(input_ids_disagree, attention_mask_disagree, "disagree")

        # 3. Title-Aware reasoning encodings
        h_A_ta = self.encode_reasoning_ta(input_ids_ta_agree, attention_mask_ta_agree)
        h_D_ta = self.encode_reasoning_ta(input_ids_ta_disagree, attention_mask_ta_disagree)

        # 4. Fuse reasoning representations
        z_agree    = torch.cat([h_A_tf, h_A_ta], dim=-1)   # (B, 1536)
        z_disagree = torch.cat([h_D_tf, h_D_ta], dim=-1)   # (B, 1536)

        # 5. Concatenate all representations
        z_all = torch.cat([z_T, z_agree, z_disagree], dim=-1)  # (B, 6918)

        # 6. Classify (dùng raw features)
        logits = self.classifier(z_all)  # (B, 1)

        # 7. Project cho contrastive loss
        z_T_proj = self.proj_news(z_T)
        z_A_proj = self.proj_reason(z_agree)
        z_D_proj = self.proj_reason(z_disagree)

        return logits, z_T_proj, z_A_proj, z_D_proj

    # ── Utility methods ───────────────────────────────────────────────────
    def freeze_backbone_layers(self, freeze_until: int = 8):
        """Freeze bottom N PhoBERT layers để tăng tốc convergence."""
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
        for layer_idx in range(freeze_until):
            for param in self.phobert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[*] Frozen layers 0-{freeze_until-1}. "
              f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    def get_parameter_groups(self, lr: float = 3e-5, lr_decay: float = 0.98):
        """Layer-wise LR decay (tương tự ICDv3.1)."""
        param_groups = []

        # Embeddings
        embed_params = [p for p in self.phobert.embeddings.parameters() if p.requires_grad]
        if embed_params:
            param_groups.append({"params": embed_params, "lr": lr * (lr_decay ** 12)})

        # Encoder layers
        for layer_idx in range(12):
            layer_params = [p for p in self.phobert.encoder.layer[layer_idx].parameters()
                            if p.requires_grad]
            if layer_params:
                param_groups.append({"params": layer_params,
                                     "lr": lr * (lr_decay ** (11 - layer_idx))})

        # Non-backbone (pooling + attention + classifier)
        non_backbone = [p for n, p in self.named_parameters()
                        if "phobert" not in n and p.requires_grad]
        if non_backbone:
            param_groups.append({"params": non_backbone, "lr": lr * 10})

        return param_groups
