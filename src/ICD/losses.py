"""
ICDv4 – Combined Loss Functions
================================
Tổng hợp tất cả loss functions cho ICDv4:
  - FocalLossWithSmoothing (kế thừa v3)
  - SoftLabelKLLoss – KL divergence với LLM soft labels
  - ContrastiveTitleReasonLoss – Cosine embedding loss (ORCD paper, Eq.4,6)
  - rdrop_loss (kế thừa v3)
  - ICDv4CombinedLoss – Wrapper tổng

Refs:
  - ORCD (Zhang et al., WWW'26): arXiv:2601.12019
  - R-Drop (Liang et al., NeurIPS'21)
  - Focal Loss (Lin et al., ICCV'17)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Kế thừa từ ICD_Model_v3
# ===========================================================================
class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with label smoothing.
    Giữ nguyên từ v3 để compatibility.
    """
    def __init__(self, alpha: float = 0.65, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            labels = labels * (1 - self.smoothing) + 0.5 * self.smoothing

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


def rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor,
               alpha: float = 1.0) -> torch.Tensor:
    """
    R-Drop: Symmetric KL divergence giữa 2 forward passes.
    Ref: Liang et al., NeurIPS'21
    """
    p = torch.sigmoid(logits1)
    q = torch.sigmoid(logits2)
    kl_pq = p * torch.log(p / (q + 1e-8) + 1e-8) + \
            (1 - p) * torch.log((1 - p) / (1 - q + 1e-8) + 1e-8)
    kl_qp = q * torch.log(q / (p + 1e-8) + 1e-8) + \
            (1 - q) * torch.log((1 - q) / (1 - p + 1e-8) + 1e-8)
    return alpha * (0.5 * (kl_pq + kl_qp)).mean()


# ===========================================================================
# Soft Label KL Loss
# ===========================================================================
class SoftLabelKLLoss(nn.Module):
    """
    KL divergence giữa model output và LLM-generated soft labels.
    Giúp model học phân phối xác suất từ LLM thay vì chỉ nhãn cứng 0/1.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, p_llm: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1) – model output
        p_llm:  (B, 1) hay (B,) – soft label từ LLM, ∈ [0,1]
        """
        p_model = torch.sigmoid(logits).squeeze(-1)  # (B,)
        p_llm = p_llm.squeeze(-1).float()            # (B,)

        # Clip để tránh log(0)
        p_model = p_model.clamp(1e-7, 1 - 1e-7)
        p_llm = p_llm.clamp(1e-7, 1 - 1e-7)

        # Binary KL: KL(p_llm || p_model) — p_llm là distribution target
        kl = p_llm * torch.log(p_llm / p_model) + \
             (1 - p_llm) * torch.log((1 - p_llm) / (1 - p_model))
        return kl.mean()


# ===========================================================================
# Contrastive Title-Reason Loss (ORCD paper, Eq.4 và Eq.6)
# ===========================================================================
class ContrastiveTitleReasonLoss(nn.Module):
    """
    Cosine embedding loss cho title-reasoning pairs.
    Dùng V_A, V_D làm soft labels (không dùng hard label).

    Theo ORCD paper (Eq.4):
      L_tat = V_A - cos(f_x, f_[x|y])  if r=agree
            = max(V_D, cos(f_x, f_[x|r]) - d)  if r=disagree

    Adaptation cho ICDv4: dùng vector representations z_T, z_A, z_D
    thay vì cross-attention vectors của ORCD gốc.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self,
                z_news: torch.Tensor,
                z_agree: torch.Tensor,
                z_disagree: torch.Tensor,
                v_agree: torch.Tensor,
                v_disagree: torch.Tensor) -> torch.Tensor:
        """
        z_news:      (B, D) – news representation
        z_agree:     (B, D) – agree reasoning representation
        z_disagree:  (B, D) – disagree reasoning representation
        v_agree:     (B,) hay (B,1) – agree score ∈ [0,1]
        v_disagree:  (B,) hay (B,1) – disagree score ∈ [0,1]

        Returns: scalar loss
        """
        v_a = v_agree.squeeze(-1).float().clamp(0, 1)      # (B,)
        v_d = v_disagree.squeeze(-1).float().clamp(0, 1)   # (B,)

        # Normalize cho cosine similarity
        z_n = F.normalize(z_news, dim=-1)
        z_a = F.normalize(z_agree, dim=-1)
        z_d = F.normalize(z_disagree, dim=-1)

        cos_agree    = (z_n * z_a).sum(dim=-1)   # (B,)
        cos_disagree = (z_n * z_d).sum(dim=-1)   # (B,)

        # L_agree:    pull news closer to agree reasoning, weighted by V_A
        # V_A cao → agree reasoning thuyết phục → title nên align nhiều hơn
        loss_agree = (v_a - cos_agree).clamp(min=0).mean()

        # L_disagree: push news away from disagree reasoning
        # V_D thấp → disagree reasoning rất khác → title nên cách xa
        loss_disagree = (v_d * torch.clamp(cos_disagree - self.margin, min=0)).mean()

        return loss_agree + loss_disagree


# ===========================================================================
# ICDv4 Combined Loss
# ===========================================================================
class ICDv4CombinedLoss(nn.Module):
    """
    Tổng hợp tất cả losses cho ICDv4:
      L_total = L_class + α * L_contrastive + β * L_RDrop

    L_class = L_focal + λ_kl * L_kl
    """
    def __init__(
        self,
        focal_alpha: float = 0.65,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        lambda_kl: float = 0.5,        # Weight cho soft label KL loss
        alpha_contrastive: float = 0.5, # Weight cho contrastive loss
        beta_rdrop: float = 1.0,        # Weight cho R-Drop loss
        contrastive_margin: float = 0.3,
    ):
        super().__init__()
        self.focal_loss = FocalLossWithSmoothing(focal_alpha, focal_gamma, label_smoothing)
        self.kl_loss = SoftLabelKLLoss()
        self.contrastive_loss = ContrastiveTitleReasonLoss(margin=contrastive_margin)
        self.lambda_kl = lambda_kl
        self.alpha_contrastive = alpha_contrastive
        self.beta_rdrop = beta_rdrop

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        p_llm_final: torch.Tensor,
        z_news: torch.Tensor,
        z_agree: torch.Tensor,
        z_disagree: torch.Tensor,
        v_agree: torch.Tensor,
        v_disagree: torch.Tensor,
        logits2: torch.Tensor = None,  # Cho R-Drop (second forward pass)
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict với keys: 'total', 'focal', 'kl', 'contrastive', 'rdrop'
        """
        # 1. Focal loss với hard labels
        l_focal = self.focal_loss(logits, labels)

        # 2. Soft label KL loss
        l_kl = self.kl_loss(logits, p_llm_final)

        # 3. Classification loss tổng
        l_class = l_focal + self.lambda_kl * l_kl

        # 4. Contrastive loss
        l_contrastive = self.contrastive_loss(
            z_news, z_agree, z_disagree, v_agree, v_disagree
        )

        # 5. R-Drop loss
        if logits2 is not None and self.beta_rdrop > 0:
            l_rdrop = rdrop_loss(logits, logits2, alpha=self.beta_rdrop)
        else:
            l_rdrop = torch.tensor(0.0, device=logits.device)

        # 6. Total
        l_total = l_class + self.alpha_contrastive * l_contrastive + l_rdrop

        return {
            "total": l_total,
            "focal": l_focal,
            "kl": l_kl,
            "class": l_class,
            "contrastive": l_contrastive,
            "rdrop": l_rdrop,
        }
