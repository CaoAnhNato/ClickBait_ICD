import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss hỗ trợ Label Smoothing cho Binary Classification.
    """
    def __init__(self, alpha=0.65, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        """
        logits: [B, 1]
        targets: [B, 1]
        """
        if self.smoothing > 0.0:
            smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        else:
            smoothed_targets = targets
            
        bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')
        probs = torch.sigmoid(logits)
        
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1.0 - pt) ** self.gamma
        
        loss = focal_weight * modulating_factor * bce_loss
        return loss.mean()

class WeightedFocalLossV6(nn.Module):
    """
    Phase 1: Loss Reweighting dựa trên Pattern Tags.
    Nâng cao trọng số cho Hardnews (để giảm FP) và Shock (để tăng Recall).
    """
    def __init__(self, alpha=0.65, gamma=2.0, smoothing=0.1, 
                 hardnews_weight=1.5, shock_weight=1.2):
        super().__init__()
        self.base_loss = FocalLossWithSmoothing(alpha, gamma, smoothing)
        self.hardnews_weight = hardnews_weight
        self.shock_weight = shock_weight

    def forward(self, logits, targets, pattern_tags):
        """
        pattern_tags: [B, 6] -> shock, lifestyle, listicle, analysis, promo, hardnews
        """
        # Calculate base focal loss per sample
        if self.base_loss.smoothing > 0.0:
            smoothed_targets = targets * (1.0 - self.base_loss.smoothing) + 0.5 * self.base_loss.smoothing
        else:
            smoothed_targets = targets
            
        bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = self.base_loss.alpha * targets + (1 - self.base_loss.alpha) * (1 - targets)
        modulating_factor = (1.0 - pt) ** self.base_loss.gamma
        
        per_sample_loss = focal_weight * modulating_factor * bce_loss # [B, 1]
        
        # Reweighting
        # pattern_tags indices: 0=shock, 5=hardnews
        shock_tag = pattern_tags[:, 0:1]
        hardnews_tag = pattern_tags[:, 5:6]
        
        weights = torch.ones_like(per_sample_loss)
        weights = weights + (self.shock_weight - 1.0) * shock_tag
        weights = weights + (self.hardnews_weight - 1.0) * hardnews_tag
        
        weighted_loss = per_sample_loss * weights
        return weighted_loss.mean()

def compute_kl_loss(p, q, pad_mask=None):
    """
    Compute KL divergence KL(p || q).
    p, q: [B, 1] logits
    """
    p_prob = torch.sigmoid(p)
    q_prob = torch.sigmoid(q)
    
    p_dist = torch.cat([1 - p_prob, p_prob], dim=-1)
    q_dist = torch.cat([1 - q_prob, q_prob], dim=-1)
    
    p_dist = torch.clamp(p_dist, 1e-7, 1.0)
    q_dist = torch.clamp(q_dist, 1e-7, 1.0)
    
    loss = F.kl_div(torch.log(q_dist), p_dist, reduction='none').sum(dim=-1) # [B]
    
    if pad_mask is not None:
        loss = loss.masked_fill(pad_mask, 0.0)
        return loss.sum() / ( (~pad_mask).sum() + 1e-8 )
    return loss.mean()

def rdrop_loss(logits1, logits2, alpha=1.0):
    """
    R-Drop Loss.
    logits1, logits2: [B, 1]
    """
    kl1 = compute_kl_loss(logits1, logits2)
    kl2 = compute_kl_loss(logits2, logits1)
    return alpha * 0.5 * (kl1 + kl2)
