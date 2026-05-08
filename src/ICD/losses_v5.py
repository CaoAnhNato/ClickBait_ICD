import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        if self.smoothing > 0.0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
            
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class SoftLabelKLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, soft_labels):
        log_probs = F.logsigmoid(logits)
        log_inv_probs = F.logsigmoid(-logits)
        pred_dist = torch.cat([log_inv_probs, log_probs], dim=-1)
        
        target_dist = torch.cat([1 - soft_labels, soft_labels], dim=-1)
        
        return self.kl_loss(pred_dist, target_dist)

def compute_rdrop_loss(logits1, logits2):
    """
    R-Drop loss cho classification nhị phân.
    """
    p = torch.sigmoid(logits1)
    q = torch.sigmoid(logits2)
    
    # Clip prob để tránh log(0)
    p = torch.clamp(p, 1e-7, 1 - 1e-7)
    q = torch.clamp(q, 1e-7, 1 - 1e-7)
    
    p_dist = torch.cat([1-p, p], dim=-1)
    q_dist = torch.cat([1-q, q], dim=-1)
    
    kl_1 = F.kl_div(torch.log(p_dist), q_dist, reduction='batchmean')
    kl_2 = F.kl_div(torch.log(q_dist), p_dist, reduction='batchmean')
    
    return 0.5 * (kl_1 + kl_2)

class RouterSupervisionLoss(nn.Module):
    def __init__(self, lambda_entropy=0.01):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.lambda_entropy = lambda_entropy

    def forward(self, router_logits, pattern_tags):
        # Normalize pattern tags thành target distribution
        # pattern_tags: (B, 6) có giá trị 0/1
        sums = pattern_tags.sum(dim=-1, keepdim=True)
        # Handle cases where all tags are 0 (assign uniform prob)
        sums = torch.where(sums == 0, torch.ones_like(sums) * pattern_tags.size(-1), sums)
        w_target = torch.where(pattern_tags.sum(dim=-1, keepdim=True) == 0, 
                             torch.ones_like(pattern_tags) / pattern_tags.size(-1), 
                             pattern_tags / sums)
                             
        # Log_softmax for KL input
        log_probs = F.log_softmax(router_logits, dim=-1)
        kl = self.kl_loss(log_probs, w_target)
        
        # Entropy regularizer (minimize entropy to encourage sharper routing if wanted, 
        # or maximize to encourage diverse? Usually we want sparse routing, so minimize entropy)
        # actually paper often uses entropy penalty to ensure load balancing. 
        # Here we just compute standard entropy.
        probs = F.softmax(router_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        
        return kl + self.lambda_entropy * entropy

class ICDv5CombinedLoss(nn.Module):
    def __init__(
        self,
        alpha_focal=1.0,
        gamma_focal=2.0,
        label_smoothing=0.05,
        lambda_kl=0.5,
        lambda_router=0.3,
        lambda_expert=1.0,
        lambda_entropy=0.01,
        beta_rdrop=1.0
    ):
        super().__init__()
        self.focal = FocalLossWithSmoothing(alpha_focal, gamma_focal, label_smoothing)
        self.kl_soft = SoftLabelKLLoss()
        self.router_loss = RouterSupervisionLoss(lambda_entropy)
        
        self.lambda_kl = lambda_kl
        self.lambda_router = lambda_router
        self.lambda_expert = lambda_expert
        self.beta_rdrop = beta_rdrop
        
    def forward(
        self,
        logits,
        labels,
        soft_labels,
        router_logits=None,
        pattern_tags=None,
        expert_aux_logits=None,
        logits2=None
    ):
        loss_dict = {}
        
        # 1. Main classification loss
        l_focal = self.focal(logits, labels)
        loss_dict["focal"] = l_focal
        total_loss = l_focal
        
        # 2. Soft-label KL loss
        if self.lambda_kl > 0 and soft_labels is not None:
            l_kl = self.kl_soft(logits, soft_labels)
            loss_dict["kl_llm"] = l_kl
            total_loss = total_loss + self.lambda_kl * l_kl
            
        # 3. Router supervision loss
        if self.lambda_router > 0 and router_logits is not None and pattern_tags is not None:
            l_router = self.router_loss(router_logits, pattern_tags)
            loss_dict["router"] = l_router
            total_loss = total_loss + self.lambda_router * l_router
            
        # 4. Expert auxiliary loss (BCE per expert for its specific pattern tag)
        if self.lambda_expert > 0 and expert_aux_logits is not None and pattern_tags is not None:
            # expert_aux_logits: (B, 6), pattern_tags: (B, 6)
            l_expert = F.binary_cross_entropy_with_logits(expert_aux_logits, pattern_tags)
            loss_dict["expert_aux"] = l_expert
            total_loss = total_loss + self.lambda_expert * l_expert
            
        # 5. R-Drop loss
        if self.beta_rdrop > 0 and logits2 is not None:
            l_rdrop = compute_rdrop_loss(logits, logits2)
            loss_dict["rdrop"] = l_rdrop
            total_loss = total_loss + self.beta_rdrop * l_rdrop
            
            # Combine focal of logits2
            l_focal2 = self.focal(logits2, labels)
            total_loss = 0.5 * (l_focal + l_focal2) + total_loss - l_focal # replace focal with avg
            loss_dict["focal"] = 0.5 * (l_focal + l_focal2)
            
        loss_dict["total"] = total_loss
        return total_loss, loss_dict
