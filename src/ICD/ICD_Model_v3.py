"""
ICD Model v3.1 – Segment-Aware Dual Pooling + R-Drop
======================================================
Iteration 2: Cải tiến từ v3 (val F1=0.747) hướng tới F1 ≥ 0.85

Thay đổi từ v3:
1. Segment-Aware Pooling: sau PhoBERT encoding, tách title/lead tokens rồi pool riêng
   → tạo explicit comparison features (|title-lead|, title*lead) theo ESIM pattern
   → kết hợp ưu điểm: cross-attention deep (PhoBERT) + explicit comparison
2. R-Drop Regularization thay Multi-Sample Dropout
   → KL divergence giữa 2 forward passes với dropout khác nhau
   → theoretically more principled, proven +1-3% improvement
3. Higher LR (3e-5) với less aggressive layer decay (0.98)
4. Label smoothing (0.05) cho Focal Loss

Refs:
- ESIM (Chen et al., ACL'17): [diff, product] comparison features
- R-Drop (Liang et al., NeurIPS'21): Regularized Dropout for consistency training
- PhoBERT (Nguyen & Nguyen, 2020): Vietnamese RoBERTa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class SegmentAwarePool(nn.Module):
    """
    Segment-Aware Pooling: tách title/lead tokens từ PhoBERT output,
    pool mỗi phần riêng biệt.
    
    Sau khi PhoBERT encode sentence pair <s> title </s></s> lead </s>,
    hidden states chứa cả title và lead tokens ĐÃ cross-attend lẫn nhau.
    Module này tách chúng ra để tạo explicit comparison features.
    
    Ưu điểm so với v3 (pool toàn bộ sequence):
    - Cho phép so sánh title vs lead một cách rõ ràng
    - Nắm bắt sự khác biệt ngữ nghĩa (clickbait signal chính)
    """
    def __init__(self, sep_token_id: int):
        super(SegmentAwarePool, self).__init__()
        self.sep_token_id = sep_token_id
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        input_ids: (batch_size, seq_len) - để tìm vị trí separator
        attention_mask: (batch_size, seq_len)
        
        Returns:
            title_pool: (batch_size, hidden_size)
            lead_pool: (batch_size, hidden_size)
        """
        batch_size = hidden_states.size(0)
        hidden_size = hidden_states.size(2)
        
        title_pool_list = []
        lead_pool_list = []
        
        for i in range(batch_size):
            # Tìm vị trí các separator tokens </s>
            # Format: <s>[0] title_tokens... </s>[sep1] </s>[sep2] lead_tokens... </s>[sep3] [PAD]...
            sep_positions = (input_ids[i] == self.sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) >= 2:
                # Title tokens: từ vị trí 1 (sau <s>) đến sep1 (không bao gồm sep1)
                title_end = sep_positions[0].item()
                # Lead tokens: từ sep2+1 đến sep3 (không bao gồm sep3)
                lead_start = sep_positions[1].item() + 1
                if len(sep_positions) >= 3:
                    lead_end = sep_positions[2].item()
                else:
                    # Fallback: tìm cuối sequence thực
                    lead_end = attention_mask[i].sum().int().item()
                
                # Pool title tokens (mean pooling)
                title_tokens = hidden_states[i, 1:title_end, :]  # Skip <s>
                if title_tokens.size(0) > 0:
                    title_pool = title_tokens.mean(dim=0)
                else:
                    title_pool = hidden_states[i, 0, :]  # Fallback to CLS
                
                # Pool lead tokens (mean pooling)
                lead_tokens = hidden_states[i, lead_start:lead_end, :]
                if lead_tokens.size(0) > 0:
                    lead_pool = lead_tokens.mean(dim=0)
                else:
                    lead_pool = hidden_states[i, 0, :]  # Fallback to CLS
            else:
                # Fallback: nếu không tìm thấy separator, dùng CLS
                title_pool = hidden_states[i, 0, :]
                lead_pool = hidden_states[i, 0, :]
            
            title_pool_list.append(title_pool)
            lead_pool_list.append(lead_pool)
        
        title_pool = torch.stack(title_pool_list, dim=0)  # (batch_size, hidden_size)
        lead_pool = torch.stack(lead_pool_list, dim=0)    # (batch_size, hidden_size)
        
        return title_pool, lead_pool


class WeightedLayerPool(nn.Module):
    """Learnable weighted sum of last K hidden states."""
    def __init__(self, num_layers: int = 4):
        super(WeightedLayerPool, self).__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, hidden_states_list: list) -> torch.Tensor:
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states_list, dim=0)
        weighted = stacked * weights.view(-1, 1, 1, 1)
        return weighted.sum(dim=0)


class ClickbaitDetectorV3_1(nn.Module):
    """
    Vietnamese Clickbait Detection Model v3.1
    
    Architecture:
    1. PhoBERT encodes sentence pair (12 layers of cross-attention)
    2. Segment-Aware Dual Pooling: tách title/lead → pool riêng
    3. ESIM-style comparison: [CLS, title_pool, lead_pool, |diff|, product]
    4. Auxiliary linguistic features
    5. MLP Classifier with dropout
    
    Input: tokenizer(title, lead) → <s> title </s></s> lead </s>
    
    Total classifier input:
    - CLS: 768
    - title_pool: 768  (mean of cross-attended title tokens)
    - lead_pool: 768   (mean of cross-attended lead tokens)
    - |title-lead|: 768 (semantic difference = clickbait signal)
    - title*lead: 768   (semantic similarity)
    - aux_features: 6
    = 3846
    """
    
    NUM_AUX_FEATURES = 6
    
    def __init__(self, model_name: str = "vinai/phobert-base", sep_token_id: int = 2,
                 dropout_rate: float = 0.3):
        super(ClickbaitDetectorV3_1, self).__init__()
        
        # 1. PhoBERT backbone
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.phobert = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size  # 768
        
        # 2. Segment-Aware Pooling
        self.segment_pool = SegmentAwarePool(sep_token_id=sep_token_id)
        
        # 3. Weighted Layer Pooling (dùng cho CLS representation)
        self.weighted_layer_pool = WeightedLayerPool(num_layers=4)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 5. MLP Classifier
        # [CLS(768), title(768), lead(768), diff(768), prod(768), aux(6)] = 3846
        classifier_input_dim = 5 * self.hidden_size + self.NUM_AUX_FEATURES
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim, 768),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                aux_features: torch.Tensor) -> torch.Tensor:
        """
        Returns: logits (batch_size, 1)
        """
        # 1. PhoBERT encoding
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states
        
        # 2. CLS representation từ weighted layer pool
        last_4_layers = list(all_hidden_states[-4:])
        weighted_hidden = self.weighted_layer_pool(last_4_layers)
        cls_output = weighted_hidden[:, 0, :]  # (B, 768)
        
        # 3. Segment-Aware Dual Pooling
        title_pool, lead_pool = self.segment_pool(last_hidden, input_ids, attention_mask)
        
        # 4. ESIM-style comparison features
        diff = torch.abs(title_pool - lead_pool)   # Semantic difference (clickbait signal)
        prod = title_pool * lead_pool               # Semantic similarity
        
        # 5. Concatenate
        combined = torch.cat([
            cls_output,    # Global representation
            title_pool,    # Cross-attended title representation
            lead_pool,     # Cross-attended lead representation
            diff,          # How different are they? (clickbait = high diff)
            prod,          # How similar are they?
            aux_features,  # Linguistic features
        ], dim=-1)  # (B, 3846)
        
        # 6. Dropout + Classifier
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        
        return logits
    
    def freeze_backbone_layers(self, freeze_until: int = 8):
        """
        Freeze bottom transformer layers để tập trung learning capacity
        vào top layers + classifier.
        
        Freeze_until=8 → freeze layers 0-7, train layers 8-11 + classifier
        Giúp convergence nhanh hơn đáng kể với ít epochs.
        """
        # Freeze embeddings
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze bottom layers
        for layer_idx in range(freeze_until):
            for param in self.phobert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[*] Frozen layers 0-{freeze_until-1}. "
              f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")
    
    def get_parameter_groups(self, lr: float = 3e-5, lr_decay: float = 0.98):
        """Layer-wise LR decay with less aggressive decay for v3.1."""
        param_groups = []
        
        # Embeddings
        embed_params = [p for p in self.phobert.embeddings.parameters() if p.requires_grad]
        if embed_params:
            param_groups.append({
                'params': embed_params,
                'lr': lr * (lr_decay ** 12)
            })
        
        # Encoder layers (only trainable ones)
        for layer_idx in range(12):
            layer_params = [p for p in self.phobert.encoder.layer[layer_idx].parameters() 
                           if p.requires_grad]
            if layer_params:
                param_groups.append({
                    'params': layer_params,
                    'lr': lr * (lr_decay ** (11 - layer_idx))
                })
        
        # Non-backbone (pooling + classifier) - highest LR
        non_backbone_params = [p for n, p in self.named_parameters() 
                              if 'phobert' not in n and p.requires_grad]
        if non_backbone_params:
            param_groups.append({
                'params': non_backbone_params,
                'lr': lr * 10  # 10x for classifier
            })
        
        return param_groups


class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with optional label smoothing.
    
    Label smoothing prevents overconfidence:
    - label 0 → smoothing/2
    - label 1 → 1 - smoothing/2
    
    Helps model generalize better on ambiguous clickbait examples.
    """
    def __init__(self, alpha: float = 0.6, gamma: float = 2.0, smoothing: float = 0.05):
        super(FocalLossWithSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        if self.smoothing > 0:
            labels = labels * (1 - self.smoothing) + 0.5 * self.smoothing
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal_weight = (1 - p_t) ** self.gamma
        
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


def rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    R-Drop: Regularized Dropout (Liang et al., NeurIPS'21)
    
    Compute symmetric KL divergence between two forward passes.
    Forces model to be consistent despite different dropout masks.
    
    alpha: weight cho KL divergence term
    """
    p = torch.sigmoid(logits1)
    q = torch.sigmoid(logits2)
    
    # Symmetric KL: 0.5 * (KL(p||q) + KL(q||p))
    # Binary KL divergence
    kl_pq = p * torch.log(p / (q + 1e-8) + 1e-8) + (1 - p) * torch.log((1 - p) / (1 - q + 1e-8) + 1e-8)
    kl_qp = q * torch.log(q / (p + 1e-8) + 1e-8) + (1 - q) * torch.log((1 - q) / (1 - p + 1e-8) + 1e-8)
    
    symmetric_kl = 0.5 * (kl_pq + kl_qp)
    return alpha * symmetric_kl.mean()
