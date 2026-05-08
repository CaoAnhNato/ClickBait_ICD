"""
ICD Model v5 – Modular Pattern-Expert Framework
=================================================
Mở rộng từ ICDv4 với:
1. Shared PhoBERT backbone.
2. 6 Expert Modules chuyên biệt (Style, Shock, Gap, Promo, Analysis, Hardnews).
3. Supervised Router điều phối weights của các experts dựa trên z_base và metadata.
4. Flatten Z và Concat vào Classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# ===========================================================================
# Pooling Modules (Kế thừa từ v4)
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
                # <s> (0), title (1 -> sep1-1), </s> (sep1), </s> (sep2), lead (sep2+1 -> end)
                sep1, sep2 = sep_positions[0].item(), sep_positions[1].item()
                # Title
                title_h = hidden_states[i, 1:sep1]
                if title_h.size(0) > 0:
                    title_pool = torch.mean(title_h, dim=0)
                else:
                    title_pool = torch.zeros(hidden_states.size(-1), device=hidden_states.device)
                
                # Lead
                # Find real end using attention mask
                valid_len = int(attention_mask[i].sum().item())
                # Handle cases where valid_len <= sep2+1
                end_pos = min(valid_len - 1, hidden_states.size(1)) # -1 for the final </s> if exists
                
                lead_h = hidden_states[i, sep2+1:end_pos]
                if lead_h.size(0) > 0:
                    lead_pool = torch.mean(lead_h, dim=0)
                else:
                    lead_pool = torch.zeros(hidden_states.size(-1), device=hidden_states.device)
            else:
                # Fallback: mean over all valid tokens
                valid_len = int(attention_mask[i].sum().item())
                valid_h = hidden_states[i, 1:valid_len-1] if valid_len > 2 else hidden_states[i, :1]
                if valid_h.size(0) > 0:
                    title_pool = lead_pool = torch.mean(valid_h, dim=0)
                else:
                    title_pool = lead_pool = torch.zeros(hidden_states.size(-1), device=hidden_states.device)
                    
            title_pool_list.append(title_pool)
            lead_pool_list.append(lead_pool)

        return torch.stack(title_pool_list), torch.stack(lead_pool_list)

class WeightedLayerPool(nn.Module):
    """Học trọng số kết hợp 4 layers cuối của PhoBERT."""
    def __init__(self, num_layers=4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, all_hidden_states: tuple):
        # Lấy 4 layers cuối
        last_4_layers = torch.stack(all_hidden_states[-4:], dim=0)  # (4, B, L, H)
        norm_weights = F.softmax(self.weights, dim=0)
        # Weighted sum: (4, B, L, H) x (4, 1, 1, 1) -> sum(dim=0) -> (B, L, H)
        weighted_sum = torch.sum(last_4_layers * norm_weights.view(-1, 1, 1, 1), dim=0)
        return weighted_sum

class AttentionPool(nn.Module):
    """Attention pooling để extract 1 vector từ chuỗi hidden states."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: (B, L, H)
        scores = self.attn(hidden_states).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        alpha = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        pooled = torch.sum(hidden_states * alpha, dim=1) # (B, H)
        return pooled

# ===========================================================================
# ICDv5 Modules
# ===========================================================================
class ExpertModule(nn.Module):
    def __init__(self, hidden_size=768, output_size=512):
        super().__init__()
        self.attention_pool = AttentionPool(hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )
        # Head phụ trợ cho Phase 1: predict pattern tag (binary)
        self.aux_head = nn.Linear(output_size, 1)
        
    def forward(self, hidden_states, attention_mask):
        pooled = self.attention_pool(hidden_states, attention_mask)
        expert_out = self.adapter(pooled)
        expert_logits = self.aux_head(expert_out)
        return expert_out, expert_logits

class Router(nn.Module):
    def __init__(self, d_base=3846, num_categories=50, num_sources=200, d_router=512, k_experts=6):
        super().__init__()
        self.cat_emb = nn.Embedding(num_categories, 32)
        self.src_emb = nn.Embedding(num_sources, 32)
        self.pattern_proj = nn.Linear(6, 32)
        
        d_in = d_base + 32 + 32 + 32
        
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_router),
            nn.GELU(),
            nn.LayerNorm(d_router),
            nn.Linear(d_router, k_experts)
        )
        
    def forward(self, z_base, category_id, source_id, pattern_tags):
        cat_e = self.cat_emb(category_id)
        src_e = self.src_emb(source_id)
        pat_e = self.pattern_proj(pattern_tags)
        
        router_in = torch.cat([z_base, cat_e, src_e, pat_e], dim=-1)
        router_logits = self.mlp(router_in)
        router_weights = F.softmax(router_logits, dim=-1)
        
        return router_weights, router_logits

class ClickbaitDetectorV5(nn.Module):
    def __init__(self, model_name_or_path: str, num_categories: int = 50, num_sources: int = 200, use_router: bool = True, sep_token_id: int = 2):
        super().__init__()
        # Backbone
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.output_hidden_states = True
        self.roberta = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        # PhoBERT (RoBERTa) dùng eos_token_id làm sep_token_id
        self.sep_token_id = getattr(self.config, "sep_token_id", getattr(self.config, "eos_token_id", sep_token_id))
        
        # Base Encoder
        self.layer_pool_news = WeightedLayerPool()
        self.seg_pool = SegmentAwarePool(self.sep_token_id)
        
        # ESIM-like interaction
        H = self.config.hidden_size # 768
        self.interaction_mlp = nn.Sequential(
            nn.Linear(4 * H, H),
            nn.GELU(),
            nn.LayerNorm(H)
        )
        
        # Base Dim: u, v, diff, mul (4*H) + CLS (H) + 6 aux = 5*H + 6 = 3846
        self.D_BASE = 5 * H + 6
        
        # Experts
        self.K_EXPERT = 6
        self.D_EXP = 512
        self.D_EXP_TOTAL = self.K_EXPERT * self.D_EXP # 3072
        
        self.experts = nn.ModuleList([ExpertModule(H, self.D_EXP) for _ in range(self.K_EXPERT)])
        
        # Router
        self.use_router = use_router
        if self.use_router:
            self.router = Router(self.D_BASE, num_categories, num_sources, d_router=512, k_experts=self.K_EXPERT)
            
        # Classifier
        self.D_CLS = self.D_BASE + self.D_EXP_TOTAL # 6918
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.D_CLS),
            nn.Linear(self.D_CLS, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
    def _encode_base(self, input_ids_news, attention_mask_news):
        # PhoBERT pass
        out = self.roberta(input_ids_news, attention_mask=attention_mask_news)
        last_hidden = out.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        
        # Layer pool cho title/lead (để giữ rich semantics như v4)
        weighted_h = self.layer_pool_news(out.hidden_states)
        
        # Segment pool
        u, v = self.seg_pool(weighted_h, input_ids_news, attention_mask_news)
        
        # ESIM interaction
        diff = torch.abs(u - v)
        mul = u * v
        interact = self.interaction_mlp(torch.cat([u, v, diff, mul], dim=-1))
        
        # CLS + u + v + diff + mul
        z_text = torch.cat([cls_token, u, v, diff, mul], dim=-1) # 5*H = 3840
        return z_text, last_hidden
        
    def forward(self, input_ids_news, attention_mask_news, category_id, source_id, pattern_tags, soft_label_llm):
        """
        Forward pass.
        Aux tags: tag_shock, tag_lifestyle, tag_listicle, tag_analysis, tag_promo, tag_hardnews
        """
        B = input_ids_news.size(0)
        
        # 1. Base Encoding
        z_text, last_hidden = self._encode_base(input_ids_news, attention_mask_news)
        
        # Aux features: pattern_tags (6)
        # Note: Không nối category_id, source_id trực tiếp mà thông qua router embedding
        # Ở đây z_base chỉ nhận pattern_tags thay cho LLM_scores trong ICDv4 để tạo 3846 dim
        z_base = torch.cat([z_text, pattern_tags], dim=-1) # 3840 + 6 = 3846
        
        # 2. Experts
        expert_outputs = []
        expert_aux_logits = []
        for expert in self.experts:
            e_out, e_aux = expert(last_hidden, attention_mask_news)
            expert_outputs.append(e_out)
            expert_aux_logits.append(e_aux)
            
        Z = torch.stack(expert_outputs, dim=1) # (B, 6, 512)
        expert_aux_logits = torch.cat(expert_aux_logits, dim=-1) # (B, 6)
        
        # 3. Router
        if self.use_router:
            router_weights, router_logits = self.router(z_base, category_id, source_id, pattern_tags)
            # Weighted mix - element wise multiplication along expert dim, then flatten
            Z_weighted = Z * router_weights.unsqueeze(-1) # (B, 6, 512)
            z_flat = Z_weighted.view(B, -1) # (B, 3072)
        else:
            # Ablation: No Router
            router_weights = None
            router_logits = None
            z_flat = Z.view(B, -1) # (B, 3072)
            
        # 4. Fusion & Classification
        z_concat = torch.cat([z_base, z_flat], dim=-1) # 3846 + 3072 = 6918
        logits = self.classifier(z_concat)
        
        return {
            "logits": logits,
            "z_base": z_base,
            "router_weights": router_weights,
            "router_logits": router_logits,
            "expert_aux_logits": expert_aux_logits
        }
        
    def freeze_backbone_layers(self, freeze_until: int):
        """Freeze layers 0 -> freeze_until-1."""
        if freeze_until <= 0:
            return
            
        # Freeze embeddings
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze encoder layers
        for idx in range(freeze_until):
            for param in self.roberta.encoder.layer[idx].parameters():
                param.requires_grad = False

    def get_parameter_groups(self, base_lr: float, lr_decay: float = 0.95):
        """Phân nhóm params để áp dụng layer-wise learning rate decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        
        # Lấy số layer thực tế
        num_layers = self.config.num_hidden_layers
        
        # Tham số backbone (chỉ lấy những cái requires_grad)
        for layer_idx in range(num_layers):
            layer = self.roberta.encoder.layer[layer_idx]
            lr = base_lr * (lr_decay ** (num_layers - layer_idx))
            
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.01,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            
        # Tham số custom head (không có decay LR, giữ nguyên base_lr)
        # Các module không thuộc roberta
        head_params = [p for n, p in self.named_parameters() if "roberta" not in n and p.requires_grad]
        
        optimizer_grouped_parameters += [
            {
                "params": head_params,
                "weight_decay": 0.01,
                "lr": base_lr,
            }
        ]
        
        return optimizer_grouped_parameters
