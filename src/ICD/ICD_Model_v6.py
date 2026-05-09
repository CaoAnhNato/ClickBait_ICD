import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class SegmentAwarePool(nn.Module):
    """
    (Optional) ESIM-like Interaction module
    """
    def __init__(self, sep_token_id):
        super().__init__()
        self.sep_token_id = sep_token_id

    def forward(self, hidden_states, input_ids):
        # hidden_states: [B, SeqLen, H]
        B, SeqLen, H = hidden_states.size()
        
        sep_masks = (input_ids == self.sep_token_id)
        
        title_pool = torch.zeros(B, H, device=hidden_states.device)
        lead_pool = torch.zeros(B, H, device=hidden_states.device)
        
        for i in range(B):
            sep_idx = torch.where(sep_masks[i])[0]
            if len(sep_idx) >= 1:
                first_sep = sep_idx[0].item()
                # Title: 1 to first_sep-1
                if first_sep > 1:
                    title_h = hidden_states[i, 1:first_sep]
                    title_pool[i] = title_h.mean(dim=0)
                
                # Lead: first_sep+1 to last_sep-1
                if len(sep_idx) >= 2:
                    last_sep = sep_idx[-1].item()
                    if last_sep > first_sep + 1:
                        lead_h = hidden_states[i, first_sep+1:last_sep]
                        lead_pool[i] = lead_h.mean(dim=0)
                else:
                    # if only 1 sep token, treat the rest as lead
                    if SeqLen > first_sep + 1:
                        lead_h = hidden_states[i, first_sep+1:]
                        lead_pool[i] = lead_h.mean(dim=0)
        
        return title_pool, lead_pool


class ClickbaitDetectorV6(nn.Module):
    """
    ICDv6 Model - Streamlined
    Supports two variants:
    - variant="simple": Use only [CLS] token (C0 Baseline)
    - variant="esim": Use [CLS] + title/lead mean pooling + diff + prod (ESIM Baseline)
    """
    def __init__(
        self,
        model_name="vinai/phobert-base-v2",
        sep_token_id=2, # default for phobert-base-v2 (eos_token_id is 2)
        dropout_rate=0.2,
        variant="simple" # "simple" or "esim"
    ):
        super().__init__()
        self.variant = variant
        self.sep_token_id = sep_token_id
        
        config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name, config=config)
        
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        
        if self.variant == "simple":
            clf_input_dim = hidden_size
        elif self.variant == "esim":
            self.segment_pool = SegmentAwarePool(sep_token_id)
            clf_input_dim = hidden_size * 5 # CLS, title, lead, diff, prod
        else:
            raise ValueError(f"Unknown variant: {variant}")
            
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids_news, attention_mask_news):
        """
        input_ids_news: [B, SeqLen]
        attention_mask_news: [B, SeqLen]
        """
        outputs = self.phobert(
            input_ids=input_ids_news,
            attention_mask=attention_mask_news,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state # [B, SeqLen, H]
        cls_output = hidden_states[:, 0, :] # [B, H]
        
        if self.variant == "simple":
            features = cls_output
        elif self.variant == "esim":
            title_pool, lead_pool = self.segment_pool(hidden_states, input_ids_news)
            diff = torch.abs(title_pool - lead_pool)
            prod = title_pool * lead_pool
            features = torch.cat([cls_output, title_pool, lead_pool, diff, prod], dim=1)
            
        features = self.dropout(features)
        logits = self.classifier(features)
        
        return logits
        
    def freeze_backbone_layers(self, freeze_until=8):
        """
        Freeze embeddings and bottom N layers of PhoBERT
        """
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
            
        for i in range(freeze_until):
            for param in self.phobert.encoder.layer[i].parameters():
                param.requires_grad = False
                
    def get_parameter_groups(self, lr=2e-5, lr_decay=0.95):
        """
        Layer-wise learning rate decay for PhoBERT
        """
        parameter_groups = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name or 'segment_pool' in name:
                classifier_params.append(param)
        
        if classifier_params:
            parameter_groups.append({
                'params': classifier_params,
                'lr': lr * 10.0 # classifier learns faster
            })
            
        # PhoBERT layers
        num_layers = self.phobert.config.num_hidden_layers
        for i in range(num_layers - 1, -1, -1):
            layer_params = []
            for name, param in self.phobert.encoder.layer[i].named_parameters():
                if param.requires_grad:
                    layer_params.append(param)
            if layer_params:
                parameter_groups.append({
                    'params': layer_params,
                    'lr': lr * (lr_decay ** (num_layers - 1 - i))
                })
                
        # Embeddings and Pooler
        emb_pool_params = []
        for name, param in self.phobert.named_parameters():
            if 'encoder.layer' not in name and param.requires_grad:
                emb_pool_params.append(param)
                
        if emb_pool_params:
            parameter_groups.append({
                'params': emb_pool_params,
                'lr': lr * (lr_decay ** num_layers)
            })
            
        return parameter_groups
