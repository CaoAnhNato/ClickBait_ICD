import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class WordLevelAttention(nn.Module):
    def __init__(self, hidden_size):
        super(WordLevelAttention, self).__init__()
        # Ma trận W_at và w_at như trong công thức toán học
        self.W_at = nn.Linear(hidden_size, hidden_size)
        self.w_at = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: (batch_size, seq_len, hidden_size) - Output từ PhoBERT
        attention_mask: (batch_size, seq_len) - 1 cho token thật, 0 cho padding token
        """
        # Tính điểm u_i = tanh(W_at * h_i + b_at)
        u_i = torch.tanh(self.W_at(hidden_states)) # Shape: (batch_size, seq_len, hidden_size)
        
        # Chiếu xuống 1 chiều: w_at^T * u_i
        scores = self.w_at(u_i).squeeze(-1) # Shape: (batch_size, seq_len)
        
        # Masking: Gán điểm cực kỳ thấp cho các padding tokens để softmax trả về 0
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        
        # Tính trọng số attention (alpha)
        alpha = torch.softmax(scores, dim=-1) # Shape: (batch_size, seq_len)
        
        # Nhân trọng số alpha với hidden_states và tính tổng bằng Batch Matrix Multiplication (tối ưu hóa)
        # alpha.unsqueeze(1) có shape (batch_size, 1, seq_len)
        # BMM( (B, 1, S), (B, S, H) ) -> (B, 1, H)
        e_out = torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1) # Shape: (batch_size, hidden_size)
        
        return e_out, alpha

class ContentModelingModule(nn.Module):
    def __init__(self, model_name="vinai/phobert-base"):
        super(ContentModelingModule, self).__init__()
        # Load PhoBERT backbone
        self.phobert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.phobert.config.hidden_size # 768 cho phobert-base
        
        # Hai attention layer độc lập để tránh information leakage
        self.title_attention = WordLevelAttention(self.hidden_size)
        self.lead_attention = WordLevelAttention(self.hidden_size)

    def forward(self, title_input_ids, title_attention_mask, lead_input_ids, lead_attention_mask):
        # 1. Đi qua PhoBERT để lấy ma trận ẩn H (hidden states)
        title_outputs = self.phobert(input_ids=title_input_ids, attention_mask=title_attention_mask)
        lead_outputs = self.phobert(input_ids=lead_input_ids, attention_mask=lead_attention_mask)
        
        # Trích xuất last_hidden_state chứa biểu diễn ngữ cảnh
        H_title = title_outputs.last_hidden_state # Shape: (batch_size, N, 768)
        H_lead = lead_outputs.last_hidden_state   # Shape: (batch_size, P, 768)
        
        # 2. Đi qua lớp Word-Level Attention để lấy vector tổng hợp e
        e_title, alpha_title = self.title_attention(H_title, title_attention_mask)
        e_lead, alpha_lead = self.lead_attention(H_lead, lead_attention_mask)
        
        # Trả về cả H (để dùng cho Interaction Module phía sau) và e (để tính Content Score)
        return H_title, H_lead, e_title, e_lead

class CharacterLevelAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(CharacterLevelAttention, self).__init__()
        # Ma trận W_at và w_at cho mức độ ký tự
        self.W_at = nn.Linear(hidden_size, hidden_size)
        self.w_at = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len)
        """
        # Tính điểm u_i = tanh(W_at * h_i)
        u_i = torch.tanh(self.W_at(hidden_states)) # Shape: (batch_size, seq_len, hidden_size)
        
        # Chiếu xuống 1 chiều: w_at^T * u_i
        scores = self.w_at(u_i).squeeze(-1) # Shape: (batch_size, seq_len)
        
        # Masking: Gán điểm cực kỳ thấp cho các padding tokens
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        
        # Tính trọng số attention (alpha)
        alpha = torch.softmax(scores, dim=-1) # Shape: (batch_size, seq_len)
        
        # Tổng hợp thành vector phong cách bằng Batched Matrix Multiplication
        e_out = torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1) # Shape: (batch_size, hidden_size)
        
        return e_out, alpha

class StyleModelingModule(nn.Module):
    """
    Module trích xuất đặc trưng văn phong từ chuỗi ký tự của tiêu đề.
    
    [v2] Giảm d_c từ 128→64 để giảm noise, phù hợp với thông tin style thực tế
    không quá phức tạp (chủ yếu là pattern dấu câu, viết hoa, cảm thán).
    Lưu ý: vocab_size phải phù hợp với CharTokenizer bên train_ICD.py.
    """
    def __init__(self, vocab_size: int, d_c: int = 64, nhead: int = 4, num_layers: int = 2):
        super(StyleModelingModule, self).__init__()
        # Lớp Embedding để chuyển ID ký tự thành vector nhúng
        self.char_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_c)
        
        # Transformer Encoder để học biểu diễn ngữ cảnh cục bộ của các ký tự
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_c, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Lớp Attention cấp độ ký tự
        self.char_attention = CharacterLevelAttention(hidden_size=d_c)

    def forward(self, char_input_ids: torch.Tensor, char_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        char_input_ids: (batch_size, max_char_len)
        char_attention_mask: (batch_size, max_char_len)
        """
        # 1. Chuyển đổi ID ký tự thành embedding
        # Shape: (batch_size, max_char_len, d_c)
        x = self.char_embedding(char_input_ids)
        
        # 2. Đi qua Transformer Encoder
        # Tham số src_key_padding_mask nhận True tại vị trí mask (padding)
        padding_mask = (char_attention_mask == 0)
        
        # Lấy trạng thái ẩn của các ký tự (batch_size, max_char_len, d_c)
        H_char = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 3. Áp dụng Attention để tính tổng có trọng số
        # e_style: (batch_size, d_c)
        e_style, _ = self.char_attention(H_char, char_attention_mask)
        
        return e_style

class InteractionModelingModule(nn.Module):
    """
    Module tương tác chéo giữa Title và Lead sử dụng Co-Attention.
    
    [v2] Thêm projection layers (2*hidden→hidden) để giảm chiều đầu ra,
    tránh dot product explosion khi tổng hợp score ở ClickbaitDetectionModel.
    Output bây giờ là r_title, r_lead có cùng dimension hidden_size (768),
    thay vì 2*hidden_size (1536) như v1.
    Ref: ORCD (WWW'26) - sử dụng cross-attention + projection trước khi aggregation.
    """
    def __init__(self, hidden_size: int = 768):
        super(InteractionModelingModule, self).__init__()
        # Khởi tạo ma trận trọng số tương tác W_c
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # [v2] Projection layers để giảm chiều từ 2*hidden_size → hidden_size
        # Thay vì output trực tiếp concatenated vector 1536-dim, ta project xuống 768-dim
        # Điều này đảm bảo output cùng scale với e_title, e_lead từ Content Module
        self.title_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        self.lead_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

    def forward(self, H_title: torch.Tensor, title_mask: torch.Tensor, H_lead: torch.Tensor, lead_mask: torch.Tensor):
        """
        H_title: (batch_size, N, hidden_size)
        title_mask: (batch_size, N)
        H_lead: (batch_size, P, hidden_size)
        lead_mask: (batch_size, P)
        """
        # 1. Tính ma trận tương đồng (affinity matrix) S
        # H_title * W_c -> (batch_size, N, hidden_size)
        H_title_Wc = self.W_c(H_title)
        
        # S = (H_title * W_c) * H_lead^T -> (batch_size, N, P)
        # torch.bmm nhân 2 batch ma trận. H_lead.transpose(1, 2) có shape (batch_size, hidden_size, P)
        S = torch.bmm(H_title_Wc, H_lead.transpose(1, 2))
        
        # 2. Xử lý Masking cho Co-attention (Loại bỏ padding token)
        # Mở rộng lead_mask thành (batch_size, 1, P)
        lead_mask_exp = lead_mask.unsqueeze(1)
        
        # Masking padding của đoạn dẫn (lead) bằng cách gắn giá trị rất âm
        S_T = S.masked_fill(lead_mask_exp == 0, -1e4)
        # Tính A_T: chú ý của đoạn dẫn đối với tiêu đề
        A_T = torch.softmax(S_T, dim=-1) # Shape: (batch_size, N, P)
        
        # Masking padding của tiêu đề (title). Lưu ý ta phải transpose S để có shape (batch_size, P, N)
        # title_mask.unsqueeze(1) có shape (batch_size, 1, N)
        S_L = S.transpose(1, 2).masked_fill(title_mask.unsqueeze(1) == 0, -1e4)
        # Tính A_L: chú ý của tiêu đề đối với đoạn dẫn
        A_L = torch.softmax(S_L, dim=-1) # Shape: (batch_size, P, N)
        
        # Lưu trữ tạm A_T và A_L để có thể test xem masking có đúng hay không
        self._A_T_temp = A_T
        self._A_L_temp = A_L
        
        # 3. Tính toán biểu diễn tương tác chéo
        # C_title = A_T x H_lead -> Shape: (batch_size, N, hidden_size)
        C_title = torch.bmm(A_T, H_lead)
        # C_lead = A_L x H_title -> Shape: (batch_size, P, hidden_size)
        C_lead = torch.bmm(A_L, H_title)
        
        # 4. Nối (concatenate) ma trận gốc với ma trận tương tác
        # R_title: (batch_size, N, 2 * hidden_size)
        R_title = torch.cat([H_title, C_title], dim=-1)
        # R_lead: (batch_size, P, 2 * hidden_size)
        R_lead = torch.cat([H_lead, C_lead], dim=-1)
        
        # 5. Áp dụng Mean-Pooling dọc theo chiều sequence (chiều N và P) có tính đến mask bằng BMM
        # Tính tổng dọc theo sequence (trọng số của padding là 0, của token thực là 1)
        # BMM( (B, 1, seq_len), (B, seq_len, 2 * hidden_size) ) -> (B, 1, 2 * hidden_size) -> squeeze(1) -> (B, 2 * hidden_size)
        sum_title = torch.bmm(title_mask.unsqueeze(1).float(), R_title).squeeze(1)
        sum_lead = torch.bmm(lead_mask.unsqueeze(1).float(), R_lead).squeeze(1)
        
        # Đếm số token thực sự. Dùng clamp_min=1e-4 để tránh lỗi chia cho 0
        len_title = torch.clamp(torch.sum(title_mask, dim=1, keepdim=True), min=1e-4)
        len_lead = torch.clamp(torch.sum(lead_mask, dim=1, keepdim=True), min=1e-4)
        
        # Chia trung bình chỉ trên các token thực → (batch_size, 2 * hidden_size)
        r_title_raw = sum_title / len_title
        r_lead_raw = sum_lead / len_lead
        
        # [v2] 6. Projection: giảm chiều từ 2*hidden_size → hidden_size
        # Đảm bảo output cùng scale với e_title, e_lead (768-dim) thay vì 1536-dim
        r_title = self.title_projection(r_title_raw) # (batch_size, hidden_size)
        r_lead = self.lead_projection(r_lead_raw)    # (batch_size, hidden_size)
        
        return r_title, r_lead

class ClickbaitDetectionModel(nn.Module):
    """
    Model chính tổng hợp 3 module: Content, Style, Interaction.
    
    [v2] Thay đổi kiến trúc tổng hợp:
    - v1: logits = alpha_t*y_t + alpha_b*y_b + alpha_s*y_s + alpha_r*y_r (dot product explosion)
    - v2: Concatenation [e_title, e_lead, e_style_proj, r_title, r_lead, interaction_feats] → MLP
    
    Interaction features bao gồm element-wise difference và element-wise product
    giữa r_title và r_lead (theo ESIM pattern, đã được chứng minh hiệu quả
    trong NLI/sentence pair tasks).
    
    Ref: ORCD (WWW'26) - concatenation of multiple representation vectors → MLP
    Ref: ESIM (ACL'17) - element-wise difference/product for sentence interaction
    """
    def __init__(self, vocab_size: int, content_model_name="vinai/phobert-base", hidden_size=768, d_c=64):
        super(ClickbaitDetectionModel, self).__init__()
        # 1. Khởi tạo 3 module con
        self.content_module = ContentModelingModule(model_name=content_model_name)
        self.style_module = StyleModelingModule(vocab_size=vocab_size, d_c=d_c)
        self.interaction_module = InteractionModelingModule(hidden_size=hidden_size)
        
        # [v2] 2. Projection layer cho style embedding: d_c → hidden_size
        # Để đồng bộ dimension trước khi concatenate
        self.style_projection = nn.Linear(d_c, hidden_size)
        
        # [v2] 3. MLP Classifier thay cho learnable alpha weights
        # Input features:
        #   - e_title: (hidden_size) = 768
        #   - e_lead: (hidden_size) = 768
        #   - e_style_proj: (hidden_size) = 768
        #   - r_title: (hidden_size) = 768  (đã projected trong InteractionModule)
        #   - r_lead: (hidden_size) = 768
        #   - r_diff: (hidden_size) = 768  (element-wise difference: |r_title - r_lead|)
        #   - r_prod: (hidden_size) = 768  (element-wise product: r_title * r_lead)
        # Total: 7 * hidden_size = 5376
        classifier_input_dim = 7 * hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(0.3),
            nn.Linear(classifier_input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask):
        # Bước 1: Trích xuất đặc trưng từ Content Module
        # H_title, H_lead có shape (batch_size, seq_len, hidden_size)
        # e_title, e_lead có shape (batch_size, hidden_size)
        H_title, H_lead, e_title, e_lead = self.content_module(title_ids, title_mask, lead_ids, lead_mask)
        
        # Bước 2: Trích xuất đặc trưng từ Style Module
        # e_style có shape (batch_size, d_c)
        e_style = self.style_module(char_ids, char_mask)
        
        # [v2] Project style embedding lên cùng dimension với content embeddings
        e_style_proj = self.style_projection(e_style) # (batch_size, hidden_size)
        
        # Bước 3: Tính toán tương tác chéo từ Interaction Module
        # [v2] r_title, r_lead giờ đã có shape (batch_size, hidden_size) thay vì (batch_size, 2*hidden_size)
        r_title, r_lead = self.interaction_module(H_title, title_mask, H_lead, lead_mask)
        
        # [v2] Bước 4: Tính interaction features theo ESIM pattern
        # Element-wise difference: nắm bắt sự khác biệt ngữ nghĩa giữa title/lead qua co-attention
        r_diff = torch.abs(r_title - r_lead) # (batch_size, hidden_size)
        # Element-wise product: nắm bắt sự tương đồng ngữ nghĩa
        r_prod = r_title * r_lead # (batch_size, hidden_size)
        
        # [v2] Bước 5: Concatenate tất cả features → MLP Classifier
        # Thay vì: logits = alpha_t*y_t + alpha_b*y_b + alpha_s*y_s + alpha_r*y_r (v1 - bị explosion)
        combined = torch.cat([
            e_title,        # Content representation của title
            e_lead,         # Content representation của lead
            e_style_proj,   # Style representation (projected)
            r_title,        # Interaction-aware title representation
            r_lead,         # Interaction-aware lead representation
            r_diff,         # Semantic difference (clickbait signal)
            r_prod          # Semantic similarity
        ], dim=-1) # (batch_size, 7 * hidden_size)
        
        logits = self.classifier(combined) # (batch_size, 1)
        
        # Trả về logits cho dự đoán, e_title và e_lead cho Joint Loss
        return logits, e_title, e_lead

class FocalLoss(nn.Module):
    """
    Focal Loss cho bài toán binary classification mất cân bằng.
    
    Ref: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    Ref: CoGate-LSTM (2510.17018) - sử dụng weighted focal loss cho toxic text classification
    
    Với dataset ViClickbait: 68.8% non-clickbait, 31.2% clickbait
    → alpha=0.6 (weight cao hơn cho minority class - clickbait)
    → gamma=2.0 (focus vào hard examples, giảm loss cho easy examples)
    
    Công thức: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.6, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        logits: (batch_size, 1) - Raw logits chưa qua sigmoid
        labels: (batch_size, 1) - Binary labels (0 hoặc 1)
        """
        # Tính BCE loss element-wise (không reduce)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        # Tính p_t (probability of correct class)
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        
        # Tính alpha_t (class-specific weight)
        # alpha cho clickbait (label=1), (1-alpha) cho non-clickbait (label=0)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        
        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()

class JointLoss(nn.Module):
    """
    Joint Loss = Focal Loss + lambda * Contrastive Loss
    
    [v2] Thay đổi so với v1:
    - Sử dụng FocalLoss thay vì BCEWithLogitsLoss (xử lý class imbalance)
    - Giảm margin: 1.0 → 0.5 (phù hợp với word overlap gap ~12% trong dataset)
    - Thêm learnable temperature parameter cho contrastive loss
    
    Ref: RoCliCo (2310.06540) - sử dụng contrastive learning với temperature scaling
    Ref: CVM (IJCAI'22) - contrastive variational modelling cho clickbait
    """
    def __init__(self, margin=0.5, lambda_weight=0.3, focal_alpha=0.6, focal_gamma=2.0):
        super(JointLoss, self).__init__()
        self.margin = margin
        self.lambda_weight = lambda_weight
        
        # [v2] Focal Loss thay vì BCE
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # [v2] Learnable temperature cho contrastive loss (init=0.07, theo SimCLR)
        # Temperature thấp → phân tách mạnh hơn, temperature cao → phân tách mềm hơn
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, e_title: torch.Tensor, e_lead: torch.Tensor):
        """
        logits: (batch_size, 1) - Logits chưa qua sigmoid
        labels: (batch_size, 1) - Nhãn thực tế (0 hoặc 1)
        e_title: (batch_size, hidden_size) - Đặc trưng content tiêu đề
        e_lead: (batch_size, hidden_size) - Đặc trưng content đoạn dẫn
        """
        # 1. Tính Focal Loss (thay vì BCE)
        cls_loss = self.focal_loss(logits, labels)
        
        # 2. Tính toán Contrastive Loss (L_CL) với temperature scaling
        # [v2] Temperature scaling: chia cosine similarity cho temperature
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=1.0)
        
        # Tính cosine similarity và scale bằng temperature
        cosine_sim = F.cosine_similarity(e_title, e_lead, dim=-1).unsqueeze(-1)
        
        # Khoảng cách D_cos: Khi similarity cao -> D_cos thấp (ngược lại)
        D_cos = 1 - cosine_sim
        
        # Nhãn 0 (Non-Clickbait): title và lead phải có tính tương đồng cao -> D_cos phải thấp
        # Phạt D_cos nếu nó lớn (tức là 2 vector khác nhau)
        loss_0 = (1 - labels) * (D_cos / temperature)
        
        # Nhãn 1 (Clickbait): title và lead lạc đề hoặc tạo khoảng trống -> D_cos phải lớn (vượt margin)
        # [v2] Margin giảm từ 1.0 → 0.5: phù hợp hơn với word overlap gap thực tế (~12%)
        loss_1 = labels * torch.clamp(self.margin - D_cos, min=0.0) / temperature
        
        # Tổng hợp Contrastive Loss của lô (batch)
        L_CL = torch.mean(loss_0 + loss_1)
        
        # 3. Tính Total Loss
        total_loss = cls_loss + self.lambda_weight * L_CL
        
        return total_loss, cls_loss, L_CL