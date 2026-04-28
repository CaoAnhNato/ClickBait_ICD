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
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        
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
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Tính trọng số attention (alpha)
        alpha = torch.softmax(scores, dim=-1) # Shape: (batch_size, seq_len)
        
        # Tổng hợp thành vector phong cách bằng Batched Matrix Multiplication
        e_out = torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1) # Shape: (batch_size, hidden_size)
        
        return e_out, alpha

class StyleModelingModule(nn.Module):
    def __init__(self, vocab_size: int, d_c: int = 128, nhead: int = 4, num_layers: int = 2):
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
    def __init__(self, hidden_size: int = 768):
        super(InteractionModelingModule, self).__init__()
        # Khởi tạo ma trận trọng số tương tác W_c
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)

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
        S_T = S.masked_fill(lead_mask_exp == 0, -1e9)
        # Tính A_T: chú ý của đoạn dẫn đối với tiêu đề
        A_T = torch.softmax(S_T, dim=-1) # Shape: (batch_size, N, P)
        
        # Masking padding của tiêu đề (title). Lưu ý ta phải transpose S để có shape (batch_size, P, N)
        # title_mask.unsqueeze(1) có shape (batch_size, 1, N)
        S_L = S.transpose(1, 2).masked_fill(title_mask.unsqueeze(1) == 0, -1e9)
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
        
        # Đếm số token thực sự. Dùng clamp_min=1e-9 để tránh lỗi chia cho 0
        len_title = torch.clamp(torch.sum(title_mask, dim=1, keepdim=True), min=1e-9)
        len_lead = torch.clamp(torch.sum(lead_mask, dim=1, keepdim=True), min=1e-9)
        
        # Chia trung bình chỉ trên các token thực
        r_title = sum_title / len_title # (batch_size, 2 * hidden_size)
        r_lead = sum_lead / len_lead    # (batch_size, 2 * hidden_size)
        
        return r_title, r_lead

class ClickbaitDetectionModel(nn.Module):
    def __init__(self, vocab_size: int, content_model_name="vinai/phobert-base", hidden_size=768, d_c=128):
        super(ClickbaitDetectionModel, self).__init__()
        # 1. Khởi tạo 3 module con
        self.content_module = ContentModelingModule(model_name=content_model_name)
        self.style_module = StyleModelingModule(vocab_size=vocab_size, d_c=d_c)
        self.interaction_module = InteractionModelingModule(hidden_size=hidden_size)
        
        # 2. Tạo các lớp tuyến tính để tính điểm riêng lẻ
        # W_t: content title -> in: 768, out: 1
        self.W_t = nn.Linear(hidden_size, 1)
        # W_b: content lead -> in: 768, out: 1
        self.W_b = nn.Linear(hidden_size, 1)
        # W_s: style -> in: 128, out: 1
        self.W_s = nn.Linear(d_c, 1)
        
        # 3. Khởi tạo các siêu tham số có thể học (learnable weights) cho việc tổng hợp
        self.alpha_t = nn.Parameter(torch.ones(1))
        self.alpha_b = nn.Parameter(torch.ones(1))
        self.alpha_s = nn.Parameter(torch.ones(1))
        self.alpha_r = nn.Parameter(torch.ones(1))

    def forward(self, title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask):
        # Bước 1: Trích xuất đặc trưng từ Content Module
        # H_title, H_lead có shape (batch_size, seq_len, hidden_size)
        # e_title, e_lead có shape (batch_size, hidden_size)
        H_title, H_lead, e_title, e_lead = self.content_module(title_ids, title_mask, lead_ids, lead_mask)
        
        # Bước 2: Trích xuất đặc trưng từ Style Module
        # e_style có shape (batch_size, d_c)
        e_style = self.style_module(char_ids, char_mask)
        
        # Bước 3: Tính toán tương tác chéo từ Interaction Module
        # r_title, r_lead có shape (batch_size, 2 * hidden_size)
        r_title, r_lead = self.interaction_module(H_title, title_mask, H_lead, lead_mask)
        
        # Bước 4: Tính các điểm số riêng biệt
        y_t = self.W_t(e_title) # (batch_size, 1)
        y_b = self.W_b(e_lead)  # (batch_size, 1)
        y_s = self.W_s(e_style) # (batch_size, 1)
        
        # Điểm tương tác y_r bằng tích vô hướng của r_title và r_lead
        # dim=-1 để tính tổng theo chiều feature, keepdim=True để giữ shape (batch_size, 1)
        y_r = torch.sum(r_title * r_lead, dim=-1, keepdim=True) 
        
        # Bước 5: Tổng hợp điểm logits
        logits = self.alpha_t * y_t + self.alpha_b * y_b + self.alpha_s * y_s + self.alpha_r * y_r
        
        # Áp dụng hàm sigmoid để lấy xác suất [0, 1]
        preds = torch.sigmoid(logits)
        
        # Trả về preds cho dự đoán, e_title và e_lead cho Joint Loss
        return preds, e_title, e_lead

class JointLoss(nn.Module):
    def __init__(self, margin=1.0, lambda_weight=0.3):
        super(JointLoss, self).__init__()
        self.margin = margin
        self.lambda_weight = lambda_weight
        # Hàm Binary Cross Entropy cho dự đoán chính
        self.bce = nn.BCELoss()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, e_title: torch.Tensor, e_lead: torch.Tensor):
        """
        preds: (batch_size, 1) - Xác suất dự đoán
        labels: (batch_size, 1) - Nhãn thực tế (0 hoặc 1)
        e_title: (batch_size, hidden_size) - Đặc trưng content tiêu đề
        e_lead: (batch_size, hidden_size) - Đặc trưng content đoạn dẫn
        """
        # 1. Tính toán loss dự đoán chính (BCE Loss)
        bce_loss = self.bce(preds, labels)
        
        # 2. Tính toán Contrastive Loss (L_CL)
        # Tính khoảng cách Cosine
        # cosine_similarity trả về shape (batch_size,), cần unsqueeze để đồng bộ với labels (batch_size, 1)
        cosine_sim = F.cosine_similarity(e_title, e_lead, dim=-1).unsqueeze(-1) 
        
        # Khoảng cách D_cos: Khi similarity cao -> D_cos thấp (ngược lại)
        D_cos = 1 - cosine_sim
        
        # Nhãn 0 (Non-Clickbait): title và lead phải có tính tương đồng cao -> D_cos phải thấp
        # Phạt D_cos nếu nó lớn (tức là 2 vector khác nhau)
        loss_0 = (1 - labels) * D_cos
        
        # Nhãn 1 (Clickbait): title và lead lạc đề hoặc tạo khoảng trống -> D_cos phải lớn (vượt margin)
        # Phạt nếu D_cos nhỏ hơn margin
        loss_1 = labels * torch.clamp(self.margin - D_cos, min=0.0)
        
        # Tổng hợp Contrastive Loss của lô (batch)
        L_CL = torch.mean(loss_0 + loss_1)
        
        # 3. Tính Total Loss
        total_loss = bce_loss + self.lambda_weight * L_CL
        
        return total_loss, bce_loss, L_CL