# ICDv5 Modular Pattern-Expert Framework – Implementation & Evaluation Plan

Project: **Capstone Project – Vietnamese Clickbait Detection**  
Models: **ICDv3, ICDv4, ICDv5 (proposed)**  
Codebase layout: see `project_structure.yaml`.

Mục tiêu file này:
- Định nghĩa rõ ràng **các bước từ Data → Reasoning → Pattern Tags → Train ICDv5 → Evaluate & Compare**.
- Thiết kế ICDv5 theo hướng **modular pattern-experts + supervised router**, tất cả tham số đều **trainable**, không hard-coded lexicon/thres.
- Mô tả chi tiết **I/O tensor shape**, tránh overflow & dimension mismatch khi implement.
- Bám sát kiến trúc ICDv3/ICDv4 hiện có (`src/ICD/ICD_Model_v3.py`, `src/ICD/ICD_Model_v4.py`).

---

## 1. Data Pipeline

### 1.1. Nguồn dữ liệu & vị trí

- Dữ liệu chính: thư mục `data/processed/cleaned/`.
  - `train.csv`, `validate.csv`, `test.csv`: đã được split và làm sạch.
  - Schema giả định (khớp `Cleaned_Clickbait_Dataset.csv`):
    - `id`: mã bài báo (vd: `article_3551`).
    - `title`: tiêu đề.
    - `lead_paragraph`: đoạn mở đầu / tóm tắt.
    - `category`: chuyên mục (Giải trí & Showbiz, Kinh doanh & Kinh tế,…).
    - `source`: tên báo.
    - `label`: `clickbait` / `non-clickbait`.
    - Các cột phụ khác (url, thumbnail, province, is_noise, …) có thể bỏ qua cho ICDv5.

### 1.2. Bước 1 – Chuẩn hóa schema

**File đề xuất:** `data/processed/cleaned/prepare_icdv5_data.py`.

Nhiệm vụ:
1. Đọc `train.csv`, `validate.csv`, `test.csv`.
2. Chuẩn hóa label về `0/1`:
   - `clickbait` → `1`, `non-clickbait` → `0`.
3. Chọn subset cột cần thiết:
   - `id`, `title`, `lead_paragraph`, `category`, `source`, `label`.
4. Lưu lại thành:
   - `data/processed/cleaned/icdv5_train_base.parquet`.
   - `data/processed/cleaned/icdv5_valid_base.parquet`.
   - `data/processed/cleaned/icdv5_test_base.parquet`.

Output format (Parquet) giúp load nhanh khi training.

### 1.3. Bước 2 – Sinh reasoning & pattern tags bằng LLM (offline)

**Mục tiêu:** tạo các trường bổ sung cho mỗi sample nhưng vẫn đảm bảo mọi thứ dùng trong model là trainable embedding, không hard-coded feature.

**Directory:** `data/processed/icdv5/` (mới).

#### 1.3.1. Reasoning

- Tái sử dụng hoặc mở rộng logic trong `src/ICD/reasoning/` (đang dùng cho ICDv4):
  - `generate_reasoning.py` → sinh `agree_reason`, `disagree_reason`, `rating_init`, `rating_agree`, `rating_disagree` theo SORG/ORCD-style.
- Input: `(title, lead_paragraph)`.
- Output (mỗi sample):
  - `reason_agree_vi`: đoạn tiếng Việt giải thích tại sao headline **là clickbait**.
  - `reason_disagree_vi`: đoạn tiếng Việt giải thích tại sao headline **không phải clickbait**.
  - `score_init`, `score_agree`, `score_disagree` ∈ [0,100].

Lưu thành file:
- `icdv5_train_reasoning.parquet`, `icdv5_valid_reasoning.parquet`, `icdv5_test_reasoning.parquet`.

#### 1.3.2. Pattern tags (multi-label)

**File đề xuất:** `src/ICD/reasoning/generate_pattern_tags.py`.

Prompt LLM (đã dùng offline, không chạy trong training): gán các tag ngữ nghĩa cho từng bài:
- `PATTERN_SHOCK`: crime/sex/disaster, hacker, chiến tranh,…
- `PATTERN_LIFESTYLE`: showbiz, đời sống, human story.
- `PATTERN_LISTICLE`: "X lý do", "10 cách", tips.
- `PATTERN_ANALYSIS`: bài phân tích chính sách / kinh tế / khoa học, dạng Q&A nhưng serious.
- `PATTERN_PROMO`: sự kiện, quảng bá, brand story.
- `PATTERN_HARDNEWS`: tin ngắn, thời sự trung lập.

Output cột:
- `tag_shock`, `tag_lifestyle`, `tag_listicle`, `tag_analysis`, `tag_promo`, `tag_hardnews` ∈ {0,1}.

Lưu vào:
- `icdv5_train_patterns.parquet`, `icdv5_valid_patterns.parquet`, `icdv5_test_patterns.parquet`.

> Lưu ý: các tag này **chỉ dùng làm supervision cho router & experts**, không đưa trực tiếp vào mô hình như feature one-hot. Router/experts học embedding & quyết định từ dữ liệu.

### 1.4. Bước 3 – Merge tất cả thành dataset huấn luyện ICDv5

**File đề xuất:** `data/processed/icdv5/build_icdv5_dataset.py`.

- Join theo `id`:
  - base (`_base.parquet`) + reasoning (`_reasoning.parquet`) + patterns (`_patterns.parquet`).
- Với mỗi split (train/valid/test) tạo một file Parquet duy nhất:
  - `icdv5_train_full.parquet`.
  - `icdv5_valid_full.parquet`.
  - `icdv5_test_full.parquet`.

Schema cuối:
- `id`, `title`, `lead_paragraph`, `category`, `source`, `label` (0/1).
- `reason_agree_vi`, `reason_disagree_vi`.
- `score_init`, `score_agree`, `score_disagree`.
- `tag_shock`, `tag_lifestyle`, `tag_listicle`, `tag_analysis`, `tag_promo`, `tag_hardnews`.

---

## 2. Dataloader & Tokenization

**Vị trí:** `src/ICD/dataset_icdv5.py`.

### 2.1. Tokenizer

- Dùng chung PhoBERT tokenizer như ICDv3/ICDv4 (`vinai/phobert-base`).
- Các max length đề xuất (tất cả là hyper‑params, set trong `training/ICD/train_ICD_v5.py`):
  - `max_len_news = 256` – cho `(title [SEP] lead)`.
  - `max_len_reason = 128` – cho mỗi reasoning.

### 2.2. Tensor output per batch

Cho batch size = `B`.

1. **News pair (base)** – giống ICDv4:
   - `input_ids_news`: `(B, max_len_news)`.
   - `attention_mask_news`: `(B, max_len_news)`.

2. **Reasoning title-free** (chỉ reasoning):
   - `input_ids_reason_agree_tf`: `(B, max_len_reason)`.
   - `attention_mask_reason_agree_tf`: `(B, max_len_reason)`.
   - `input_ids_reason_disagree_tf`, `attention_mask_reason_disagree_tf`: tương tự.

3. **Reasoning title-aware** (title + reasoning):
   - `input_ids_reason_agree_ta`: `(B, max_len_news)` – encode `[CLS] title [SEP] reason_agree [SEP]`.
   - `attention_mask_reason_agree_ta`: `(B, max_len_news)`.
   - `input_ids_reason_disagree_ta`, `attention_mask_reason_disagree_ta`: tương tự.

4. **Metadata cho experts & router**:
   - `category_id`: `(B,)` – integer; map từ category→id qua một `nn.Embedding` trong model.
   - `source_id`: `(B,)` – tương tự.
   - `pattern_tags`: `(B, K)` – K=6; từ cột `tag_*` cast sang float32.
   - `label`: `(B,)` – 0/1 float32.
   - `soft_label_llm`: `(B,)` – từ `score_init/agree/disagree` chuẩn hóa; tính ở dataloader.

Dataloader đảm bảo mọi tensor được đưa lên GPU bằng `.to(device)` trong training script.

---

## 3. ICDv5 Model – Chi tiết kiến trúc & shapes

**File:** `src/ICD/ICD_Model_v5.py`.

### 3.1. Tổng quan

ICDv5 gồm:
- **PhoBERT backbone shared** (12 layers, output hidden states).
- **BaseEncoder**: giống ICDv3/4 → `z_base` (news pair + ESIM + aux).
- **4–5 Pattern Experts**: mỗi expert là adapter nhỏ + pooling specialist → `z_e_k`.
- **Router**: module nhận `z_base` + embedding category/source + pattern logits, xuất `w ∈ R^K` (softmax) – trọng số cho experts.
- **Classifier**: nhận `z_mix = concat(z_base, z_experts_weighted)` → logit.
- **Projection heads** cho contrastive (nếu cần).

Tất cả tham số (PhoBERT, adapters, router, classifier, embeddings) đều `requires_grad=True` trừ khi freezing một số layer backbone bằng hàm riêng.

### 3.2. Kích thước chuẩn

- PhoBERT hidden size: `H = 768`.
- Số experts: `K = 5` (đề xuất: `style`, `shock`, `gap`, `promo`, `analysis`; `hardnews` có thể gộp vào `gap`/`analysis`).
- BaseEncoder output: `D_base = 5*H + D_aux` (như ICDv4: `[CLS, title, lead, |diff|, prod, aux]`). Với `D_aux = 6` ⇒ `D_base = 3846`.[`ICD_Model_v4.py`]
- Mỗi expert output: `D_exp = 512` (chọn số cố định, giảm dimension qua linear).
- Vector experts gộp: `z_exp = concat(z_e_1, ..., z_e_K)` có size `K * D_exp = 2560`.
- Router hidden size: `D_router = 512`.
- Classifier input: `D_cls = D_base + D_exp = 3846 + 2560 = 6406`.

Tất cả con số này viết thành constant trong class để tránh nhầm dimension.

### 3.3. Modules – chi tiết

#### 3.3.1. PhoBERT backbone & pooling reuse

Tái sử dụng các class từ ICDv4:
- `SegmentAwarePool`.
- `WeightedLayerPool`.

```python
class ICDv5(nn.Module):
    H = 768
    NUM_AUX = 6
    D_BASE = 5 * H + NUM_AUX  # 3846
    D_EXP = 512
    K_EXPERT = 5
    D_EXP_TOTAL = D_EXP * K_EXPERT  # 2560
    D_ROUTER = 512
    D_CLS = D_BASE + D_EXP_TOTAL  # 6406
```

Trong `__init__`:
- `self.config = AutoConfig.from_pretrained(..., output_hidden_states=True)`.
- `self.phobert = AutoModel.from_pretrained(..., config=self.config)`.
- `self.segment_pool`, `self.weighted_layer_pool` như v4.

#### 3.3.2. BaseEncoder

Hàm `encode_news` giống ICDv4, input:
- `input_ids_news: (B, L_n)`.
- `attention_mask_news: (B, L_n)`.
- `aux_features: (B, NUM_AUX)` – có thể là zeros hoặc handcrafted numeric features nếu sau này cần (hiện tại có thể để `NUM_AUX=0` cho đơn giản nhưng giữ interface).

Steps:
1. Run PhoBERT → `last_hidden: (B, L_n, H)`, `hidden_states list`.
2. Weighted pooling last 4 layers → `weighted: (B, L_n, H)`.
3. `cls = weighted[:,0,:]`.
4. Use `SegmentAwarePool` trên `last_hidden` để lấy `title_pool`, `lead_pool`.
5. `diff = |title_pool - lead_pool|`, `prod = title_pool * lead_pool`.
6. `z_base = concat(cls, title_pool, lead_pool, diff, prod, aux_features) → (B, D_BASE)`.
7. Apply dropout.

Đảm bảo thứ tự concat giống ICDv3/ICDv4 để sau này có thể reuse checkpoint bằng partial‑load nếu muốn.

#### 3.3.3. Expert modules

Thiết kế chung một class `ExpertModule`, dùng PhoBERT chung nhưng adapter riêng:

```python
class ExpertModule(nn.Module):
    def __init__(self, hidden_size: int, d_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, d_out),
        )
        self.attn_pool = AttentionPool(hidden_size)

    def forward(self, hidden_states, attention_mask):
        pooled = self.attn_pool(hidden_states, attention_mask)  # (B, H)
        return self.proj(pooled)  # (B, d_out)
```

Trong ICDv5:
- `self.expert_style = ExpertModule(H, D_EXP)` – input title‑only tokens.
- `self.expert_shock = ExpertModule(H, D_EXP)` – input news pair hoặc news+shock reasoning.
- `self.expert_gap = ExpertModule(H, D_EXP)` – có thể dùng lại ESIM kết quả `title_pool, lead_pool`, nhưng để đơn giản: feed news hidden_states (`last_hidden`).
- `self.expert_promo = ExpertModule(H, D_EXP)` – dùng news pair, trọng tâm category/source embed.
- `self.expert_anal = ExpertModule(H, D_EXP)` – dùng news pair, chú ý long‑form.

Input cho mỗi expert sẽ do `encode_experts` xây dựng từ `last_hidden_news` + các tokenization variant (title‑only,…). Điều quan trọng: mọi weight trong `ExpertModule` là trainable; không có rule thủ công.

#### 3.3.4. Router

Router dùng `z_base` + embedding metadata + pattern tags.

```python
class Router(nn.Module):
    def __init__(self, d_base, num_expert, d_router, num_cat, num_src, num_pattern):
        super().__init__()
        self.cat_emb = nn.Embedding(num_cat, 32)
        self.src_emb = nn.Embedding(num_src, 32)
        self.pattern_proj = nn.Linear(num_pattern, 32)
        self.fc = nn.Sequential(
            nn.Linear(d_base + 32 + 32 + 32, d_router),
            nn.GELU(),
            nn.LayerNorm(d_router),
            nn.Linear(d_router, num_expert),
        )

    def forward(self, z_base, category_id, source_id, pattern_tags):
        e_cat = self.cat_emb(category_id)      # (B, 32)
        e_src = self.src_emb(source_id)        # (B, 32)
        e_pat = self.pattern_proj(pattern_tags)  # (B, 32)
        z = torch.cat([z_base, e_cat, e_src, e_pat], dim=-1)  # (B, d_base+96)
        logits = self.fc(z)                    # (B, K)
        w = torch.softmax(logits, dim=-1)      # (B, K)
        return w, logits
```

Router outputs:
- `w`: trọng số softmax cho từng expert (trainable, differentiable).
- `logits`: dùng trong loss router (KL với distribution target từ pattern tags).

#### 3.3.5. Classifier head

```python
self.classifier = nn.Sequential(
    nn.LayerNorm(self.D_CLS),
    nn.Linear(self.D_CLS, 1024),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 256),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1),
)
```

Trong `forward` ICDv5:
1. Encode news → `z_base` + `last_hidden_news`.
2. Encode experts:
   - `z_style = expert_style(hidden_title_only, mask_title_only)`.
   - `z_shock = expert_shock(last_hidden_news, mask_news)`.
   - `z_gap = expert_gap(last_hidden_news, mask_news)`.
   - `z_promo = expert_promo(last_hidden_news, mask_news)`.
   - `z_anal = expert_anal(last_hidden_news, mask_news)`.
3. Stack experts: `Z = torch.stack([z_style, z_shock, z_gap, z_promo, z_anal], dim=1)` → shape `(B, K, D_EXP)`.
4. Router: `w, router_logits = router(z_base, category_id, source_id, pattern_tags)` → `(B, K)`.
5. Weighted sum experts: `z_mix_exp = (w.unsqueeze(-1) * Z).sum(dim=1)` → `(B, D_EXP)`.
6. Concat: `z_concat = torch.cat([z_base, z_mix_exp.repeat(1, self.K_EXPERT)], dim=-1)` **hoặc** đơn giản hơn: concat `z_base` + flatten `Z` weighted (để khớp `D_CLS=6406`). Ở đây nên cố định:
   - `z_flat = Z.view(B, -1)` → `(B, K*D_EXP)`.
   - `z_concat = torch.cat([z_base, z_flat], dim=-1)`.
7. Logit: `logits = classifier(z_concat)`.

Nhờ cách này dimension luôn cố định, không phụ thuộc K trong runtime.

---

## 4. Loss Functions & Training Objectives

**File:** `src/ICD/losses_v5.py` (mới), import dùng trong `training/ICD/train_ICD_v5.py`.

### 4.1. Main clickbait loss

- Binary focal loss + label smoothing:
  - Tham số (hyper‑param, trainable only via learning, không tham số cứng): `gamma`, `alpha` set trong config.
- Input: `logits` từ classifier, labels `y ∈ {0,1}`.

### 4.2. Soft label distillation từ LLM

- `soft_label_llm ∈ [0,1]` từ scores.
- Loss: `KLDivLoss` giữa `sigmoid(logits)` và `soft_label_llm` (với temperature nếu cần).

### 4.3. Router supervision

- Từ pattern tags `pattern_tags (B,K)` tạo distribution target `w_target`:
  - Normalize theo hàng: nếu sample có nhiều tag, chia đều; nếu không tag nào, dùng uniform.
- Loss: `KLDivLoss(log_softmax(router_logits), w_target)`.
- Thêm entropy regularizer: `λ * mean(sum(w * log w))` để khuyến khích router sparse.

### 4.4. Contrastive (optional, nếu muốn giữ từ ICDv4)

- Dùng projection heads `proj_news`, `proj_reason` giống ICDv4.[`ICD_Model_v4.py`]
- Loss InfoNCE giữa news representation và reasoning agree/disagree.

### 4.5. Distillation từ ICDv4 (không bắt buộc ngay)

- Nếu có logit/ prob của ICDv4 trên train set (có thể tạo offline): `p_v4(x)`.
- Loss: `KL(sigmoid(logits_v5) || p_v4)` cho các sample mà ICDv4 predict đúng (để bảo toàn kiến thức).

### 4.6. Tổng loss

Trong training loop:

```python
L_main = focal_with_label_smoothing(logits, y)
L_kl_llm = kl_with_soft_label(logits, soft_label_llm)
L_router = kl_router(router_logits, w_target) + entropy_reg(w)
L_contr = contrastive_loss(z_news_proj, z_reason_proj)
L_distill = kl_distill(logits, prob_v4)  # optional

loss = L_main + λ_llm * L_kl_llm + λ_router * L_router + λ_contr * L_contr + λ_distill * L_distill
```

Tất cả λ là hyper‑params trong `training/ICD/train_ICD_v5.py` (không hard-coded trong model).

---

## 5. Training Pipeline

**Script:** `training/ICD/train_ICD_v5.py`.

### 5.1. Cấu hình

- Dùng `argparse` hoặc `yaml` config, tương tự `train_ICD_v4.py`:
  - `--max_len_news`, `--max_len_reason`, `--batch_size`, `--lr`, `--lr_decay`, `--epochs`, `--freeze_layers`, `--alpha_focal`, `--gamma_focal`, `--lambda_kl`, `--lambda_router`, `--lambda_contr`, `--warmup_ratio`, `--patience`, v.v.
- Tất cả là tham số trainable/optimizible, **không có lexicon hay rule cứng**.

### 5.2. Optimizer & scheduler

- Sử dụng `AdamW`.
- Parameter groups:
  - Embedding/encoder layers PhoBERT với layer-wise lr decay (reuse `get_parameter_groups` từ ICDv4).[`ICD_Model_v4.py`]
  - Non‑backbone (experts, router, classifier) với lr × 10.
- Scheduler: linear warmup + cosine decay hoặc exponential decay như v4.

### 5.3. Training phases

1. **Phase 1 – Pretrain experts + router (optional nhưng nên có)**
   - Freeze PhoBERT hoặc chỉ fine‑tune nhẹ.
   - Task phụ: predict `pattern_tags` từ router (classification), predict `pattern_tags` hoặc shock score từ từng expert.
   - Giúp mỗi expert học domain riêng trước khi join task clickbait.

2. **Phase 2 – Joint training ICDv5**
   - Unfreeze các layer backbone cần thiết (ví dụ freeze dưới 9 layer như `ICDv4_9FL`).
   - Train với full loss (main + KL LLM + router + contrastive + distill if any).
   - Early stopping theo F1 trên `validate.csv`.

3. Lưu checkpoint tốt nhất:
   - `result/results_icdv5/icdv5_best.pt`.
   - Lưu thêm `config.json`, tokenizer name, etc.

---

## 6. Evaluation & Comparison

**Scripts (training/ICD/):**
- `eval_only.py` – mở rộng để nhận `--model v3|v4|v5`.
- `compare_v3_v4_v5.py` – so sánh 3 phiên bản.

### 6.1. Eval ICDv5

- Input: `icdv5_test_full.parquet`.
- Load checkpoint ICDv5.
- Tính:
  - Accuracy, Precision, Recall, F1.
  - Confusion matrix.
  - Brier score & calibration curve.
  - F1 theo pattern subset (dùng `pattern_tags` trên test): SHOCK, LIFESTYLE, LISTICLE, ANALYSIS, PROMO, HARDNEWS.
- Lưu ra:
  - `result/results_icdv5/test_metrics_full.json`.
  - `result/results_icdv5/test_predictions_full.parquet` (id, label, prob, pred, pattern_tags, router_weights,…).

### 6.2. So sánh với ICDv3 & ICDv4

**Script:** `training/ICD/compare_v3_v4_v5.py`.

Đầu vào:
- `result/…/ICDv3/test_predictions_full.parquet`.
- `result/…/ICDv4/test_predictions_full.parquet`.
- `result/results_icdv5/test_predictions_full.parquet`.

Các bước:
1. Join theo `id` và nhãn ground truth.
2. Tính metrics cho từng model.
3. McNemar test giữa (v3,v5) và (v4,v5).
4. Confusion matrix & calibration plots cho cả 3 model.
5. Bảng F1 per pattern subset.
6. Báo cáo Markdown: `src/experience/comparison/comparison_icdv3_icdv4_icdv5.md`.

Report nên làm giống format bạn đã dùng cho v3 vs v4.

---

## 7. Kiểm soát lỗi dimension & overflow

Để tránh crash khi implement:

1. **Define mọi dimension là constant** trong class (`H`, `D_BASE`, `D_EXP`, `K_EXPERT`, `D_CLS`) và luôn dùng chúng khi tạo Linear/LayerNorm.
2. **Unit test shapes**:
   - Tạo file `test/test_icdv5_shapes.py`.
   - Sinh batch dummy với `B=2`, `L_n=256`, `L_r=128`.
   - Chạy full forward ICDv5, assert:
     - `z_base.shape == (B, D_BASE)`.
     - `Z_expert.shape == (B, K_EXPERT, D_EXP)`.
     - `router_weights.shape == (B, K_EXPERT)` (softmax sum ~1).
     - `logits.shape == (B,1)`.
3. **Use `float32` everywhere**, tránh mixed precision trừ khi đã test kỹ.
4. **Gradient clipping** trong training script: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` để tránh exploding gradients.
5. **Max length & memory**:
   - Giữ `max_len_news <= 256`, `max_len_reason <= 128` như ICDv4 để tránh OOM.
   - Batch size dựa trên GPU (đã có trong `hw_profile` của bạn).

---

## 8. Tóm tắt file & vị trí cần tạo

- `data/processed/cleaned/prepare_icdv5_data.py`
- `src/ICD/reasoning/generate_pattern_tags.py`
- `data/processed/icdv5/build_icdv5_dataset.py`
- `src/ICD/dataset_icdv5.py`
- `src/ICD/ICD_Model_v5.py`
- `src/ICD/losses_v5.py`
- `training/ICD/train_ICD_v5.py`
- `training/ICD/compare_v3_v4_v5.py`
- `test/test_icdv5_shapes.py`

Với bản thiết kế này, bạn có thể lần lượt tạo các file code tương ứng mà không phải quyết định thêm options mơ hồ: mọi tensor shape, module, luồng dữ liệu đã được cố định rõ ràng, và toàn bộ thông tin “pattern / reasoning / style” đều được học qua tham số trainable thay vì lexicon hoặc heuristic cứng.