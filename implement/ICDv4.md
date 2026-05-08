# ICDv4 – Implement khung LLM‑assisted reasoning cho clickbait (không code)

Tài liệu này mô tả **cụ thể, tuần tự** cách implement và vận hành ICDv4 trong codebase hiện tại, để so sánh với baseline ICDv3 (ICD_Model_v3.1). Không liệt kê code chi tiết; mọi bước chỉ mô tả kiến trúc, dữ liệu, pipeline và các thí nghiệm bắt buộc.

Mục tiêu:
- Thay thế ý tưởng lexicon "shock" cảm tính bằng **khung LLM‑assisted opposing‑stance reasoning** giống các phương pháp LFND‑AB (fake news) và SORG/ORCD (clickbait).
- Thiết kế một model local ICDv4 tận dụng reasoning sinh bởi LLM (agree/disagree + soft score) để cải thiện phát hiện clickbait ở các pattern khó trong `label_issues.json` (crime/sex/moral, curiosity gap, advertorial nuance…).
- Quy trình đánh giá chuẩn: sinh data, train ICDv3 baseline, train ICDv4, so sánh metrics, sau đó chạy ablation.

---

## 1. Chuẩn bị dữ liệu gốc và cấu trúc thư mục

1. Sử dụng dataset đã được làm sạch ở đường dẫn:
   - `data/processed/cleaned/Cleaned_Clickbait_Dataset.csv`

2. Yêu cầu tối thiểu các cột trong CSV:
   - `id` hoặc `index`: định danh duy nhất cho mỗi sample (nếu chưa có, cần tạo một cột `id` tăng dần).
   - `title`: tiêu đề tin.
   - `lead`: đoạn mô tả/lead.
   - `label`: nhãn nhị phân (1 = clickbait, 0 = non‑clickbait).

3. Chuẩn hoá split:
   - Dùng lại **y hệt** split train/validate/test mà ICDv3 đang sử dụng là 'data/processed/cleaned/test_clean.csv', 'data/processed/cleaned/train_clean.csv', 'data/processed/cleaned/validate_clean.csv'.

4. Tạo cấu trúc thư mục mới cho ICDv4:
   - `src/data/processed/icdv4/` – chứa các file reasoning và dataset sau khi merge.
   - `src/experience/icdv3/` – log, checkpoint của baseline ICDv3.
   - `src/experience/icdv4/` – log, checkpoint của ICDv4 và các ablation.

---

## 2. Pha 1 – Sinh reasoning đối nghịch bằng LLM (offline)

Pha này chạy **offline, một lần** trên toàn bộ dataset (train+validate+test). Không dùng nhãn thật trong prompt. Mọi output được lưu lại để dùng nhiều lần.
### 2.1. Thiết kế output reasoning cho mỗi sample

Với mỗi dòng trong `Cleaned_Clickbait_Dataset.csv` (xác định bởi `id`, `title`, `lead`):

Sinh và lưu các thông tin sau vào một bản ghi JSON:

- `id`: giống với CSV gốc.
- `initial_score` (`V_I`): điểm clickbait ban đầu trên thang 0–100 do LLM đánh giá cho `(title, lead)`.
- `initial_reason` (`R_I`): đoạn giải thích ngắn vì sao cho điểm `V_I`.
- `agree_reason` (`R_A`): đoạn reasoning khi **giả sử headline là clickbait**.
- `agree_score` (`V_A`): điểm clickbait mới sau khi xem xét `R_A` (phải **cao hơn** `V_I` và đáng kể hơn 50).
- `disagree_reason` (`R_D`): đoạn reasoning khi **giả sử headline là non‑clickbait**.
- `disagree_score` (`V_D`): điểm clickbait mới sau khi xem xét `R_D` (phải **thấp hơn** `V_I` và đáng kể thấp hơn 50).

Tất cả được lưu thành một file JSONL:
- `data/processed/icdv4/reasoning_all.jsonl` – mỗi dòng là một object JSON như trên, key là `id` để merge lại với CSV gốc.

### 2.2. Bước A – Rating ban đầu (initial title rating)

1. Cho mỗi `(id, title, lead)`:
   - Tạo prompt yêu cầu LLM:
     - Đọc `title` + `lead` tiếng Việt.
     - Đánh giá **mức độ clickbait của headline** trên thang 0–100.
     - In ra 2 phần: điểm số `V_I` và đoạn giải thích `R_I`.

2. Áp dụng cơ chế giống Algorithm 1 (SORG):
   - Đặt tham số `α` (ví dụ `α = 30`).
   - Nếu `V_I` nằm ngoài khoảng `[α, 100−α]` (quá gần 0 hoặc 100), yêu cầu LLM **re‑rating** bằng prompt giải thích vì sao cần điều chỉnh (tăng hoặc giảm) và cho điểm mới.
   - Lặp tối đa một số lần cố định (ví dụ 3–5 lần) cho đến khi `V_I` nằm trong `[α, 100−α]` hoặc hết số lần.

3. Lưu `V_I` và `R_I` vào record reasoning cho sample đó.

### 2.3. Bước B – Sinh reasoning Agree/Disagree (Self‑renewal opposing‑stance)

1. Với mỗi sample đã có `V_I, R_I`:
   - Sinh **reasoning Agree** (`R_A, V_A`):
     - Prompt LLM: “Giả sử headline này được xem là **clickbait**, hãy viết một đoạn lý do giải thích vì sao, và sau đó đánh lại mức độ clickbait trên thang 0–100.”
   - Sinh **reasoning Disagree** (`R_D, V_D`):
     - Prompt LLM: “Giả sử headline này được xem là **không phải clickbait**, hãy viết một đoạn lý do giải thích vì sao, và sau đó đánh lại mức độ clickbait trên thang 0–100.”

2. Ràng buộc giống Algorithm 2 (SORG):
   - Với `Agree`:
     - `V_A ≥ 50 + γ` và `V_A − V_I ≥ β` (ví dụ `γ = 5`, `β = 10`).
     - Nếu không đạt, yêu cầu LLM tự phê bình reasoning cũ (giải thích vì sao chưa đủ thuyết phục) rồi sinh lại reasoning + score mới, tối đa M lần.
   - Với `Disagree`:
     - `V_D ≤ 50 − γ` và `V_I − V_D ≥ β`.
     - Nếu không đạt, làm tương tự.

3. Kết quả cuối cùng sau vòng lặp:
   - Một cặp reasoning đối nghịch (`R_A, V_A`) và (`R_D, V_D`) thỏa tiêu chí phân cực.
   - Append vào record tương ứng trong `reasoning_all.jsonl`.

### 2.4. Kiểm tra chất lượng và hoàn tất reasoning dataset

1. Sau khi chạy xong toàn bộ dataset:
   - Kiểm tra tỷ lệ sample có reasoning hợp lệ (`V_A` và `V_D` đạt điều kiện). Nếu có lỗi, log và giải quyết (ví dụ đánh dấu `reasoning_status = failed` cho 1 số ít sample, sẽ xử lý đặc biệt trong dataset).

2. Output cuối pha 1:
   - `data/processed/icdv4/reasoning_all.jsonl` – file reasoning đầy đủ cho mọi mẫu.

---

## 3. Pha 2 – Xây dataset ICDv4 từ CSV + reasoning

Pha này merge CSV gốc với reasoning, xây thêm các trường cần thiết cho training ICDv4.

### 3.1. Merge dữ liệu

1. Đọc `Cleaned_Clickbait_Dataset.csv` và `reasoning_all.jsonl`, join theo `id`.
2. Bỏ các sample không có reasoning hợp lệ (nếu có), hoặc gắn cờ riêng nếu cần xử lý đặc biệt.

### 3.2. Chuẩn hóa soft label từ LLM

Tạo các trường mới:

1. `p_llm_initial` – xác suất clickbait từ `V_I`:
   - Chuẩn hóa: `p_llm_initial = V_I / 100` (giá trị [0,1]).
2. `p_llm_agree` – từ `V_A / 100`.
3. `p_llm_disagree` – từ `V_D / 100`.
4. `p_llm_final` – soft label dùng để distillation:
   - Nếu `label = 1`: `p_llm_final = (1 − λ) * 1.0 + λ * p_llm_initial` (λ nhỏ, ví dụ 0.2).
   - Nếu `label = 0`: `p_llm_final = (1 − λ) * 0.0 + λ * p_llm_initial`.

Lưu ý: chỉ dùng `V_I` để tạo soft label, `V_A, V_D` chủ yếu dùng cho contrastive learning.

### 3.3. Lưu dataset đã mở rộng

1. Tạo file mới:
   - `data/processed/icdv4/Cleaned_Clickbait_with_reasoning.parquet`
   - Mỗi record gồm:
     - Tất cả cột gốc (`id, title, lead, label,...`).
     - `initial_score, initial_reason, agree_reason, agree_score, disagree_reason, disagree_score`.
     - `p_llm_initial, p_llm_agree, p_llm_disagree, p_llm_final`.

2. Đảm bảo mọi split train/validate/test đều có đầy đủ trường này.

---

## 4. Pha 3 – Thiết kế kiến trúc ICDv4 (concept, không code)

ICDv4 là một model local, fine‑tune trên PhoBERT, mở rộng ICDv3.1 theo khung ORCD/SORG:[file:2][file:6]

### 4.1. Input và encoder

1. **Input chính** (như ICDv3.1):
   - `(title, lead)` được tokenizer PhoBERT thành sentence pair.
   - `label` nhị phân.
   - `p_llm_final` (soft label).

2. **Input reasoning**:
   - `agree_reason` (`R_A`).
   - `disagree_reason` (`R_D`).

3. **Encoders**:
   - **Encoder_T** (news encoder): PhoBERT + SegmentAwarePool + WeightedLayerPool như ICDv3.1, tạo vector `h_T` (CLS + title/lead pooled + ESIM features).
   - **Encoder_A_TF** (title‑free agree reasoning): encoder riêng cho `R_A`, trả về `h_A_tf`.
   - **Encoder_D_TF** (title‑free disagree reasoning): encoder riêng cho `R_D`, trả về `h_D_tf`.
   - **Encoder_A_TA** (title‑aware agree reasoning): encode concat `(title, agree_reason)` dưới dạng sentence pair, trả về `h_A_ta`.
   - **Encoder_D_TA** (title‑aware disagree reasoning): encode `(title, disagree_reason)`, trả về `h_D_ta`.

Các encoder có thể share backbone PhoBERT một phần hoặc toàn bộ, nhưng ICDv4 chuẩn sẽ được implement với **một backbone PhoBERT dùng chung**, các head pooling khác nhau cho news vs reasoning để tiết kiệm tham số.

### 4.2. Fused representation và head phân loại

1. Xây các representation trung gian:
   - `z_T` – vector news (như ICDv3.1): concat `[CLS, title_pool, lead_pool, |title−lead|, title*lead]`.
   - `z_A` – fused agree: concat `h_A_tf` và `h_A_ta`.
   - `z_D` – fused disagree: concat `h_D_tf` và `h_D_ta`.

2. Xây vector tổng hợp cuối cùng để phân loại:
   - `z_all = [z_T ; z_A ; z_D]` (concat theo chiều feature).

3. Classifier head ICDv4:
   - LayerNorm trên `z_all`.
   - MLP nhiều tầng (ẩn 768, 256, 1) với GELU + Dropout, tương tự ICDv3.1 nhưng input_dim lớn hơn.
   - Output: logit clickbait cho mỗi sample.

### 4.3. Loss function tổng

ICDv4 sử dụng tổ hợp loss sau (trong mọi run chuẩn):

1. **Loss phân loại chính với soft label**:
   - Dùng logit ra xác suất `p_model` qua sigmoid.
   - Tính **Binary Cross Entropy** với nhãn thật `label` (kèm label smoothing như ICDv3.1).
   - Thêm **KL divergence** giữa `p_model` và `p_llm_final` (soft label từ LLM).
   - Loss này giúp model vừa tôn trọng label thật, vừa tận dụng phân phối mềm từ LLM giống cách dùng rating trong ORCD/LFND‑AB.

2. **Loss contrastive title–reason**:
   - Định nghĩa các pair:
     - Positive: `(z_T, z_A)` nếu `agree_score > disagree_score`.
     - Negative: `(z_T, z_D)` và reasoning của sample khác.
   - Áp dụng InfoNCE hoặc contrastive loss tương tự để kéo `z_T` gần `z_A` và xa `z_D`.
   - Loss này buộc model học representation nơi reasoning "clickbait" align tốt với headline clickbait, giống module contrastive trong hai paper.

3. **R‑Drop / consistency loss**:
   - Duy trì R‑Drop như ICDv3.1: chạy hai forward pass với dropout, tính KL giữa hai distribution output.

4. Loss tổng:
   - `L_total = L_class + α * L_contrastive + β * L_RDrop` với α, β được cố định trong config ICDv4.

### 4.4. Khác biệt chính ICDv4 so với ICDv3

1. Thêm 4 encoder reasoning (title‑free + title‑aware, agree + disagree) và fused vector `z_A, z_D`.
2. Thêm loss contrastive giữa news và reasoning.
3. Thêm soft label distillation từ `p_llm_final`.
4. Giữ nguyên backbone PhoBERT + SegmentAwarePooling + ESIM từ ICDv3.1 để đảm bảo cải tiến đến từ reasoning.

---

## 5. Pha 4 – Train baseline ICDv3 trên dataset cleaned

Trước khi train ICDv4, cần có kết quả baseline ICDv3 trên cùng `Cleaned_Clickbait_Dataset.csv` và cùng split.

### 5.1. Chuẩn bị

1. Sử dụng đúng code ICDv3.1 hiện tại (`ClickbaitDetectorV3_1`) cùng pipeline training đã ổn định.
2. Dữ liệu input:
   - Đọc `Cleaned_Clickbait_Dataset.csv` và splits từ `data/processed/cleaned/test_clean.csv`, `data/processed/cleaned/train_clean.csv`, `data/processed/cleaned/validate_clean.csv`.
   - Không dùng bất kỳ field reasoning hay soft label nào.

### 5.2. Cấu hình train baseline

1. Hyperparameters:
   - Learning rate, batch size, số epoch, scheduler, label smoothing, R‑Drop… giữ nguyên như config tốt nhất hiện tại.
2. Log & checkpoint:
   - Lưu checkpoint tốt nhất (theo F1 trên dev) vào `experiments/icdv3/checkpoints/icdv3_best.pt`.
   - Lưu log training và metrics dev.

### 5.3. Đánh giá baseline

1. Chạy inference trên test set ViClickbait:
   - Lưu `logits`, `probs`, `pred_labels` cho từng `id`.
   - Tính F1, Precision, Recall tổng.
   - Tính F1, Precision, Recall theo `pattern_id` (P1, P2, P3…).
   - Tính Brier score và vẽ calibration curve (lưu file ảnh và số liệu).

2. Lưu kết quả vào:
   - `src/experience/icdv3/results/test_metrics.json` (global + per pattern + calibration stats).
   - `src/experience/icdv3/results/test_predictions.parquet` (id, label, prob, pred).

---

## 6. Pha 5 – Train ICDv4 trên dataset với reasoning

Pha này sử dụng dataset đã merge reasoning để train ICDv4.

### 6.1. Chuẩn bị dataloader ICDv4

1. Sử dụng file:
   - `data/processed/icdv4/Cleaned_Clickbait_with_reasoning.parquet`.
2. Với mỗi sample, dataloader trả về:
   - `input_ids, attention_mask` cho `(title, lead)`.
   - `input_ids_agree, attention_mask_agree` cho `agree_reason`.
   - `input_ids_disagree, attention_mask_disagree` cho `disagree_reason`.
   - `label` (0/1).
   - `p_llm_final` (float).

### 6.2. Cấu hình train ICDv4

1. Sử dụng cấu hình nền giống ICDv3.1:
   - PhoBERT backbone, optimizer, scheduler, số epoch, batch size.
2. Khác biệt:
   - Enable các encoder reasoning và head contrastive.
   - Enable loss soft label distillation (KL với `p_llm_final`).
   - Enable loss contrastive (với hệ số α cố định).
   - Giữ R‑Drop.

3. Checkpoint & log:
   - Lưu checkpoint tốt nhất (theo F1 dev) vào `src/experience/icdv4/checkpoints/icdv4_full.pt`.
   - Lưu log training và metrics trên dev.

### 6.3. Đánh giá ICDv4 trên test

1. Chạy inference ICDv4 trên test set (cùng `id` và split với ICDv3):
   - Lưu `logits`, `probs`, `pred_labels` cho từng `id`.
2. Tính metrics tương tự baseline:
   - F1, Precision, Recall tổng.
   - F1, Precision, Recall per `pattern_id`.
   - Brier score.
   - Calibration curve.

3. Lưu kết quả vào:
   - `src/experience/icdv4/results/test_metrics_full.json`.
   - `src/experience/icdv4/results/test_predictions_full.parquet`.

---

## 7. Pha 6 – So sánh ICDv4 và ICDv3

Sau khi có kết quả test của cả ICDv3 và ICDv4:

1. Tạo bảng so sánh toàn cục:
   - F1, Precision, Recall tổng của ICDv3 vs ICDv4.
   - Brier score của ICDv3 vs ICDv4.

2. So sánh per pattern:
   - Với mỗi `pattern_id` (P1, P2, P3…):
     - F1, Precision, Recall của ICDv3 vs ICDv4.
   - Đặc biệt highlight:
     - P1: crime/sex/moral shock – xem recall clickbait có tăng không (giảm FN kiểu trong `label_issues.json`).
     - P2: curiosity gap – xem model có nhận ra các headline “đổi xăng lấy điện”, “biển mây, thác nước…” tốt hơn không.
     - P3: promo/advertorial – model có phân biệt được nuance promo vs clickbait lố hay không.

3. So sánh calibration:
   - Đặt hai đường calibration curve (ICDv3 vs ICDv4) trên cùng đồ thị.
   - Đánh giá xem ICDv4 có gần đường y=x hơn không.

4. Ghi kết luận định tính:
   - ICDv4 được xem là cải tiến nếu:
     - F1 tổng ≥ ICDv3.
     - F1/Recall trên P1 và P2 không giảm, lý tưởng là tăng.
     - Brier score giảm, calibration tốt hơn.

---

## 8. Pha 7 – Chạy ablation bắt buộc cho ICDv4

Để chứng minh từng thành phần của ICDv4 đều đóng góp rõ ràng (không phải patch tuỳ hứng), cần thực hiện **3 ablation bắt buộc**, mỗi ablation là một model riêng với cùng kiến trúc base nhưng tắt một phần chức năng.

Các ablation dùng chung reasoning dataset và split; chỉ thay đổi cấu hình loss/encoder.

### 8.1. ICDv4‑no‑reasoning (chỉ soft label, không encoder reasoning)

Mục tiêu: kiểm tra tác dụng **encoder reasoning + contrastive**.

Thiết lập:
- Vẫn dùng dữ liệu `p_llm_final`.
- Bỏ toàn bộ input `agree_reason`, `disagree_reason` và các encoder tương ứng.
- Vector vào classifier chỉ là `z_T` (giống ICDv3.1).
- Loss:
  - Giữ CE + soft label (KL với `p_llm_final`).
  - Giữ R‑Drop.
  - **Tắt hoàn toàn loss contrastive**.

Train, evaluate như ICDv4 full, lưu kết quả vào `src/experience/icdv4/results/test_metrics_no_reasoning.json`.

### 8.2. ICDv4‑no‑soft (reasoning có, không distillation)

Mục tiêu: kiểm tra tác dụng **soft label distillation**.

Thiết lập:
- Giữ đầy đủ encoder reasoning (agree + disagree, title‑free + title‑aware).
- Input không sử dụng `p_llm_final` trong loss.
- Loss:
  - CE chuẩn với nhãn cứng (label smoothing như ICDv3.1).
  - **Không dùng KL với `p_llm_final`**.
  - Giữ loss contrastive và R‑Drop.

Train, evaluate như ICDv4 full, lưu kết quả vào `src/experience/icdv4/results/test_metrics_no_soft.json`.

### 8.3. ICDv4‑no‑contrastive (reasoning có, không contrastive)

Mục tiêu: kiểm tra tác dụng **contrastive learning title–reason**.

Thiết lập:
- Giữ đầy đủ encoder reasoning.
- Giữ soft label distillation với `p_llm_final`.
- Loss:
  - CE (nhãn cứng) + KL (soft label).
  - **Tắt loss contrastive** giữa `z_T, z_A, z_D`.
  - Giữ R‑Drop.

Train, evaluate như ICDv4 full, lưu kết quả vào `src/experience/icdv4/results/test_metrics_no_contrastive.json`.

### 8.4. Tổng hợp kết quả ablation

1. Tạo bảng so sánh các model: ICDv3, ICDv4‑full, ICDv4‑no‑reasoning, ICDv4‑no‑soft, ICDv4‑no‑contrastive.
2. So sánh trên các trục:
   - F1 tổng, F1 P1, F1 P2, Brier score.
3. Rút ra kết luận rõ ràng:
   - Việc thêm reasoning (so với ICDv4‑no‑reasoning) có tăng chất lượng không.
   - Việc thêm soft label (so với ICDv4‑no‑soft) có cải thiện calibration và/hoặc F1 không.
   - Việc dùng contrastive (so với ICDv4‑no‑contrastive) có tăng robustness per‑pattern không.

---

## 9. Tổng kết

Bản implement này định nghĩa **một khung ICDv4 duy nhất, không chứa optionals**, gồm:
- Pha sinh reasoning đối nghịch bằng LLM theo spirit SORG/ORCD (rating ban đầu + self‑renewal reasoning).
- Pha merge dataset và xây soft label từ LLM.
- Kiến trúc ICDv4 với news encoder ICDv3.1 + 4 reasoning encoders + fused head, kèm loss CE + soft label, contrastive và R‑Drop.
- Quy trình train/eval ICDv3 baseline, ICDv4 full, và 3 ablation bắt buộc.

Người implement chỉ cần đi lần lượt các pha trong tài liệu này (tạo script, class, config tương ứng) là có thể tái hiện đầy đủ khung lý thuyết ICDv4 và kiểm chứng định lượng so với ICDv3 trên ViClickbait.
