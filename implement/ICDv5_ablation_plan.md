# ICDv5 – Ablation Testing Plan

Mục tiêu file này:
- Định nghĩa **quy trình ablation** sau khi đã train xong ICDv5 full.
- Chuẩn hóa cách run, log và so sánh 3 biến thể chính: `ICDv5_Full`, `ICDv5_NoRouterSup`, `ICDv5_NoRouter`.
- Đảm bảo tất cả các bước đều cụ thể, có đường dẫn script/ràng buộc shapes rõ ràng (không mơ hồ) và dễ implement trong cây thư mục hiện tại.

Giả định:
- Kiến trúc ICDv5 đã được implement theo `ICDv5_implementation_plan.md`.
- Dataset: `data/processed/icdv5/icdv5_{train,valid,test}_full.parquet`.
- Scripts train/eval ICDv5 đặt trong `training/ICD/`.

---

## 1. Các biến thể cần ablation

### 1.1. ICDv5_Full (mô hình chuẩn)

**Tên model:** `ICDv5_Full`  
**Config chính:**
- K = 6 experts (STYLE, SHOCK, GAP, PROMO, ANALYSIS, HARDNEWS).
- Router **có supervision** bằng pattern tags (KL + entropy regularizer).
- Loss tổng:
  - `L_main`: Focal + label smoothing (clickbait vs non-clickbait).
  - `L_kl_llm`: KL với soft label từ LLM.
  - `L_router`: KL(router_logits, w_target) + entropy(w).
  - `L_contr`: OFF trong phiên bản này.
  - `L_distill`: optional – nếu có prob từ ICDv4, có thể bật, nhưng nên ghi rõ trong config.

**Check:** Đây chính là model đã train xong và lưu checkpoint `icdv5_full_best.pt` trong `result/results_icdv5/`.

### 1.2. ICDv5_NoRouterSup (router không supervision)

**Tên model:** `ICDv5_NoRouterSup`  
**Ý nghĩa:** Kiểm tra xem router có tự học được pattern gating mà không cần pattern tags từ LLM hay không.

Thay đổi so với Full:
- Vẫn giữ **K = 6 experts** và module Router y hệt ICDv5.
- **Tắt hoàn toàn `L_router`**:
  - Không tính KL với `w_target`.
  - Không thêm entropy regularizer.
- Các loss còn lại (`L_main`, `L_kl_llm`, `L_distill`) giữ nguyên.

Implementation gợi ý:
- Trong `training/ICD/train_ICD_v5.py` thêm flag:
  - `--no_router_sup` (bool).
- Nếu flag bật: set `lambda_router = 0.0` trong loss.

Checkpoint output:
- `result/results_icdv5/icdv5_noroutersup_best.pt`.
- Metrics/predictions lưu vào `result/results_icdv5/ICDv5_NoRouterSup/test_*.json|parquet`.

### 1.3. ICDv5_NoRouter (không dùng router, experts chỉ concat)

**Tên model:** `ICDv5_NoRouter`  
**Ý nghĩa:** Kiểm tra lợi ích của bản thân cơ chế router (so với việc chỉ thêm nhiều expert và concat thẳng).

Thay đổi so với Full:
- **Bỏ module Router trong forward pass**:
  - Không gọi `router(z_base, category_id, source_id, pattern_tags)`.
  - Thay vào đó, lấy tensor experts `Z` shape `(B, K, D_EXP)` và flatten thẳng: `z_flat = Z.view(B, K * D_EXP)`.
  - `z_concat = torch.cat([z_base, z_flat], dim=-1)` (đúng kích thước `D_CLS`).
- Loss tổng: chỉ còn `L_main` + `L_kl_llm` (+ `L_distill` nếu dùng). Không có router loss.
- Pattern tags vẫn có thể exist trong dataset nhưng **không được dùng trong forward / loss**.

Implementation gợi ý:
- Trong `ICD_Model_v5.py`, thêm tham số khởi tạo `use_router: bool = True`.
- Nếu `use_router=False`:
  - Skips router module.
  - Không cần compute `router_weights`.
- Trong training script, tạo config riêng `--no_router`.

Checkpoint output:
- `result/results_icdv5/icdv5_norouter_best.pt`.
- Metrics/predictions tương ứng.

---

## 2. Quy trình chạy Ablation

### 2.1. Chuẩn bị chung

1. Đảm bảo đã có:
   - `icdv5_full_best.pt` (ICDv5_Full) sau khi train xong.
   - Dataset parquet đầy đủ (`icdv5_{train,valid,test}_full.parquet`).
2. Set seed cố định cho mọi run (vd `seed=42`) trong `train_ICD_v5.py` để giảm variance.
3. Giữ nguyên các hyper-parameters sau giữa 3 runs:
   - `batch_size`, `lr`, `lr_decay`, `freeze_layers`, `max_len_news`, `max_len_reason`, `epochs`, `patience`, `warmup_ratio`, `alpha_focal`, `gamma_focal`, `lambda_kl`, `lambda_distill` (nếu dùng).

### 2.2. Run 1 – ICDv5_NoRouterSup

Command gợi ý:
```bash
python training/ICD/train_ICD_v5.py \
  --model_name ICDv5 \
  --run_name "ICDv5_NoRouterSup" \
  --no_router_sup True \
  --lambda_router 0.0 \
  --output_dir "result/results_icdv5/ICDv5_NoRouterSup" \
  --train_path "data/processed/icdv5/icdv5_train_full.parquet" \
  --valid_path "data/processed/icdv5/icdv5_valid_full.parquet" \
  --test_path  "data/processed/icdv5/icdv5_test_full.parquet"
```

Output cần có:
- `best_checkpoint.pt`.
- `test_metrics_full.json` (Accuracy, P, R, F1, Brier,…).
- `test_predictions_full.parquet` (id, label, prob, pred, pattern_tags, router_weights).

### 2.3. Run 2 – ICDv5_NoRouter

Command gợi ý:
```bash
python training/ICD/train_ICD_v5.py \
  --model_name ICDv5 \
  --run_name "ICDv5_NoRouter" \
  --no_router True \
  --output_dir "result/results_icdv5/ICDv5_NoRouter" \
  --train_path "data/processed/icdv5/icdv5_train_full.parquet" \
  --valid_path "data/processed/icdv5/icdv5_valid_full.parquet" \
  --test_path  "data/processed/icdv5/icdv5_test_full.parquet"
```

Trong script:
- Nếu `--no_router=True`, khởi tạo model với `use_router=False` và bỏ mọi loss liên quan router.

Output tương tự Run 1.

### 2.4. Run 3 – ICDv5_Full (nếu cần re-run để so sánh chuẩn)

Nếu đã có checkpoint và metrics cho ICDv5_Full, có thể skip training lại. Nếu không, chạy:

```bash
python training/ICD/train_ICD_v5.py \
  --model_name ICDv5 \
  --run_name "ICDv5_Full" \
  --output_dir "result/results_icdv5/ICDv5_Full" \
  --train_path "data/processed/icdv5/icdv5_train_full.parquet" \
  --valid_path "data/processed/icdv5/icdv5_valid_full.parquet" \
  --test_path  "data/processed/icdv5/icdv5_test_full.parquet"
```

Đảm bảo `lambda_router > 0` và `use_router=True`.

---

## 3. Script phân tích và so sánh ablation

**File:** `training/ICD/ablation_icdv5.py`.

### 3.1. Input

- `result/results_icdv5/ICDv5_Full/test_metrics_full.json`.
- `result/results_icdv5/ICDv5_NoRouterSup/test_metrics_full.json`.
- `result/results_icdv5/ICDv5_NoRouter/test_metrics_full.json`.
- `result/results_icdv5/ICDv5_Full/test_predictions_full.parquet`.
- `result/results_icdv5/ICDv5_NoRouterSup/test_predictions_full.parquet`.
- `result/results_icdv5/ICDv5_NoRouter/test_predictions_full.parquet`.

### 3.2. Các bước phân tích

1. **Load metrics** và tạo bảng tổng:
   - Accuracy, Precision, Recall, F1, Brier cho 3 model.
2. **F1 theo pattern subset**:
   - Sử dụng cột `pattern_tags` trong predictions.
   - Với mỗi pattern (SHOCK, LIFESTYLE, LISTICLE, ANALYSIS, PROMO, HARDNEWS):
     - Lọc các sample có `tag_* = 1`.
     - Tính Precision, Recall, F1 cho từng model.
3. **Entropy router** (chỉ ICDv5_Full, ICDv5_NoRouterSup):
   - Từ predictions, lấy `router_weights` (shape `(N, K)`), compute entropy per sample.
   - Log `mean_entropy_overall` + `mean_entropy_per_pattern`.
   - Nếu entropy thấp hơn ở Full so với NoRouterSup, chứng tỏ supervision hoàn toàn có tác dụng.
4. **McNemar Test**:
   - So sánh:
     - Full vs NoRouterSup.
     - Full vs NoRouter.
   - Lưu `b, c, chi2, p, significant` như bạn đã làm cho ICDv3 vs ICDv4.
5. **Xuất báo cáo Markdown**:
   - Lưu thành `src/experience/comparison/icdv5_ablation_report.md`.
   - Nội dung gồm:
     - Bảng metrics tổng.
     - Bảng F1 per pattern.
     - Kết quả McNemar.
     - Phân tích entropy router.
     - Kết luận text (1–2 đoạn): router có đóng góp gì, pattern tags có giúp gì, experts có thực sự hoạt động không.

### 3.3. Chạy script

```bash
python training/ICD/ablation_icdv5.py \
  --full_dir       "result/results_icdv5/ICDv5_Full" \
  --noroutersup_dir "result/results_icdv5/ICDv5_NoRouterSup" \
  --norouter_dir    "result/results_icdv5/ICDv5_NoRouter"
```

---

## 4. Tiêu chí đánh giá kết quả ablation

Khi đọc `icdv5_ablation_report.md`, tập trung vào các câu hỏi sau:

1. **Router supervision có thực sự cần không?**
   - Nếu `F1(Full) > F1(NoRouterSup)` và đặc biệt `F1 per pattern` (SHOCK, LIFESTYLE, LISTICLE,…) tăng đáng kể, đồng thời entropy(router) giảm ⇒ pattern tags từ LLM thực sự giúp router học gating.

2. **Router có đáng để giữ không?**
   - Nếu `F1(Full) ≫ F1(NoRouter)` và `F1(NoRouterSup)` chỉ ở mức ngang NoRouter ⇒ router + supervision đóng góp chính.
   - Nếu `F1(Full) ≈ F1(NoRouter)` ⇒ router không mang lại gì, kiến trúc MoE hiện tại cần điều chỉnh.

3. **Pattern nào hưởng lợi nhiều nhất từ ICDv5?**
   - So sánh F1 per pattern giữa 3 model để xác định nhóm SHOCK/LIFESTYLE/LISTICLE/ANALYSIS/PROMO/HARDNEWS nào được cải thiện.

4. **Có expert nào "chết" không?**
   - Nếu với pattern X, router luôn cho weight gần 0 cho expert tương ứng ⇒ kiến trúc hoặc supervision của expert đó có vấn đề (cần debug riêng).

Bám vào các tiêu chí trên, bạn có thể quyết định có cần ICDv6 hay không, và nếu có thì nên sửa vào đâu (router, experts hay loss).