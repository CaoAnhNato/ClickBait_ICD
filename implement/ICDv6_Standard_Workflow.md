# Quy trình chuẩn huấn luyện và Ablation Study cho ICDv6

Dưới đây là quy trình tối ưu để đạt được kết quả tốt nhất (SOTA) cho mô hình ICDv6, kết hợp giữa việc tối ưu hóa tự động và thực nghiệm từng giai đoạn.

## Bước 1: Chuẩn bị dữ liệu sạch (Data Preparation)
Đảm bảo dữ liệu đã được tiền xử lý với VnCoreNLP và tích hợp các pattern tags từ các phiên bản trước.
```bash
conda run -n MLE python data/processed/icdv6/build_icdv6_dataset.py
```
> [!NOTE]
> Bước này sẽ tạo ra các file `.parquet` trong `data/processed/icdv6/` giúp tăng tốc độ load dữ liệu gấp 5-10 lần so với CSV.

## Bước 2: Tìm kiến trúc Baseline tốt nhất (C0 Selection)
Chạy thử nghiệm cả hai biến thể để chọn ra "xương sống" (Backbone) cho các giai đoạn sau.
*   **Simple**: Chỉ dùng token `[CLS]`.
*   **ESIM**: Dùng tương tác Title/Lead.

```bash
# Thử nghiệm Simple
python training/ICD/train_ICD_v6.py --variant simple --num_epochs 10 --run_name "C0_Simple_Initial"

# Thử nghiệm ESIM
python training/ICD/train_ICD_v6.py --variant esim --num_epochs 10 --run_name "C0_ESIM_Initial"
```
*So sánh F1-score trên console hoặc MLflow để chọn Variant tốt hơn.*

## Bước 3: Tối ưu hóa tham số tự động (Hyperparameter Tuning)
Sử dụng Optuna để tìm bộ trọng số lý tưởng cho Phase 1 và Phase 2. Đây là bước quan trọng nhất để vượt qua Baseline.
```bash
python training/ICD/tune_ICD_v6.py --n_trials 30 --num_epochs 5 --freeze_layers 8
```
**Mục tiêu**: Lấy được bộ thông số `w_hardnews`, `w_shock`, `lr`, và `rdrop_alpha` tối ưu từ kết quả in ra ở console.

## Bước 4: Huấn luyện chính thức theo giai đoạn (Final Training)
Sử dụng bộ tham số vừa tìm được ở Bước 3 để huấn luyện mô hình đầy đủ (Full training).
```bash
python training/ICD/train_ICD_v6.py \
  --variant [variant_tu_buoc_2] \
  --num_epochs 20 \
  --use_reweighting --w_hardnews [best_w1] --w_shock [best_w2] \
  --use_pattern_residual \
  --lr [best_lr] --rdrop_alpha [best_rdrop] \
  --run_name "ICDv6_Final_Model"
```

## Bước 5: Chạy Ablation Study tổng thể (Verification)
Sử dụng script tự động để tạo bảng so sánh cho báo cáo/đồ án.
```bash
chmod +x training/ICD/run_full_ablation_v6.sh
./training/ICD/run_full_ablation_v6.sh
```

## Bước 6: Phân tích và Kết luận
1.  **MLflow/WandB**: Truy cập để xem biểu đồ hội tụ.
2.  **Report**: Kiểm tra file `src/experience/icdv6/report_{variant}.txt` để xem Precision/Recall của từng lớp.
3.  **Error Analysis**: Nếu kết quả chưa tốt ở lớp Hardnews, hãy tăng `--w_hardnews` và chạy lại.

---
**Lời khuyên từ Antigravity**: Luôn ưu tiên dùng `--freeze_layers 8` (hoặc 6-10) trong quá trình Tuning để tiết kiệm thời gian, sau đó có thể Unfreeze (set về 0) ở Phase huấn luyện cuối cùng để đạt độ chính xác tối đa.
