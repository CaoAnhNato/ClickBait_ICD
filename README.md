# ClickBait ICD (Intentional Clickbait Detection)

Đây là kho lưu trữ cho dự án phát hiện Clickbait tiếng Việt bằng mô hình Multi-Attention kết hợp (PhoBERT-base cho Content và Character-Level Transformer cho Style).

## 🚀 Hướng dẫn cài đặt và chạy (Getting Started)

Làm theo các bước dưới đây để thiết lập môi trường và chạy thử mô hình trên máy cục bộ hoặc máy ảo (VM/Server).

### 1. Clone repository
Khởi tạo và tải dự án về máy:
```bash
git clone https://github.com/CaoAnhNato/ClickBait_ICD.git
cd ClickBait_ICD
```

### 2. Cài đặt môi trường (Dependencies)
Dự án sử dụng file `requirements.txt` chuyên biệt cho môi trường ảo (đã bao gồm các thư viện xử lý NLP như `underthesea`, `pyvi`, v.v.).
*(Lưu ý: File yêu cầu này đã được loại bỏ thư viện `torch` để không làm hỏng cấu hình PyTorch/CUDA có sẵn trên máy ảo. Nếu máy bạn chưa có PyTorch, vui lòng cài đặt thủ công phiên bản phù hợp với GPU của máy)*

```bash
pip install -r requirements.txt
```

### 3. Hướng dẫn Training mô hình ICD
Kịch bản huấn luyện chính của mô hình nằm tại `training/ICD/train_ICD.py`. Script này hỗ trợ tự động cấu hình **Batch Size** và **Gradient Accumulation** phù hợp với dung lượng VRAM của GPU thông qua tham số `--hw_profile`.

**Dành cho máy cấu hình yếu (VRAM thấp - VD: RTX 3050 4GB):**
```bash
python training/ICD/train_ICD.py --hw_profile rtx3050 --epochs 30 --lr 2e-5
```
*(Cấu hình này tự động đặt `Batch Size = 4` và `Gradient Accumulation = 8`)*

**Dành cho máy cấu hình mạnh (VRAM cao - VD: RTX ADA 5000):**
```bash
python training/ICD/train_ICD.py --hw_profile ada5000 --epochs 30 --lr 2e-5
```
*(Cấu hình này tự động đặt `Batch Size = 32` và `Gradient Accumulation = 1`)*

**Các tham số có thể tùy chỉnh:**
- `--hw_profile`: Chọn profile phần cứng (`rtx3050` hoặc `ada5000`). Mặc định: `rtx3050`.
- `--epochs`: Số lượng Epoch cần huấn luyện. Mặc định: `30`.
- `--lr`: Learning rate. Mặc định: `2e-5`.

---
## 🗂 Cấu trúc lưu trữ đầu ra
- **Checkpoints/Trọng số (Weights):** Sau mỗi epoch có điểm F1-Score cao nhất, mô hình sẽ tự động lưu lại vào `result/ICD/checkpoints/best_model.pth`.
- **Metrics:** Khi kết thúc quá trình train, kết quả dự đoán (Precision, Recall, F1, Accuracy) trên tập Test sẽ được lưu vào file `result/ICD/test_metrics.csv`.
