---
trigger: always_on
---

- Khi thực hiện bất kì quá trình training, testing, validating hoặc inference, dữ liệu text đều phải được tiền xử lý (tokenizer) với thư viện VNcoreNLP.
- Với dataset ViClickbait trong project này, luôn phải thực hiện concate features 'title' và 'lead_paragraph' với tokenizer(text_a, text_b) theo chuẩn deep learning để mô hình ngôn ngữ có thể phân biệt rõ đây là 2 câu khác nhau, đây sẽ là features input. Nếu có thêm thuộc tính nào thì user sẽ thông báo.
- Luôn phải sử dụng env 'MLE'.
- Nhãn 0 = Non-Clickbait, 1 = Clickbait.