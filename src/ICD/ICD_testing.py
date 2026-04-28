import unittest
import torch
from transformers import AutoConfig, AutoModel
from ICD_Model import ContentModelingModule, StyleModelingModule, InteractionModelingModule, ClickbaitDetectionModel, JointLoss

class TestContentModelingModule(unittest.TestCase):
    def setUp(self):
        # Khởi tạo mô hình giả lập (dummy model) để test hình học (shape) nhanh chóng
        print("\n[+] Đang khởi tạo mô hình để test...")
        config = AutoConfig.from_pretrained("vinai/phobert-base")
        config.num_hidden_layers = 2 # Giảm số lượng layer xuống 2 để test cực nhanh
        
        self.module = ContentModelingModule("vinai/phobert-base")
        self.module.phobert = AutoModel.from_config(config) # Override bằng dummy config
        self.module.eval() # Chế độ evaluation
        
        self.batch_size = 4
        self.N = 15 # Độ dài title
        self.P = 35 # Độ dài lead_paragraph
        self.hidden_size = 768

    def test_output_shapes(self):
        # 1. Tạo tensor đầu vào giả lập (Mock inputs)
        title_ids = torch.randint(0, 1000, (self.batch_size, self.N))
        title_mask = torch.ones(self.batch_size, self.N)
        
        lead_ids = torch.randint(0, 1000, (self.batch_size, self.P))
        lead_mask = torch.ones(self.batch_size, self.P)

        # 2. Chạy qua module
        with torch.no_grad():
            H_title, H_lead, e_title, e_lead = self.module(title_ids, title_mask, lead_ids, lead_mask)

        # 3. Kiểm tra các Shape đầu ra có đúng lý thuyết không
        self.assertEqual(H_title.shape, (self.batch_size, self.N, self.hidden_size), "Sai shape của H_title")
        self.assertEqual(H_lead.shape, (self.batch_size, self.P, self.hidden_size), "Sai shape của H_lead")
        self.assertEqual(e_title.shape, (self.batch_size, self.hidden_size), "Sai shape của e_title")
        self.assertEqual(e_lead.shape, (self.batch_size, self.hidden_size), "Sai shape của e_lead")
        print("[+] Test Output Shapes: PASSED")

    def test_attention_masking(self):
        # Kiểm tra xem padding token có bị loại bỏ đúng cách không
        title_ids = torch.randint(0, 1000, (1, self.N))
        title_mask = torch.ones(1, self.N)
        
        # Giả lập 5 token cuối cùng là padding (mask = 0)
        title_mask[0, -5:] = 0 
        
        lead_ids = torch.randint(0, 1000, (1, self.P))
        lead_mask = torch.ones(1, self.P)

        with torch.no_grad():
            _, _, _, e_lead = self.module(title_ids, title_mask, lead_ids, lead_mask)
            # Truy cập thủ công vào WordLevelAttention để lấy alpha
            title_outputs = self.module.phobert(input_ids=title_ids, attention_mask=title_mask)
            _, alpha_title = self.module.title_attention(title_outputs.last_hidden_state, title_mask)

        # Trọng số attention của 5 token cuối cùng phải xấp xỉ bằng 0
        padding_attention_sum = alpha_title[0, -5:].sum().item()
        self.assertAlmostEqual(padding_attention_sum, 0.0, places=5, msg="Padding tokens bị chia attention sai logic!")
        
        # Tổng của tất cả các alpha phải bằng 1
        total_attention = alpha_title.sum().item()
        self.assertAlmostEqual(total_attention, 1.0, places=5, msg="Tổng attention weights khác 1!")
        print("[+] Test Attention Masking: PASSED")

class TestStyleModelingModule(unittest.TestCase):
    def setUp(self):
        print("\n[+] Đang khởi tạo StyleModelingModule để test...")
        self.vocab_size = 150
        self.d_c = 128
        self.module = StyleModelingModule(vocab_size=self.vocab_size, d_c=self.d_c, nhead=4, num_layers=2)
        self.module.eval()
        
        self.batch_size = 4
        self.max_char_len = 100

    def test_output_shapes(self):
        # 1. Tạo tensor đầu vào giả lập
        char_input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_char_len))
        char_attention_mask = torch.ones(self.batch_size, self.max_char_len)
        
        # 2. Chạy qua module
        with torch.no_grad():
            e_style = self.module(char_input_ids, char_attention_mask)
            
        # 3. Kiểm tra shape
        self.assertEqual(e_style.shape, (self.batch_size, self.d_c), "Sai shape của e_style")
        print("[+] Test StyleModeling Output Shapes: PASSED")

    def test_attention_masking(self):
        char_input_ids = torch.randint(0, self.vocab_size, (1, self.max_char_len))
        char_attention_mask = torch.ones(1, self.max_char_len)
        
        # Giả lập 10 token cuối cùng là padding (mask = 0)
        char_attention_mask[0, -10:] = 0
        
        with torch.no_grad():
            # Truy cập các bước bên trong để lấy alpha (trọng số attention)
            x = self.module.char_embedding(char_input_ids)
            padding_mask = (char_attention_mask == 0)
            H_char = self.module.transformer_encoder(x, src_key_padding_mask=padding_mask)
            _, alpha_char = self.module.char_attention(H_char, char_attention_mask)
            
        # Trọng số attention của 10 token padding phải xấp xỉ 0
        padding_attention_sum = alpha_char[0, -10:].sum().item()
        self.assertAlmostEqual(padding_attention_sum, 0.0, places=5, msg="Padding tokens bị chia attention sai logic trong StyleModeling!")
        
        # Tổng của tất cả các alpha phải bằng 1
        total_attention = alpha_char.sum().item()
        self.assertAlmostEqual(total_attention, 1.0, places=5, msg="Tổng attention weights khác 1 trong StyleModeling!")
        print("[+] Test StyleModeling Attention Masking: PASSED")

class TestInteractionModelingModule(unittest.TestCase):
    def setUp(self):
        print("\n[+] Đang khởi tạo InteractionModelingModule để test...")
        self.hidden_size = 768
        self.module = InteractionModelingModule(hidden_size=self.hidden_size)
        self.module.eval()
        
        self.batch_size = 4
        self.N = 15 # Độ dài tiêu đề
        self.P = 35 # Độ dài đoạn dẫn

    def test_output_shapes(self):
        H_title = torch.rand(self.batch_size, self.N, self.hidden_size)
        title_mask = torch.ones(self.batch_size, self.N)
        
        H_lead = torch.rand(self.batch_size, self.P, self.hidden_size)
        lead_mask = torch.ones(self.batch_size, self.P)
        
        with torch.no_grad():
            r_title, r_lead = self.module(H_title, title_mask, H_lead, lead_mask)
            
        self.assertEqual(r_title.shape, (self.batch_size, 2 * self.hidden_size), "Sai shape của r_title")
        self.assertEqual(r_lead.shape, (self.batch_size, 2 * self.hidden_size), "Sai shape của r_lead")
        print("[+] Test Interaction Output Shapes: PASSED")

    def test_attention_masking(self):
        H_title = torch.rand(1, self.N, self.hidden_size)
        title_mask = torch.ones(1, self.N)
        title_mask[0, -5:] = 0 # Giả lập 5 token cuối của title là padding
        
        H_lead = torch.rand(1, self.P, self.hidden_size)
        lead_mask = torch.ones(1, self.P)
        lead_mask[0, -10:] = 0 # Giả lập 10 token cuối của lead là padding
        
        with torch.no_grad():
            r_title, r_lead = self.module(H_title, title_mask, H_lead, lead_mask)
            # Lấy ma trận softmax lưu tạm trong object
            A_T = self.module._A_T_temp
            A_L = self.module._A_L_temp
            
        # A_T: Chú ý của đoạn dẫn đối với tiêu đề (N x P). Trọng số rơi vào 10 padding tokens của lead phải bằng 0.
        padding_sum_T = A_T[0, :, -10:].sum().item()
        self.assertAlmostEqual(padding_sum_T, 0.0, places=5, msg="Trọng số A_T đổ vào padding token của lead khác 0!")
        
        # A_L: Chú ý của tiêu đề đối với đoạn dẫn (P x N). Trọng số rơi vào 5 padding tokens của title phải bằng 0.
        padding_sum_L = A_L[0, :, -5:].sum().item()
        self.assertAlmostEqual(padding_sum_L, 0.0, places=5, msg="Trọng số A_L đổ vào padding token của title khác 0!")
        
        print("[+] Test Interaction Attention Masking: PASSED")

    def test_mean_pooling_with_mask(self):
        # Đảm bảo mean-pooling tính trung bình chính xác, bỏ qua padding token
        H_title = torch.ones(1, self.N, self.hidden_size)
        title_mask = torch.zeros(1, self.N)
        title_mask[0, 0] = 1 # Chỉ có 1 token ở vị trí 0 là thật
        
        H_lead = torch.ones(1, self.P, self.hidden_size)
        lead_mask = torch.zeros(1, self.P)
        lead_mask[0, 0] = 1 # Chỉ có 1 token ở vị trí 0 là thật
        
        # Đặt W_c thành ma trận đơn vị để H_title x W_c x H_lead^T ra kết quả dễ dự đoán
        self.module.W_c.weight.data.copy_(torch.eye(self.hidden_size))
        
        with torch.no_grad():
            r_title, r_lead = self.module(H_title, title_mask, H_lead, lead_mask)
            
        # Vì chỉ có 1 token thực mang giá trị 1 ở mỗi bên, tổng vector thu được sẽ là 1 vector toàn số 1.
        # Nếu hàm chia cho tổng số chiều dài max_len thay vì số lượng token thực thì giá trị sẽ bị nhỏ hơn rất nhiều (1/N hoặc 1/P).
        self.assertAlmostEqual(r_title[0, 0].item(), 1.0, places=5, msg="Mean-Pooling chia sai cho tổng padding thay vì số token thực!")
        self.assertAlmostEqual(r_lead[0, 0].item(), 1.0, places=5, msg="Mean-Pooling chia sai cho tổng padding thay vì số token thực!")
        
        print("[+] Test Mean Pooling With Mask: PASSED")

class TestClickbaitDetectionModel(unittest.TestCase):
    def setUp(self):
        print("\n[+] Đang khởi tạo ClickbaitDetectionModel để test...")
        self.vocab_size = 150
        self.hidden_size = 768
        
        # Rút gọn PhoBERT layer để test chạy nhanh
        config = AutoConfig.from_pretrained("vinai/phobert-base")
        config.num_hidden_layers = 1
        
        self.model = ClickbaitDetectionModel(vocab_size=self.vocab_size, hidden_size=self.hidden_size)
        self.model.content_module.phobert = AutoModel.from_config(config)
        self.model.eval()
        
        self.batch_size = 4
        self.N = 15
        self.P = 35
        self.max_char_len = 100

    def test_forward_pass(self):
        # Tạo dữ liệu giả lập
        title_ids = torch.randint(0, 1000, (self.batch_size, self.N))
        title_mask = torch.ones(self.batch_size, self.N)
        
        lead_ids = torch.randint(0, 1000, (self.batch_size, self.P))
        lead_mask = torch.ones(self.batch_size, self.P)
        
        char_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_char_len))
        char_mask = torch.ones(self.batch_size, self.max_char_len)
        
        with torch.no_grad():
            preds, e_title, e_lead = self.model(title_ids, title_mask, lead_ids, lead_mask, char_ids, char_mask)
            
        # Kiểm tra shape
        self.assertEqual(preds.shape, (self.batch_size, 1), "Sai shape của predictions")
        self.assertEqual(e_title.shape, (self.batch_size, self.hidden_size), "Sai shape của e_title")
        self.assertEqual(e_lead.shape, (self.batch_size, self.hidden_size), "Sai shape của e_lead")
        
        # Kiểm tra giá trị preds nằm trong [0, 1] do sigmoid
        self.assertTrue((preds >= 0).all() and (preds <= 1).all(), "Giá trị preds ngoài khoảng [0, 1]")
        
        print("[+] Test ClickbaitDetectionModel Forward Pass: PASSED")

class TestJointLoss(unittest.TestCase):
    def setUp(self):
        print("\n[+] Đang khởi tạo JointLoss để test...")
        self.loss_fn = JointLoss(margin=1.0, lambda_weight=0.3)
        self.batch_size = 4
        self.hidden_size = 768

    def test_loss_computation(self):
        # Giả lập prediction, nhãn và đặc trưng
        preds = torch.rand(self.batch_size, 1) # [0, 1]
        labels = torch.randint(0, 2, (self.batch_size, 1)).float() # 0 hoặc 1
        
        e_title = torch.rand(self.batch_size, self.hidden_size)
        e_lead = torch.rand(self.batch_size, self.hidden_size)
        
        total_loss, bce_loss, L_CL = self.loss_fn(preds, labels, e_title, e_lead)
        
        # Kiểm tra tính toán không ra NaN
        self.assertFalse(torch.isnan(total_loss), "Total Loss bị NaN")
        self.assertFalse(torch.isnan(bce_loss), "BCE Loss bị NaN")
        self.assertFalse(torch.isnan(L_CL), "Contrastive Loss bị NaN")
        print("[+] Test JointLoss Computation: PASSED")

    def test_contrastive_logic(self):
        # Test case: Clickbait (labels=1) nhưng e_title và e_lead giống hệt nhau (D_cos = 0)
        preds = torch.tensor([[0.8]]) # Dự đoán ngẫu nhiên
        labels = torch.tensor([[1.0]]) # Nhãn là Clickbait
        
        e_title = torch.ones(1, self.hidden_size)
        e_lead = torch.ones(1, self.hidden_size) # Giống hệt e_title
        
        total_loss, bce_loss, L_CL = self.loss_fn(preds, labels, e_title, e_lead)
        
        # Vì labels=1 và D_cos = 0, hàm max(0, margin - D_cos) sẽ bằng margin (1.0)
        # Contrastive Loss (L_CL) phải lớn hơn 0 (bị phạt vì Clickbait mà lại giống nhau)
        self.assertTrue(L_CL.item() > 0, "Lỗi logic Contrastive Loss: D_cos=0 và nhãn=1 nhưng không bị phạt!")
        self.assertAlmostEqual(L_CL.item(), self.loss_fn.margin, places=5, msg="Giá trị L_CL không khớp với logic margin!")
        
        print("[+] Test JointLoss Contrastive Logic: PASSED")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)