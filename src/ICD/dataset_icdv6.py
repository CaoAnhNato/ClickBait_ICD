import torch
from torch.utils.data import Dataset
import pandas as pd

class ClickbaitDatasetV6(Dataset):
    """
    Dataset cho ICDv6. Đơn giản hóa từ v5:
    - Không cần soft_label_llm
    - Pattern tags & metadata là optional
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        tokenizer,
        max_len_news: int = 256,
        use_pattern_tags: bool = False,
        use_metadata: bool = False
    ):
        self.df = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len_news = max_len_news
        self.use_pattern_tags = use_pattern_tags
        self.use_metadata = use_metadata

        # Kiểm tra cột bắt buộc
        required = ["title_seg", "lead_seg", "label"]
        if self.use_pattern_tags:
            required.extend(["tag_shock", "tag_lifestyle", "tag_listicle", "tag_analysis", "tag_promo", "tag_hardnews"])
        if self.use_metadata:
            required.extend(["category_id", "source_id"])

        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Thiếu column: {col}")

    def __len__(self):
        return len(self.df)

    def _encode(self, text_a: str, text_b: str, max_len: int = 256) -> dict:
        enc = self.tokenizer(
            text_a, text_b,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0).float(),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        title_seg = str(row["title_seg"])
        lead_seg = str(row["lead_seg"])

        # Tokenize (title_seg, lead_seg)
        enc_news = self._encode(title_seg, lead_seg, max_len=self.max_len_news)

        label = torch.tensor(row["label"], dtype=torch.float32).unsqueeze(-1)

        result = {
            "input_ids_news": enc_news["input_ids"],
            "attention_mask_news": enc_news["attention_mask"],
            "label": label,
        }

        if self.use_metadata:
            result["category_id"] = torch.tensor(row["category_id"], dtype=torch.long)
            result["source_id"] = torch.tensor(row["source_id"], dtype=torch.long)

        if self.use_pattern_tags:
            tags = [
                row["tag_shock"], row["tag_lifestyle"], row["tag_listicle"], 
                row["tag_analysis"], row["tag_promo"], row["tag_hardnews"]
            ]
            result["pattern_tags"] = torch.tensor(tags, dtype=torch.float32)
            
        return result
