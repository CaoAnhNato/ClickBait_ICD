import torch
from torch.utils.data import Dataset
import pandas as pd

class ClickbaitDatasetV5(Dataset):
    """
    Dataset cho ICDv5:
    Trả về:
      - input_ids_news, attention_mask_news (từ title_seg và lead_seg)
      - category_id
      - source_id
      - pattern_tags (6 chiều)
      - label (0/1)
      - soft_label_llm (từ score_init)
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        tokenizer,
        max_len_news: int = 256,
    ):
        self.df = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len_news = max_len_news

        # Các cột cần thiết
        required = [
            "title_seg", "lead_seg", "category_id", "source_id", "label_bin", "soft_label_llm",
            "tag_shock", "tag_lifestyle", "tag_listicle", "tag_analysis", "tag_promo", "tag_hardnews"
        ]
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

        # Metadata
        cat_id = torch.tensor(row["category_id"], dtype=torch.long)
        src_id = torch.tensor(row["source_id"], dtype=torch.long)
        
        # Pattern tags
        tags = [
            row["tag_shock"], row["tag_lifestyle"], row["tag_listicle"], 
            row["tag_analysis"], row["tag_promo"], row["tag_hardnews"]
        ]
        pattern_tags = torch.tensor(tags, dtype=torch.float32)

        # Labels
        label = torch.tensor(row["label_bin"], dtype=torch.float32).unsqueeze(-1)
        soft_label = torch.tensor(row["soft_label_llm"], dtype=torch.float32).unsqueeze(-1)

        return {
            "input_ids_news": enc_news["input_ids"],
            "attention_mask_news": enc_news["attention_mask"],
            "category_id": cat_id,
            "source_id": src_id,
            "pattern_tags": pattern_tags,
            "label": label,
            "soft_label_llm": soft_label,
        }
