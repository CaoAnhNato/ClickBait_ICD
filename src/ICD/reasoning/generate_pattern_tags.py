"""
ICDv5 - Generate Pattern Tags
=============================
Sinh 6 multi-label pattern tags bằng LLM (gpt-4o-mini).
Sử dụng ProcessPoolExecutor kết hợp ThreadPoolExecutor, vô hạn retry.
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import threading
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

# Configuration
API_KEY = "sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v"
BASE_URL = "https://direct.shopaikey.com/v1"
MODEL_NAME = "gpt-4o-mini"

BASE_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 30.0
RESPONSE_TIMEOUT = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(processName)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Bạn là chuyên gia phân loại báo chí. Hãy phân tích bài báo sau và gán các nhãn (pattern tags) phù hợp dưới dạng multi-label (0 hoặc 1).

6 nhãn bao gồm:
1. tag_shock: Tin tức liên quan đến tội phạm, tình dục, thảm họa, bạo lực, giật gân, rùng rợn.
2. tag_lifestyle: Showbiz, đời sống cá nhân, human story, tâm sự, giới trẻ, drama giải trí.
3. tag_listicle: Bài viết dạng danh sách (VD: 10 cách, 5 thói quen, X lý do...), hướng dẫn, mẹo vặt, tips.
4. tag_analysis: Phân tích chuyên sâu về chính sách, kinh tế, xã hội, khoa học, Q&A nghiêm túc.
5. tag_promo: Quảng bá sự kiện, sản phẩm, thương hiệu, PR, khuyến mãi, deal hot.
6. tag_hardnews: Tin ngắn thời sự, chính trị, ngoại giao, báo cáo trung lập, sự kiện chính quy.

Lưu ý: Một bài báo CÓ THỂ có nhiều nhãn (ví dụ: vừa lifestyle vừa shock), hoặc có thể không có nhãn nào (tất cả là 0).

Bài báo:
Tiêu đề: {title}
Sapo: {lead}

HÃY TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON NHƯ SAU, KHÔNG GIẢI THÍCH GÌ THÊM:
{{
    "tag_shock": 0,
    "tag_lifestyle": 0,
    "tag_listicle": 0,
    "tag_analysis": 0,
    "tag_promo": 0,
    "tag_hardnews": 0
}}
"""

def parse_json_response(text: str) -> dict:
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Parse JSON error: {e}")

def call_llm(client: OpenAI, title: str, lead: str, sample_id: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(title=title, lead=lead)
    attempt = 0
    delay = BASE_RETRY_DELAY
    
    while True:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Bạn là hệ thống trả về JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={ "type": "json_object" },
                max_tokens=256,
                timeout=RESPONSE_TIMEOUT,
            )
            content = response.choices[0].message.content
            if content:
                parsed = parse_json_response(content)
                # Ensure all 6 keys exist
                for key in ["tag_shock", "tag_lifestyle", "tag_listicle", "tag_analysis", "tag_promo", "tag_hardnews"]:
                    if key not in parsed:
                        parsed[key] = 0
                    else:
                        parsed[key] = int(parsed[key])
                return parsed
            raise ValueError("Empty LLM response")
            
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or "quota" in err_str:
                wait = min(delay * (1 + random.uniform(0, 0.5)), MAX_RETRY_DELAY)
                time.sleep(wait)
                delay = min(delay * 2, MAX_RETRY_DELAY)
            elif "timeout" in err_str or "connection" in err_str:
                time.sleep(min(delay, MAX_RETRY_DELAY))
            else:
                time.sleep(delay)
                delay = min(delay * 1.5, MAX_RETRY_DELAY)


def process_chunk(chunk_idx: int, rows: list, n_threads: int, temp_file_path: Path):
    """Xử lý một chunk dữ liệu bằng ThreadPoolExecutor."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    write_lock = threading.Lock()
    
    # Load already done in this chunk if resuming
    done_ids = set()
    if temp_file_path.exists():
        with open(temp_file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                done_ids.add(data["id"])
                
    rows_to_process = [r for r in rows if r["id"] not in done_ids]
    
    def process_row(row):
        sample_id = row["id"]
        title = str(row.get("title", ""))
        lead = str(row.get("lead_paragraph", ""))
        
        tags = call_llm(client, title, lead, sample_id)
        
        result = {"id": sample_id}
        result.update(tags)
        
        with write_lock:
            with open(temp_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return sample_id
        
    completed = 0
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(process_row, row): row for row in rows_to_process}
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
            except Exception as e:
                # Vô hạn retry bên trong call_llm đã lo, nếu ra ngoài này là lỗi nghiêm trọng
                logger.error(f"[Chunk {chunk_idx}] Lỗi cực kì nghiêm trọng: {e}")
                
    return chunk_idx, completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    
    input_file = BASE_DIR / "data" / "processed" / "icdv5" / f"icdv5_{args.split}_base.parquet"
    output_dir = BASE_DIR / "data" / "processed" / "icdv5"
    output_parquet = output_dir / f"icdv5_{args.split}_patterns.parquet"
    temp_dir = output_dir / "temp_patterns"
    temp_dir.mkdir(exist_ok=True)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
        
    df = pd.read_parquet(input_file)
    df["id"] = df["id"].astype(str)
    all_rows = df.to_dict("records")
    
    # Split into chunks for processes
    n_chunks = args.processes
    chunk_size = len(all_rows) // n_chunks + 1
    chunks = [all_rows[i:i + chunk_size] for i in range(0, len(all_rows), chunk_size)]
    
    logger.info(f"Bắt đầu xử lý {len(all_rows)} samples ({n_chunks} processes x {args.threads} threads)")
    
    futures = []
    with ProcessPoolExecutor(max_workers=n_chunks) as executor:
        for i, chunk in enumerate(chunks):
            temp_file = temp_dir / f"{args.split}_chunk_{i}.jsonl"
            futures.append(executor.submit(process_chunk, i, chunk, args.threads, temp_file))
            
        for future in as_completed(futures):
            chunk_idx, count = future.result()
            logger.info(f"Chunk {chunk_idx} completed {count} items.")
            
    # Merge temp files
    logger.info("Merging temporary files...")
    all_results = []
    for i in range(len(chunks)):
        temp_file = temp_dir / f"{args.split}_chunk_{i}.jsonl"
        if temp_file.exists():
            with open(temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    all_results.append(json.loads(line))
            # Clean up
            temp_file.unlink()
            
    # Save final parquet
    res_df = pd.DataFrame(all_results)
    if len(res_df) > 0:
        res_df.to_parquet(output_parquet, index=False)
        logger.info(f"Saved {len(res_df)} pattern tags to {output_parquet}")
    else:
        logger.warning("No results to save.")
        
if __name__ == "__main__":
    main()
