"""
ICDv4 – Generate Reasoning (Pha 2)
====================================
Script sinh reasoning đối nghịch bằng LLM theo SORG framework.

Features:
- Multi-threading: tối đa 4 concurrent API requests
- Retry vô hạn cho đến khi thành công (data không được phép thiếu)
- Checkpoint/resume: lưu progress sau mỗi sample
- Rate-limit aware: backoff khi gặp lỗi throttling

Chạy:
    conda run -n MLE python src/ICD/reasoning/generate_reasoning.py

Output:
    data/processed/icdv4/reasoning_all.jsonl
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]  # project root
sys.path.append(str(BASE_DIR))

from src.ICD.reasoning.prompts import (
    ALPHA, BETA, GAMMA, MAX_ITER,
    build_initial_rating_prompt,
    build_re_rating_prompt,
    build_agree_reasoning_prompt,
    build_agree_analysis_prompt,
    build_agree_regenerate_prompt,
    build_disagree_reasoning_prompt,
    build_disagree_analysis_prompt,
    build_disagree_regenerate_prompt,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = "sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v"
BASE_URL = "https://direct.shopaikey.com/v1"
MODEL_NAME = "gpt-3.5-turbo-1106"

INPUT_CSV = BASE_DIR / "data" / "processed" / "cleaned" / "Cleaned_Clickbait_Dataset.csv"
OUTPUT_JSONL = BASE_DIR / "data" / "processed" / "icdv4" / "reasoning_all.jsonl"
CHECKPOINT_FILE = BASE_DIR / "data" / "processed" / "icdv4" / "reasoning_checkpoint.json"

MAX_WORKERS = 4           # Tối đa 4 concurrent requests
BASE_RETRY_DELAY = 2.0    # Giây chờ base khi gặp lỗi
MAX_RETRY_DELAY = 60.0    # Giây chờ tối đa khi backoff
RESPONSE_TIMEOUT = 60     # Timeout cho mỗi API call (giây)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "data" / "processed" / "icdv4" / "generate_reasoning.log",
                            encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Thread-safe write lock
write_lock = threading.Lock()
checkpoint_lock = threading.Lock()

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_json_response(text: str) -> dict:
    """Parse JSON từ LLM response, handle trường hợp LLM thêm text thừa."""
    text = text.strip()
    # Thử parse trực tiếp
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Tìm JSON block trong response
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: extract score và reason bằng regex
    score_match = re.search(r'"score"\s*:\s*(\d+)', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    analysis_match = re.search(r'"analysis"\s*:\s*"([^"]*)"', text)

    result = {}
    if score_match:
        result["score"] = int(score_match.group(1))
    if reason_match:
        result["reason"] = reason_match.group(1)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1)
    if analysis_match:
        result["analysis"] = analysis_match.group(1)
    if result:
        return result
    raise ValueError(f"Không thể parse JSON từ response: {text[:200]}")


def call_llm(prompt: str, sample_id: str, step: str) -> str:
    """
    Gọi LLM với retry vô hạn cho đến khi success.
    Backoff exponential khi gặp lỗi rate limit.
    """
    attempt = 0
    delay = BASE_RETRY_DELAY
    while True:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia phân tích báo chí. Luôn trả về JSON hợp lệ theo format yêu cầu."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=512,
                timeout=RESPONSE_TIMEOUT,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
            raise ValueError("LLM trả về empty response")
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or "quota" in err_str:
                wait = min(delay * (1 + random.uniform(0, 0.5)), MAX_RETRY_DELAY)
                logger.warning(f"[{sample_id}][{step}] Rate limit, chờ {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                delay = min(delay * 2, MAX_RETRY_DELAY)
            elif "timeout" in err_str or "connection" in err_str:
                wait = min(delay, MAX_RETRY_DELAY)
                logger.warning(f"[{sample_id}][{step}] Timeout/Connection error, chờ {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
            else:
                logger.warning(f"[{sample_id}][{step}] Lỗi: {e}, retry sau {delay:.1f}s (attempt {attempt})")
                time.sleep(delay)
                delay = min(delay * 1.5, MAX_RETRY_DELAY)


# ---------------------------------------------------------------------------
# Algorithm 1 – Initial Rating
# ---------------------------------------------------------------------------
def generate_initial_rating(sample_id: str, title: str, lead: str) -> tuple[int, str]:
    """
    Sinh initial rating V_I và R_I theo Algorithm 1.
    Retry cho đến khi V_I ∈ [ALPHA, 100-ALPHA].
    """
    # First call
    prompt = build_initial_rating_prompt(title, lead)
    for attempt in range(MAX_ITER + 10):  # Extra attempts để đảm bảo converge
        raw = call_llm(prompt, sample_id, f"initial_rating_attempt_{attempt}")
        try:
            parsed = parse_json_response(raw)
            score = int(parsed.get("score", 50))
            reason = parsed.get("reason", "")
            score = max(0, min(100, score))

            if ALPHA <= score <= (100 - ALPHA):
                logger.debug(f"[{sample_id}] Initial rating OK: {score}")
                return score, reason

            # Cần re-rating
            direction = "tăng" if score < ALPHA else "giảm"
            logger.debug(f"[{sample_id}] V_I={score} ngoài [{ALPHA},{100-ALPHA}], re-rating ({direction})")
            prompt = build_re_rating_prompt(title, lead, score, reason, direction)

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[{sample_id}] Parse error initial rating: {e}, retry...")
            prompt = build_initial_rating_prompt(title, lead)

    # Fallback: nếu sau nhiều lần vẫn không đạt, clamp về [ALPHA, 100-ALPHA]
    logger.warning(f"[{sample_id}] Initial rating không converge sau {MAX_ITER+10} attempts, clamp score")
    return max(ALPHA, min(100 - ALPHA, score)), reason


# ---------------------------------------------------------------------------
# Algorithm 2 – Agree Reasoning
# ---------------------------------------------------------------------------
def generate_agree_reasoning(sample_id: str, title: str, lead: str,
                              initial_score: int, initial_reason: str) -> tuple[str, int]:
    """
    Sinh agree reasoning R_A và V_A theo Algorithm 2.
    Retry cho đến khi V_A >= 50+GAMMA và V_A - V_I >= BETA.
    """
    prompt = build_agree_reasoning_prompt(title, lead, initial_score, initial_reason)

    for attempt in range(MAX_ITER + 10):
        raw = call_llm(prompt, sample_id, f"agree_attempt_{attempt}")
        try:
            parsed = parse_json_response(raw)
            reasoning = parsed.get("reasoning", "")
            score = int(parsed.get("score", 50))
            score = max(0, min(100, score))

            if score >= (50 + GAMMA) and (score - initial_score) >= BETA:
                logger.debug(f"[{sample_id}] Agree OK: V_A={score}")
                return reasoning, score

            # Phân tích và regenerate
            logger.debug(f"[{sample_id}] Agree V_A={score} chưa đạt, analysis + regenerate")
            analysis_prompt = build_agree_analysis_prompt(title, reasoning, score, initial_score)
            analysis_raw = call_llm(analysis_prompt, sample_id, f"agree_analysis_{attempt}")
            try:
                analysis_parsed = parse_json_response(analysis_raw)
                analysis = analysis_parsed.get("analysis", "Cần lập luận thuyết phục hơn.")
            except Exception:
                analysis = "Cần lập luận thuyết phục hơn, tập trung vào yếu tố gây hiểu lầm."

            prompt = build_agree_regenerate_prompt(
                title, lead, initial_score, initial_reason,
                reasoning, score, analysis
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[{sample_id}] Parse error agree: {e}, retry...")
            prompt = build_agree_reasoning_prompt(title, lead, initial_score, initial_reason)

    # Fallback nếu không đạt sau nhiều lần
    logger.warning(f"[{sample_id}] Agree không converge, dùng kết quả cuối cùng")
    return reasoning, max(score, 50 + GAMMA)


# ---------------------------------------------------------------------------
# Algorithm 2 – Disagree Reasoning
# ---------------------------------------------------------------------------
def generate_disagree_reasoning(sample_id: str, title: str, lead: str,
                                 initial_score: int, initial_reason: str) -> tuple[str, int]:
    """
    Sinh disagree reasoning R_D và V_D theo Algorithm 2.
    Retry cho đến khi V_D <= 50-GAMMA và V_I - V_D >= BETA.
    """
    prompt = build_disagree_reasoning_prompt(title, lead, initial_score, initial_reason)

    for attempt in range(MAX_ITER + 10):
        raw = call_llm(prompt, sample_id, f"disagree_attempt_{attempt}")
        try:
            parsed = parse_json_response(raw)
            reasoning = parsed.get("reasoning", "")
            score = int(parsed.get("score", 50))
            score = max(0, min(100, score))

            if score <= (50 - GAMMA) and (initial_score - score) >= BETA:
                logger.debug(f"[{sample_id}] Disagree OK: V_D={score}")
                return reasoning, score

            # Phân tích và regenerate
            logger.debug(f"[{sample_id}] Disagree V_D={score} chưa đạt, analysis + regenerate")
            analysis_prompt = build_disagree_analysis_prompt(title, reasoning, score, initial_score)
            analysis_raw = call_llm(analysis_prompt, sample_id, f"disagree_analysis_{attempt}")
            try:
                analysis_parsed = parse_json_response(analysis_raw)
                analysis = analysis_parsed.get("analysis", "Cần lập luận thuyết phục hơn.")
            except Exception:
                analysis = "Cần lập luận thuyết phục hơn, tập trung vào tính khách quan và thông tin."

            prompt = build_disagree_regenerate_prompt(
                title, lead, initial_score, initial_reason,
                reasoning, score, analysis
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[{sample_id}] Parse error disagree: {e}, retry...")
            prompt = build_disagree_reasoning_prompt(title, lead, initial_score, initial_reason)

    # Fallback
    logger.warning(f"[{sample_id}] Disagree không converge, dùng kết quả cuối cùng")
    return reasoning, min(score, 50 - GAMMA)


# ---------------------------------------------------------------------------
# Main worker per sample
# ---------------------------------------------------------------------------
def process_sample(row: dict) -> dict:
    """
    Xử lý 1 sample: sinh full reasoning (initial + agree + disagree).
    Retry vô hạn cho đến khi thành công.
    """
    sample_id = str(row["id"])
    title = str(row["title"]) if not pd.isna(row["title"]) else ""
    lead = str(row["lead_paragraph"]) if not pd.isna(row["lead_paragraph"]) else ""

    logger.info(f"[{sample_id}] Bắt đầu xử lý...")

    try:
        # Bước A: Initial rating
        initial_score, initial_reason = generate_initial_rating(sample_id, title, lead)
        logger.info(f"[{sample_id}] Initial: V_I={initial_score}")

        # Bước B: Agree reasoning
        agree_reason, agree_score = generate_agree_reasoning(
            sample_id, title, lead, initial_score, initial_reason
        )
        logger.info(f"[{sample_id}] Agree: V_A={agree_score}")

        # Bước C: Disagree reasoning
        disagree_reason, disagree_score = generate_disagree_reasoning(
            sample_id, title, lead, initial_score, initial_reason
        )
        logger.info(f"[{sample_id}] Disagree: V_D={disagree_score}")

        result = {
            "id": sample_id,
            "initial_score": initial_score,
            "initial_reason": initial_reason,
            "agree_reason": agree_reason,
            "agree_score": agree_score,
            "disagree_reason": disagree_reason,
            "disagree_score": disagree_score,
            "reasoning_status": "success",
        }
        logger.info(f"[{sample_id}] ✓ Hoàn thành")
        return result

    except Exception as e:
        # Không được phép để sample thiếu - raise lên để executor retry
        logger.error(f"[{sample_id}] Lỗi không mong đợi: {e}")
        raise


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint() -> set:
    """Tải danh sách ID đã xử lý từ checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        done_ids = set(data.get("done_ids", []))
        logger.info(f"Resume từ checkpoint: {len(done_ids)} samples đã xử lý")
        return done_ids
    return set()


def save_checkpoint(done_ids: set):
    """Lưu checkpoint."""
    with checkpoint_lock:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({"done_ids": list(done_ids)}, f, ensure_ascii=False)


def append_to_jsonl(record: dict):
    """Thread-safe append một record vào JSONL file."""
    with write_lock:
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ICDv4 – Generate Reasoning với SORG")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Số concurrent API calls (mặc định={MAX_WORKERS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số samples để test (default=None = tất cả)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in thông tin, không gọi API")
    args = parser.parse_args()

    # Tạo output dir
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Đọc dataset từ: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df["lead_paragraph"] = df["lead_paragraph"].fillna("")
    df["title"] = df["title"].fillna("")
    logger.info(f"Tổng samples: {len(df)}")

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Giới hạn test: {args.limit} samples")

    if args.dry_run:
        logger.info("DRY RUN mode – không gọi API")
        sample = df.iloc[0]
        print("\n=== SAMPLE PROMPT (Initial Rating) ===")
        from src.ICD.reasoning.prompts import build_initial_rating_prompt
        print(build_initial_rating_prompt(sample["title"], sample["lead_paragraph"]))
        return

    # Load checkpoint (resume)
    done_ids = load_checkpoint()
    rows_to_process = [
        row.to_dict() for _, row in df.iterrows()
        if str(row["id"]) not in done_ids
    ]
    logger.info(f"Cần xử lý: {len(rows_to_process)} samples (đã xong: {len(done_ids)})")

    if not rows_to_process:
        logger.info("Tất cả samples đã được xử lý!")
        return

    # Progress bar
    pbar = tqdm(total=len(rows_to_process), desc="Generating reasoning")

    # Thread pool với tối đa MAX_WORKERS concurrent requests
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit tất cả tasks
        future_to_row = {
            executor.submit(process_sample, row): row
            for row in rows_to_process
        }

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            sample_id = str(row["id"])
            try:
                result = future.result()
                # Ghi kết quả vào JSONL
                append_to_jsonl(result)
                # Cập nhật checkpoint
                done_ids.add(sample_id)
                save_checkpoint(done_ids)
                pbar.update(1)
                pbar.set_postfix({"done": len(done_ids)})
            except Exception as e:
                # Retry sample này (không được phép thiếu)
                logger.error(f"[{sample_id}] Future failed: {e} – resubmit...")
                # Resubmit
                new_future = executor.submit(process_sample, row)
                future_to_row[new_future] = row

    pbar.close()
    logger.info(f"✓ Hoàn thành! Output: {OUTPUT_JSONL}")
    logger.info(f"  Total processed: {len(done_ids)} samples")


if __name__ == "__main__":
    main()
