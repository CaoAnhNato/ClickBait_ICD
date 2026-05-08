"""
ICDv4 – SORG Prompt Templates (Tiếng Việt)
==========================================
Prompt templates cho việc sinh reasoning đối nghịch theo framework SORG
(Self-renewal Opposing-stance Reasoning Generation).

Ref: Zhang et al. (WWW'26) – "Acting Flatterers via LLMs Sycophancy:
     Combating Clickbait with LLMs Opposing-Stance Reasoning"
     arXiv: 2601.12019
"""

# ===========================================================================
# SORG Hyperparameters (theo optimal settings từ paper)
# ===========================================================================
ALPHA = 30    # Khoảng cách tới biên 0/100 cho initial rating: V_I ∈ [α, 100-α]
BETA = 10     # Expected rating change tối thiểu: |V_A - V_I| >= β
GAMMA = 5     # Polarity threshold: V_A >= 50+γ và V_D <= 50-γ
MAX_ITER = 5  # Số lần retry tối đa (tăng lên 5 để đảm bảo data không thiếu)
SOFT_LABEL_LAMBDA = 0.2  # λ cho soft label: p_llm_final blend


# ===========================================================================
# Bước A – Initial Title Rating (Algorithm 1)
# ===========================================================================
def build_initial_rating_prompt(title: str, lead: str) -> str:
    """
    Prompt đánh giá mức độ clickbait ban đầu của headline.
    
    LLM trả về JSON với 2 trường:
      - score: int (0-100)
      - reason: str (giải thích ngắn)
    """
    return f"""Bạn là chuyên gia phân tích báo chí trực tuyến.

Nhiệm vụ: Đánh giá mức độ "clickbait" của tiêu đề tin tức dưới đây.

Tiêu đề: {title}
Mô tả: {lead}

Định nghĩa clickbait: Tiêu đề sử dụng ngôn ngữ phóng đại, mơ hồ, hoặc gây tò mò có chủ đích \
để thu hút click mà không cung cấp đủ thông tin thực chất.

Hãy đánh giá theo thang điểm 0-100:
- 0: Hoàn toàn không phải clickbait (thông tin rõ ràng, trung thực)
- 50: Khó xác định (có yếu tố gây tò mò nhưng không rõ ràng)
- 100: Clickbait điển hình (cố tình gây hiểu lầm, phóng đại)

Yêu cầu phân tích dựa trên 4 tiêu chí:
1. Thông thường: Tiêu đề có chứa thông tin sai lệch hoặc bất thường không?
2. Logic: Có sự nhảy vọt hoặc mâu thuẫn trong lập luận không?
3. Đầy đủ thông tin: Có thông tin bị cố tình che giấu hoặc tạo hồi hộp không?
4. Khách quan: Có ngôn ngữ cảm xúc hoặc kích động không?

Trả về ĐÚNG định dạng JSON sau (không thêm gì khác):
{{"score": <số nguyên 0-100>, "reason": "<giải thích ngắn gọn 30-50 từ tiếng Việt>"}}"""


def build_re_rating_prompt(title: str, lead: str, prev_score: int,
                            prev_reason: str, direction: str) -> str:
    """
    Prompt re-rating khi V_I nằm ngoài [α, 100-α].
    direction: "tăng" hoặc "giảm"
    """
    return f"""Bạn là chuyên gia phân tích báo chí trực tuyến.

Tiêu đề: {title}
Mô tả: {lead}

Đánh giá trước đó:
- Điểm: {prev_score}/100
- Lý do: {prev_reason}

Vấn đề: Điểm {prev_score} quá {'thấp (gần 0)' if direction == 'tăng' else 'cao (gần 100)'}, \
cần phải {direction} để phản ánh mức độ không chắc chắn thực sự.

Hãy đánh giá lại với điểm số nằm trong khoảng [{ALPHA}, {100 - ALPHA}].

Trả về ĐÚNG định dạng JSON sau (không thêm gì khác):
{{"score": <số nguyên {ALPHA}-{100 - ALPHA}>, "reason": "<giải thích lại 30-50 từ tiếng Việt>"}}"""


# ===========================================================================
# Bước B – Agree Reasoning (giả sử là clickbait)
# ===========================================================================
def build_agree_reasoning_prompt(title: str, lead: str,
                                  initial_score: int, initial_reason: str) -> str:
    """
    Prompt sinh agree reasoning: LLM giả sử headline là clickbait
    và giải thích tại sao, sau đó cho điểm mới V_A > V_I.
    """
    return f"""Bạn là chuyên gia phân tích báo chí trực tuyến.

Tiêu đề: {title}
Mô tả: {lead}

Đánh giá ban đầu (trung lập): {initial_score}/100
Lý do ban đầu: {initial_reason}

**Nhiệm vụ: Hãy giả định rằng tiêu đề này LÀ clickbait và viết lập luận giải thích tại sao.**

Phân tích từ 4 khía cạnh để ủng hộ quan điểm "đây là clickbait":
1. Thông thường: Tiêu đề có chứa thông tin khó tin hoặc bất thường không?
2. Logic: Có sự nhảy vọt, mâu thuẫn hoặc thiếu căn cứ không?
3. Đầy đủ thông tin: Có thông tin quan trọng bị che giấu hoặc tạo hồi hộp không cần thiết?
4. Khách quan: Có ngôn ngữ cảm xúc, phóng đại hoặc kích động không?

Yêu cầu:
- Lập luận phải thuyết phục và làm tăng mức độ clickbait (điểm mới PHẢI cao hơn {initial_score})
- Điểm mới (V_A) phải thỏa: V_A >= {50 + GAMMA} và V_A - {initial_score} >= {BETA}
- Reasoning: 40-60 từ tiếng Việt

Trả về ĐÚNG định dạng JSON sau (không thêm gì khác):
{{"reasoning": "<lập luận 40-60 từ tiếng Việt>", "score": <số nguyên thỏa điều kiện>}}"""


def build_agree_analysis_prompt(title: str, prev_reasoning: str,
                                 prev_score: int, initial_score: int) -> str:
    """
    Prompt phân tích tại sao agree reasoning chưa đạt yêu cầu.
    (Gọi khi V_A chưa đủ cao hoặc không tăng đủ so với V_I)
    """
    return f"""Bạn là chuyên gia phân tích chất lượng lập luận.

Tiêu đề đang xét đề cập đến clickbait.
Điểm ban đầu (V_I): {initial_score}/100
Lập luận "agree" trước đó: {prev_reasoning}
Điểm sau lập luận (V_A): {prev_score}/100

**Vấn đề**: Lập luận chưa đủ thuyết phục. Yêu cầu V_A >= {50 + GAMMA} và V_A - V_I >= {BETA}.

Phân tích điểm yếu của lập luận trên (tính hợp lý, logic, tính thuyết phục).

Trả về ĐÚNG định dạng JSON sau (không thêm gì khác):
{{"analysis": "<phân tích điểm yếu 30-50 từ tiếng Việt>"}}"""


def build_agree_regenerate_prompt(title: str, lead: str, initial_score: int,
                                   initial_reason: str, prev_reasoning: str,
                                   prev_score: int, analysis: str) -> str:
    """
    Prompt tái sinh agree reasoning dựa trên phân tích điểm yếu.
    """
    return f"""Bạn là chuyên gia phân tích báo chí trực tuyến.

Tiêu đề: {title}
Mô tả: {lead}

Điểm ban đầu: {initial_score}/100
Lý do ban đầu: {initial_reason}

Lập luận trước đó (chưa đạt): {prev_reasoning}
Điểm trước đó: {prev_score}/100
Phân tích điểm yếu: {analysis}

**Nhiệm vụ: Tái sinh lập luận "clickbait" mạnh hơn**, khắc phục các điểm yếu đã phân tích.

Yêu cầu:
- Lập luận mới PHẢI thuyết phục hơn, điểm mới >= {50 + GAMMA} và tăng ít nhất {BETA} so với {initial_score}
- Phân tích đầy đủ 4 khía cạnh: thông thường, logic, đầy đủ thông tin, khách quan
- Reasoning: 40-60 từ tiếng Việt

Trả về ĐÚNG định dạng JSON sau (không thêm gì khác):
{{"reasoning": "<lập luận mới 40-60 từ tiếng Việt>", "score": <số nguyên thỏa điều kiện>, "explanation": "<giải thích cải tiến 20-30 từ>"}}"""


# ===========================================================================
# Bước C – Disagree Reasoning (giả sử KHÔNG phải clickbait)
# ===========================================================================
def build_disagree_reasoning_prompt(title: str, lead: str,
                                     initial_score: int, initial_reason: str) -> str:
    """
    Prompt sinh disagree reasoning: LLM giả sử headline KHÔNG phải clickbait
    và giải thích tại sao, sau đó cho điểm mới V_D < V_I.
    """
    return f"""Bạn là chuyên gia phân tích báo chí trực tuyến.

Tiêu đề: {title}
Mô tả: {lead}

Đánh giá ban đầu (trung lập): {initial_score}/100
Lý do ban đầu: {initial_reason}

**Nhiệm vụ: Hãy giả định rằng tiêu đề này KHÔNG phải clickbait và viết lập luận giải thích tại sao.**

Phân tích từ 4 khía cạnh để bác bỏ quan điểm "đây là clickbait":
1. Thông thường: Tiêu đề phản ánh thông tin thực tế, có thể xảy ra trong cuộc sống?
2. Logic: Lập luận nhất quán, không có sự nhảy vọt?
3. Đầy đủ thông tin: Tiêu đề cung cấp đủ thông tin để hiểu nội dung?
4. Khách quan: Ngôn ngữ trung lập, không kích động cảm xúc?

Yêu cầu:
- Lập luận phải thuyết phục và làm giảm mức độ clickbait (điểm mới PHẢI thấp hơn {initial_score})
- Điểm mới (V_D) phải thỏa: V_D <= {50 - GAMMA} và {initial_score} - V_D >= {BETA}
- Reasoning: 40-60 từ tiếng Việt

Trả về ĐÚNG định dạng JSON sau (không thêm gì khác):
{{"reasoning": "<lập luận 40-60 từ tiếng Việt>", "score": <số nguyên thỏa điều kiện>}}"""


def build_disagree_analysis_prompt(title: str, prev_reasoning: str,
                                    prev_score: int, initial_score: int) -> str:
    """
    Prompt phân tích tại sao disagree reasoning chưa đạt yêu cầu.
    """
    return f"""Bạn là chuyên gia phân tích chất lượng lập luận.

Tiêu đề đang xét không phải clickbait.
Điểm ban đầu (V_I): {initial_score}/100
Lập luận "disagree" trước đó: {prev_reasoning}
Điểm sau lập luận (V_D): {prev_score}/100

**Vấn đề**: Lập luận chưa đủ thuyết phục. Yêu cầu V_D <= {50 - GAMMA} và V_I - V_D >= {BETA}.

Phân tích điểm yếu của lập luận trên (tính hợp lý, logic, tính thuyết phục).

Trả về ĐÚNG định dạng JSON sau (không thêm gì khác):
{{"analysis": "<phân tích điểm yếu 30-50 từ tiếng Việt>"}}"""


def build_disagree_regenerate_prompt(title: str, lead: str, initial_score: int,
                                      initial_reason: str, prev_reasoning: str,
                                      prev_score: int, analysis: str) -> str:
    """
    Prompt tái sinh disagree reasoning dựa trên phân tích điểm yếu.
    """
    return f"""Bạn là chuyên gia phân tích báo chí trực tuyến.

Tiêu đề: {title}
Mô tả: {lead}

Điểm ban đầu: {initial_score}/100
Lý do ban đầu: {initial_reason}

Lập luận trước đó (chưa đạt): {prev_reasoning}
Điểm trước đó: {prev_score}/100
Phân tích điểm yếu: {analysis}

**Nhiệm vụ: Tái sinh lập luận "non-clickbait" mạnh hơn**, khắc phục các điểm yếu đã phân tích.

Yêu cầu:
- Lập luận mới PHẢI thuyết phục hơn, điểm mới <= {50 - GAMMA} và giảm ít nhất {BETA} so với {initial_score}
- Phân tích đầy đủ 4 khía cạnh: thông thường, logic, đầy đủ thông tin, khách quan
- Reasoning: 40-60 từ tiếng Việt

Trả về ĐÚNG định dạng JSON sau (không thêm gì khác):
{{"reasoning": "<lập luận mới 40-60 từ tiếng Việt>", "score": <số nguyên thỏa điều kiện>, "explanation": "<giải thích cải tiến 20-30 từ>"}}"""
