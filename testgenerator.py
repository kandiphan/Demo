from openai import OpenAI

# ===================== LLM SETUP =====================
GROQ_API_KEY = "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"


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "llama-3.1-8b-instant"

def call_llm(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Bạn là mô hình ngôn ngữ tuân thủ tuyệt đối định dạng và chỉ dùng dữ liệu trong CONTEXT."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()

# ===================== CONTEXT BUILDER (DÙNG CHUNG) =====================

def fnum(x, nd=3):
    return f"{x:.{nd}f}"

def fcap(x):
    return f"{x/1e9:,.0f} tỷ"

def inline(df, fields):
    rows = []
    for _, r in df.iterrows():
        parts = [r["ticker"]]
        for f in fields:
            if f in r:
                if f == "Market_Cap":
                    parts.append(f"{f} {fcap(r[f])}")
                else:
                    parts.append(f"{f} {fnum(r[f])}")
        rows.append(" (" + ", ".join(parts) + ")")
    return " và ".join(rows)

    

def build_context(title, df, fields):
    return f"""
{title.upper()}:

{inline(df, fields)}
"""

# ===================== RECOMMENDATION =====================

def build_recommend_context(groups, industry):
    return f"""
NGÀNH: {industry}

TOP THEO SCORE:
{inline(groups["top_score"], ["Score", "Market_Cap"])}

TOP THEO MARKET CAP:
{inline(groups["top_cap"], ["Score", "Market_Cap"])}

CÂN BẰNG:
{inline(groups["best_balance"], ["Score", "Market_Cap"])}
"""

def ask_llm_recommend(context, industry):
    prompt = f"""
Bạn là chuyên gia phân tích đầu tư định lượng. 
CHỈ được sử dụng dữ liệu trong CONTEXT.

{context}

Yêu cầu:
- Viết đúng 2 đoạn phân tích + 1 đoạn kết luận.
- Không bullet, không tiêu đề.
- Mỗi mã phải kèm: Ticker (Score x.xxx, Market Cap y tỷ).
- Văn phong học thuật.
- Đoạn kết luận phải đưa ra khoảng cổ phiếu cụ thể được đánh giá là đáng để đầu tư, là cổ phiếu vừa thuộc nhóm vốn hóa lớn vừa có Score cao nhất trong nhóm này, thể hiện sự cân bằng giữa chất lượng doanh nghiệp, mức độ ổn định và tiềm năng sinh lời dài hạn.

Câu mở đầu đoạn 1:
"Dựa vào dữ liệu và mô hình đã sử dụng, ngành {industry} có:"
Đoạn 1:
Phân tích nhóm cổ phiếu có Score cao, nhấn mạnh đây là nhóm có tiềm năng tăng trưởng tương đối trong ngành nhưng đi kèm mức độ rủi ro cao hơn do quy mô vốn hóa nhỏ, độ ổn định và thanh khoản hạn chế.

Đoạn 2:
Phân tích nhóm dẫn đầu về quy mô vốn hóa, làm rõ vai trò của các cổ phiếu này như trụ cột ngành, có thanh khoản tốt, biến động thấp và phù hợp với chiến lược đầu tư an toàn, dài hạn.

Đoạn kết luận (1–2 câu, không dài hơn):
Từ góc độ đầu tư bền vững, cổ phiếu đáng chú ý nhất là cổ phiếu thuộc nhóm dẫn đầu về quy mô vốn hóa nhưng có Score cao nhất, thể hiện sự cân bằng giữa chất lượng doanh nghiệp, mức độ ổn định và tiềm năng sinh lời dài hạn.
"""
    return call_llm(prompt)

# ===================== COMPARISON =====================

def build_comparison_context(table):
    # Chuẩn hóa tên cột
    col_map = {
        "roa": "ROA",
        "de": "DE",
        "pb": "PB",
        "bv": "BV",
        "market_cap": "Market_Cap"
    }
    table = table.rename(columns={c: col_map[c] for c in table.columns if c in col_map})

    # === CHUẨN HÓA ĐƠN VỊ GIỐNG RECOMMEND ===
    # Đảm bảo Market_Cap luôn là ĐỒNG
    if "Market_Cap" in table.columns:
        # Nếu đang là đơn vị tỷ thì nhân lại 1e9
        if table["Market_Cap"].max() < 1e7:  # heuristic: < 10 triệu tỷ
            table["Market_Cap"] = table["Market_Cap"] * 1e9

    fields = ["Score", "ROA", "DE", "PB", "BV", "Market_Cap"]
    return build_context("So sánh doanh nghiệp", table, fields)



def ask_llm_comparison(context, table):
    a, b = table.iloc[0]["ticker"], table.iloc[1]["ticker"]

    prompt = f"""
Bạn là chuyên gia phân tích tài chính.

CHỈ dùng số liệu trong CONTEXT.

{context}

Yêu cầu:
- 1 đoạn phân tích + 1 câu kết luận.
- Không bullet, không tiêu đề.
- Mỗi mã phải kèm đầy đủ:
  Ticker (Score, ROA, D/E, BV, P/B, Market Cap).
- Sau BV và Market Cap phải ghi đơn vị "tỷ"
- Lưu ý: Score càng cao cho thấy cấu trúc tài chính càng tối ưu cho khả năng tạo lợi nhuận trên mỗi cổ phiếu trong dài hạn

Câu mở đầu:
"Dựa trên số liệu tài chính, có thể so sánh {a} và {b} như sau:"

Phải nêu:
- So sánh Score
- So sánh ROA (sinh lời)
- So sánh D/E (rủi ro)
- So sánh P/B (định giá)
- So sánh BV

Câu kết luận:
Tổng hợp lại, cổ phiếu đáng ưu tiên đầu tư hơn là ..., vì ...
"""
    return call_llm(prompt)

# ===================== RANKING =====================

def build_ranking_context(table, industry, factor):
    # Chuẩn hóa tên cột
    col_map = {
        "roa": "ROA",
        "de": "DE",
        "pb": "PB",
        "bv": "BV",
        "market_cap": "Market_Cap"
    }
    table = table.rename(columns={c: col_map[c] for c in table.columns if c in col_map})

    return build_context(
        f"Top {len(table)} cổ phiếu ngành {industry} theo {factor}",
        table,
        [factor]
    )


def ask_llm_ranking(context, industry, factor):
    prompt = f"""
Bạn là hệ thống báo cáo định lượng.

CHỈ sao chép và sắp xếp lại dữ liệu trong CONTEXT.

{context}

Yêu cầu:
- 1 đoạn duy nhất.
- Mỗi mã: MÃ ({factor} = GIÁ_TRỊ đúng như CONTEXT, giữ nguyên đơn vị nếu có).
- Thứ tự đúng như bảng.
- Không diễn giải.

Câu cuối:
Đây là toàn bộ nhóm dẫn đầu theo {factor} của ngành {industry}.
"""
    return call_llm(prompt)


# ===================== FINANCIAL REPORT =====================

def ask_llm_financial(context, ticker, year):
    prompt = f"""
Bạn là chuyên gia phân tích tài chính doanh nghiệp.

{context}

Sau khi hiển thị nguyên văn các bảng, hãy phân tích:
- Cấu trúc tài sản và nguồn vốn
- Hiệu quả sinh lời
- Chất lượng dòng tiền
- Đánh giá sức khỏe tài chính tổng thể
- Thêm đoạn cuối: Lưu ý: đây là phân tích về báo cáo tài chính của doanh nghiệp dựa trên dữ liệu năm gần nhất có thể lấy từ nguồn vnstock, mong quý khách thông cảm.

Không bịa số, không suy diễn.
"""
    return call_llm(prompt)




