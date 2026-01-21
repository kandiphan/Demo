from intent_router import detect_intent
from retriever import (
    retrieve_stock_groups,
    get_comparison,
    get_ranking,
    get_latest_financials,
    ALL_INDUSTRIES_LOWER
)

from testgenerator import (
    build_recommend_context, ask_llm_recommend,
    build_comparison_context, ask_llm_comparison,
    build_ranking_context, ask_llm_ranking,
    ask_llm_financial
)
import re


# ===================== INDUSTRY =====================

def extract_industry(query_lower):
    for ind in ALL_INDUSTRIES_LOWER:
        if ind in query_lower:
            return ind

    keyword_map = {
        "bia": "bia và đồ uống",
        "đồ uống": "bia và đồ uống",
        "ngân hàng": "ngân hàng",
        "bank": "ngân hàng",
        "y tế": "thiết bị và dịch vụ y tế",
        "dược": "dược phẩm",
        "bệnh viện": "thiết bị và dịch vụ y tế",
        "bán lẻ": "bán lẻ",
        "bảo hiểm nhân thọ": "bảo hiểm nhân thọ",
        "nhân thọ": "bảo hiểm nhân thọ",
        "bảo hiểm": "bảo hiểm phi nhân thọ",
        "chứng khoán": "dịch vụ tài chính",
        "tài chính": "dịch vụ tài chính",
        "xây dựng": "xây dựng và vật liệu",
        "vật liệu": "xây dựng và vật liệu",
        "điện": "sản xuất & phân phối điện",
        "dầu khí": "sản xuất dầu khí",
        "viễn thông": "viễn thông di động",
        "thép": "kim loại",
        "kim loại": "kim loại",
        "hóa chất": "hóa chất",
        "khai khoáng": "khai khoáng",
        "vận tải": "vận tải",
        "ô tô": "ô tô và phụ tùng",
        "xe": "ô tô và phụ tùng",
        "phần mềm": "phần mềm & dịch vụ máy tính",
        "cntt": "phần mềm & dịch vụ máy tính",
        "thực phẩm": "sản xuất thực phẩm",
        "truyền thông": "truyền thông",
        "thuốc lá": "thuốc lá",
        "điện tử": "điện tử & thiết bị điện",
    }

    for k, v in keyword_map.items():
        if k in query_lower:
            return v

    return None

# ===================== FACTOR =====================

FACTOR_MAP = {
    "roa": "ROA",
    "roe": "ROE",
    "de": "DE",
    "d/e": "DE",
    "pb": "PB",
    "p/b": "PB",
    "eps": "EPS",
    "bv": "BV",
    "giá trị sổ sách": "BV",
    "vốn hóa": "Market_Cap",
    "market cap": "Market_Cap",
    "lợi nhuận trên cổ phiếu": "EPS",
    "score": "Score"
}

def extract_factor(query_lower):
    for k, v in FACTOR_MAP.items():
        if k in query_lower:
            return v
    return None

def extract_top_n(query_lower, default=5):
    m = re.search(r"top\s*(\d+)", query_lower)
    if m:
        return int(m.group(1))
    return default

# ===================== MAIN ENGINE =====================

def answer(query: str):
    intent = detect_intent(query)
    ql = query.lower()

    if intent == "out_of_domain":
        return "Hệ thống chỉ hỗ trợ các câu hỏi về tài chính, cổ phiếu và báo cáo doanh nghiệp."

    # ===== CASE 1: RECOMMENDATION =====
    if intent == "recommendation":
        industry = extract_industry(ql)
        if not industry:
            return "Không xác định được ngành."

        groups = retrieve_stock_groups(industry)
        if not groups:
            return "Không có dữ liệu cho ngành này."

        context = build_recommend_context(groups, industry)
        return ask_llm_recommend(context, industry)

    # ===== CASE 2: COMPARISON =====
    if intent == "comparison":
        tickers = re.findall(r"\b[A-Z]{2,5}\b", query.upper())
        table = get_comparison(tickers)

        table = table.rename(columns={
            "roa": "ROA", "ROA_mean": "ROA", "return_on_assets": "ROA",
            "de": "DE", "de_ratio": "DE",
            "pb": "PB", "p_b": "PB",
            "bv": "BV", "book_value": "BV",
            "market_cap": "Market_Cap", "MarketCap": "Market_Cap",
        })

        context = build_comparison_context(table)
        return ask_llm_comparison(context, table)

    # ===== CASE 3: RANKING =====
    if intent == "ranking":
        industry = extract_industry(ql)
        if not industry:
            return "Không xác định được ngành."

        factor = extract_factor(ql)
        if not factor:
            return "Không xác định được chỉ số (ROA, ROE, D/E, P/B, EPS, BV, Market Cap, Score)."

        top_n = extract_top_n(ql, default=5)

        table = get_ranking(industry, factor, top_n=top_n, ascending=False)
        if table is None or table.empty:
            return f"Ngành {industry} không có dữ liệu cho chỉ số {factor}."

        context = build_ranking_context(table, industry, factor)
        return ask_llm_ranking(context, industry, factor)

    # ===== CASE 4: FINANCIAL STATEMENT =====
    if intent == "financial_statement":
        ticker = re.findall(r"\b[A-Z]{2,5}\b", query.upper())[-1]

        fin_data = get_latest_financials(ticker)
        if not fin_data:
            return "Không lấy được báo cáo tài chính cho mã này."

    latest_year = fin_data["year"].max()
    return ask_llm_financial(fin_data, ticker, latest_year)


        

