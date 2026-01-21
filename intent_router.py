import re

def detect_intent(query: str):
    q = query.lower()

    finance_keywords = [
        "cổ phiếu", "ngành", "ngân hàng", "bất động", "thép", "chứng khoán",
        "roa", "de", "d/e", "p/b", "pb", "doanh thu", "lợi nhuận",
        "báo cáo", "vốn hóa", "so sánh", "top", "xếp hạng",
        "chiến lược", "rủi ro", "chu kỳ"
    ]

    if not any(k in q for k in finance_keywords):
        return "out_of_domain"

    if any(x in q for x in ["so sánh", "vs", "với"]):
        return "comparison"

    if any(x in q for x in ["top", "xếp hạng", "cao nhất", "thấp nhất"]):
        return "ranking"

    if any(x in q for x in ["roa", "d/e", "de", "p/b", "pb", "đòn bẩy"]):
        return "ratio_analysis"

    if "báo cáo tài chính" in q or "bctc" in q:
        return "financial_statement"

    if any(x in q for x in ["báo cáo", "doanh thu", "chiến lược", "rủi ro"]):
        return "report_qa"


    if any(x in q for x in ["chu kỳ", "toàn ngành", "triển vọng"]):
        return "sector_analysis"

    return "recommendation"
