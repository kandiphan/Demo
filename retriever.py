import pandas as pd
from pathlib import Path


# ===================== CONFIG =====================
CSV_PATH = "ket_qua_ranking_co_phieu_ml_industry_scaled.csv"
MODEL_NAME = "gemma2:2b"
TOP_SCORE_N = 5
TOP_MARKETCAP_N = 5

# ===================== LOAD DATA =====================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
ALL_INDUSTRIES = sorted(df["industry"].dropna().unique().tolist())
ALL_INDUSTRIES_LOWER = [x.lower() for x in ALL_INDUSTRIES]

df = df.rename(columns={
    "Score_industry": "Score",
    "Market_Cap": "Market_Cap",
    "Rank_in_Industry": "Rank_in_Industry"
})

df = df.dropna(subset=["Score", "Market_Cap"]).reset_index(drop=True)

# ===================== NORMALIZE INDUSTRY =====================
def normalize_industry(x):
    if not isinstance(x, str):
        return None
    return x.lower()

df["industry_norm"] = df["industry"].apply(normalize_industry)

# -------- Recommendation --------
TOP_SCORE_N = 5
TOP_MARKETCAP_N = 5

def retrieve_stock_groups(industry):
    data = df[df["industry_norm"] == industry].copy()
    if data.empty:
        return None

    top_score = data.sort_values("Score", ascending=False).head(TOP_SCORE_N)
    top_cap = data.sort_values("Market_Cap", ascending=False).head(TOP_MARKETCAP_N)

    balanced = top_score[top_score["ticker"].isin(top_cap["ticker"])]
    best_growth = top_score[~top_score["ticker"].isin(top_cap["ticker"])].head(1)
    best_safe = top_cap.sort_values("Score", ascending=False).head(1)
    best_balance = balanced.sort_values("Score", ascending=False).head(1)

    return {
        "top_score": top_score,
        "top_cap": top_cap,
        "best_growth": best_growth,
        "best_safe": best_safe,
        "best_balance": best_balance
    }


# -------- Comparison --------
def get_comparison(tickers):
    cols = ["ticker", "Score", "Market_Cap", "ROA", "DE", "BV", "PB"]
    out = df[df["ticker"].isin(tickers)][cols].copy()
    out["Market_Cap"] = out["Market_Cap"] / 1e9  # sang tỷ
    out["BV"] = out["BV"] / 1e9  # sang tỷ
    return out





# -------- Ranking --------
def get_ranking(industry, factor, top_n=10, ascending=False):
    data = df[df["industry_norm"] == industry].copy()
    if data.empty or factor not in data.columns:
        return None

    result = data.sort_values(factor, ascending=ascending).head(top_n)
    return result[["ticker", factor]]



# -------- PDF RAG (simple) --------
from vnstock import Vnstock

def get_latest_financials(ticker):
    vn = Vnstock()
    stock = vn.stock(symbol=ticker, source="VCI")

    bs = stock.finance.balance_sheet(period="year")
    is_ = stock.finance.income_statement(period="year")
    cf = stock.finance.cash_flow(period="year")

    if bs is None or is_ is None or cf is None:
        return None

    year = bs["yearReport"].max()

    return {
        "year": year,
        "balance_sheet": bs[bs["yearReport"] == year],
        "income_statement": is_[is_["yearReport"] == year],
        "cash_flow": cf[cf["yearReport"] == year]
    }
