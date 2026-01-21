import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
PRICE_DIR = os.path.join(BASE_DIR, "priceoffline")


def download_price(symbol, start="2020-01-01", end=None):
    file_path = os.path.join(PRICE_DIR, f"{symbol}_close_2022_now.csv")

    if not os.path.exists(file_path):
        print(f"Không tìm thấy file giá của {symbol}")
        return None

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # LỌC THEO THỜI GIAN (GIỮ GIỐNG HÀM ONLINE)
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]

    df = df.rename(columns={"close_price": symbol})

    if df.empty:
        return None

    return df[[symbol]]


def download_multiple_prices(symbols, start="2020-01-01", end=None):
    """
    Load dữ liệu giá cho nhiều cổ phiếu (outer join)
    Giữ nguyên interface như bản dùng vnstock
    """
    price_dfs = []

    for sym in symbols:
        df = download_price(sym, start=start, end=end)
        if df is not None:
            price_dfs.append(df)

    if len(price_dfs) == 0:
        return None

    prices = pd.concat(price_dfs, axis=1, join="outer")
    prices = prices.sort_index()

    return prices


def download_market_index(start="2020-01-01", end=None):
    """
    Load VNINDEX offline (nếu có file)
    """
    return download_price("VNINDEX", start=start, end=end)
