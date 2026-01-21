import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
PRICE_DIR = os.path.join(BASE_DIR, "price_offline")


def load_price(symbol):
    """
    Đọc dữ liệu giá đóng cửa từ file CSV offline
    Trả về DataFrame:
        - index: thời gian
        - 1 cột: giá đóng cửa (tên cột = mã cổ phiếu)
    """
    file_path = os.path.join(PRICE_DIR, f"{symbol}_close_2022_now.csv")

    if not os.path.exists(file_path):
        print(f"Không tìm thấy file giá của {symbol}")
        return None

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df = df.rename(columns={"close_price": symbol})

    return df[[symbol]]


def load_multiple_prices(symbols):
    """
    Load dữ liệu giá cho nhiều cổ phiếu (outer join)
    """
    price_dfs = []

    for sym in symbols:
        df = load_price(sym)
        if df is not None:
            price_dfs.append(df)

    if len(price_dfs) == 0:
        return None

    prices = pd.concat(price_dfs, axis=1, join="outer")
    prices = prices.sort_index()

    return prices


def load_market_index():
    """
    Nếu có file VNINDEX_close_2022_now.csv thì dùng, 
    không thì tạm thời return None
    """
    return load_price("VNINDEX")
