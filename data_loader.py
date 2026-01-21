from vnstock import Quote
import pandas as pd


def download_price(symbol, start="2020-01-01", end=None):
    """
    Tải dữ liệu giá đóng cửa của 1 cổ phiếu
    Trả về DataFrame:
        - index: thời gian
        - 1 cột: giá đóng cửa (tên cột = mã cổ phiếu)
    """

    try:
        quote = Quote(symbol=symbol, source="VCI")

        df = quote.history(
            start=start,
            end=end,
            interval="d"
        )

        # Kiểm tra dữ liệu
        if df is None or df.empty:
            return None

        # Chỉ giữ cột cần thiết
        df = df[["time", "close"]].copy()

        # Chuẩn hóa thời gian
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

        # Đổi tên cột thành mã cổ phiếu
        df = df.rename(columns={"close": symbol})

        return df

    except Exception as e:
        print(f"Lỗi tải dữ liệu {symbol}: {e}")
        return None


def download_multiple_prices(symbols, start="2020-01-01", end=None):
    """
    Tải dữ liệu giá cho nhiều cổ phiếu
    Trả về DataFrame giá (KHÔNG dropna)
    """

    price_dfs = []

    for sym in symbols:
        df = download_price(sym, start=start, end=end)
        if df is not None:
            price_dfs.append(df)

    # Không tải được mã nào
    if len(price_dfs) == 0:
        return None

    # Ghép dữ liệu theo thời gian (outer join)
    prices = pd.concat(price_dfs, axis=1, join="outer")

    # Sắp xếp theo thời gian
    prices = prices.sort_index()

    return prices
def download_market_index(start="2020-01-01", end=None):
    """
    Tải dữ liệu VNINDEX (market)
    """
    return download_price("VNINDEX", start=start, end=end)
