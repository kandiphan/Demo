# app.py
import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import matplotlib.pyplot as plt


from data_loader import (
    download_multiple_prices,
    download_market_index
)

from preprocessing import (
    calculate_log_returns,
    estimate_betas,
    estimate_market_parameters,
    capm_expected_returns,
    capm_covariance_matrix
)

from optimizer import optimize_capm_portfolio

import subprocess
import os

ML_SCRIPT = "testml.py"
ML_OUTPUT = "ket_qua_ranking_co_phieu_ml_industry_scaled.csv"

# Chỉ chạy ML nếu file kết quả chưa tồn tại
if not os.path.exists(ML_OUTPUT):
    with st.spinner("Đang huấn luyện mô hình ML và tạo bảng xếp hạng..."):
        subprocess.run(["python", ML_SCRIPT], check=True)

from testengine import answer 
# ===================== CONFIG =====================
st.set_page_config(
    page_title="CAPM Portfolio Optimization (Excel-style)",
    layout="centered"
)

st.title("📊 CAPM Portfolio Optimization – Excel Solver Logic")


# ===================== INPUT =====================
symbols_input = st.text_input(
    "Nhập mã cổ phiếu (cách nhau bởi dấu phẩy)",
    "VNM, FPT, HPG"
)

start_date = st.date_input(
    "Ngày bắt đầu",
    value=pd.to_datetime("2020-01-01")
)

rf = st.number_input(
    "Risk-free rate (rf – theo NĂM)",
    value=0.04,
    step=0.005,
    format="%.3f"
)


# ===================== RUN =====================
if st.button("Tối ưu danh mục"):

    symbols = [s.strip().upper() for s in symbols_input.split(",")]

    if len(symbols) < 2:
        st.warning("Cần ít nhất 2 cổ phiếu")
        st.stop()

    # ===================== LOAD DATA =====================
    with st.spinner("Đang tải dữ liệu giá..."):
        prices = download_multiple_prices(
            symbols,
            start=start_date.strftime("%Y-%m-%d")
        )

        market_price = download_market_index(
            start=start_date.strftime("%Y-%m-%d")
        )

    if prices is None or market_price is None:
        st.error("Không tải được dữ liệu")
        st.stop()

    # ===================== LOG RETURNS =====================
    stock_log_returns = calculate_log_returns(prices)
    market_log_returns = calculate_log_returns(market_price)["VNINDEX"]

    # Đồng bộ thời gian (GIỐNG EXCEL)
    data = stock_log_returns.join(market_log_returns, how="inner")
    # ÉP lại đúng khoảng thời gian user chọn
    data = data[data.index >= pd.to_datetime(start_date)]
    stock_log_returns = data[symbols]
    market_log_returns = data["VNINDEX"]

    # ===================== CAPM PARAMETERS =====================
    betas = estimate_betas(stock_log_returns, market_log_returns)

    expected_rm, market_variance = estimate_market_parameters(
        market_log_returns
    )

    # 👉 Market premium (THIẾU DÒNG NÀY TRƯỚC ĐÂY)
    market_premium = expected_rm - rf

    expected_returns = capm_expected_returns(
        betas=betas,
        expected_rm=expected_rm,
        rf=rf
    )

    cov_capm = stock_log_returns.cov() * 252

    st.write("Expected returns:", expected_returns)
    st.write("Cov diag:", np.diag(cov_capm))

    # ===================== OPTIMIZATION (EXCEL SOLVER) =====================
    with st.spinner("Đang tối ưu danh mục (Excel Solver logic)..."):
      weights = optimize_capm_portfolio(
            expected_returns=expected_returns,
            cov=cov_capm,
            rf=rf
)
    # Đặt index cho weights (để hiển thị đẹp)
    weights.index = betas.index

   # ===================== OUTPUT =====================
    st.subheader("📊 Tỷ trọng tối ưu (Max Sharpe – CAPM)")

    st.dataframe(weights.rename("Weight"))

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(weights.values, labels=weights.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")  # Đảm bảo hình tròn
    st.pyplot(fig)

    # ===================== PORTFOLIO METRICS =====================
    port_return = np.dot(weights.values, expected_returns.values)
    port_variance = weights.values @ cov_capm.values @ weights.values
    port_vol = np.sqrt(port_variance)

    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0

    st.markdown("### 📈 Chỉ số danh mục")
    st.write(f"📌 Expected Return: **{port_return:.4f}**")
    st.write(f"📌 Std Dev σp: **{port_vol:.4f}**")
    st.write(f"📌 Sharpe Ratio: **{sharpe:.4f}**")

    # ===================== CAPM TABLE =====================
    st.markdown("### 📉 Tham số CAPM")
    st.dataframe(
        pd.DataFrame({
            "Beta": betas,
            "Expected Return": expected_returns
        })
    )

    st.write("Số quan sát:", len(stock_log_returns))
    st.write(
        "Thời gian:",
        stock_log_returns.index.min(),
        "→",
        stock_log_returns.index.max()
    )
# =========================
# ==============SIDEBAR===========
with st.sidebar:
    st.markdown("## 🤖 Trợ lý phân tích cổ phiếu")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Chào bạn 👋\n\n"
                    "Dựa trên mô hình định lượng và dữ liệu đã huấn luyện, tôi chỉ có thể trả lời các câu hỏi liên quan đến: "
                    "ROA, ROE, P/B, D/E, EPS, xếp hạng cổ phiếu theo ngành, so sánh cổ phiếu, "
                    "và phân tích thông tin báo cáo tài chính.\n\n"
                    "Ví dụ câu hỏi:\n"
                    "- Cổ phiếu bất động sản nào đáng để đầu tư?\n"
                    "- So sánh VCB với TCB\n"
                    "- Top 10 cổ phiếu vốn hóa cao nhất ngành kim loại\n"
                    "- Phân tích báo cáo tài chính FPT"
                )
            }
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Nhập câu hỏi về cổ phiếu...")

    from queue import Queue

    if user_input:
        # 1. Lưu và hiển thị ngay câu hỏi người dùng
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input)

        # 2. Tạo bong bóng bot với trạng thái đang suy luận
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("⏳ *Đang suy luận...*")

        def bot_task():
            try:
                return answer(user_input)
            except Exception as e:
                return f"Lỗi hệ thống: {e}"

        from queue import Queue
        q = Queue()

        def run():
            q.put(bot_task())

        t = threading.Thread(target=run)
        t.start()

        # 3. Chờ bot xử lý
        while t.is_alive():
            typing_placeholder.markdown("⏳ *Đang suy luận...*")
            time.sleep(0.2)

        final_answer = q.get()

        # 4. Hiệu ứng gõ chữ
        typed = ""
        for ch in final_answer:
            typed += ch
            typing_placeholder.markdown(typed)
            time.sleep(0.01)

        # 5. Lưu lịch sử
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": final_answer}
        )

        st.rerun()



