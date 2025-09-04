import streamlit as st
import pandas as pd
import datetime

# Tiêu đề app
st.set_page_config(page_title="AI Crypto Trading", layout="wide")
st.title("📊 AI Crypto Trading Dashboard")

# Tạo bảng giả lập dữ liệu
if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame(columns=["Thời gian", "Coin", "Lệnh", "Giá vào", "Giá hiện tại", "P/L (%)", "Ghi chú"])

# Nhập lệnh mới
with st.form("new_trade_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        coin = st.text_input("Coin", "BTCUSDT")
    with col2:
        action = st.selectbox("Lệnh", ["Long", "Short"])
    with col3:
        entry_price = st.number_input("Giá vào", min_value=0.0, step=0.1)

    note = st.text_input("Ghi chú", "")
    submitted = st.form_submit_button("➕ Thêm lệnh")

    if submitted:
        new_trade = {
            "Thời gian": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Coin": coin,
            "Lệnh": action,
            "Giá vào": entry_price,
            "Giá hiện tại": entry_price,
            "P/L (%)": 0.0,
            "Ghi chú": note
        }
        st.session_state.trades = pd.concat([st.session_state.trades, pd.DataFrame([new_trade])], ignore_index=True)

# Cập nhật giá
if not st.session_state.trades.empty:
    st.write("### 📌 Danh sách lệnh")
    st.dataframe(st.session_state.trades, use_container_width=True)

    # Giả lập cập nhật P/L
    update_btn = st.button("🔄 Cập nhật giá (giả lập)")
    if update_btn:
        st.session_state.trades["Giá hiện tại"] = st.session_state.trades["Giá vào"] * (1 + (0.01))
        st.session_state.trades["P/L (%)"] = ((st.session_state.trades["Giá hiện tại"] - st.session_state.trades["Giá vào"]) / st.session_state.trades["Giá vào"]) * 100
        st.success("✅ Đã cập nhật giá (tăng 1%)")

    # Xuất Excel
    excel_btn = st.download_button(
        label="📥 Tải Excel",
        data=st.session_state.trades.to_csv(index=False).encode("utf-8"),
        file_name="trade_log.csv",
        mime="text/csv"
    )
