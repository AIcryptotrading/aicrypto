import streamlit as st
import pandas as pd
import ccxt
import datetime

# ==============================
# Khởi tạo Binance Spot
# ==============================
binance = ccxt.binance()
binance.load_markets()

# ==============================
# Hàm lấy giá từ Binance
# ==============================
def get_price(symbol):
    try:
        ticker = binance.fetch_ticker(symbol.replace("USDT", "/USDT"))
        return float(ticker['last'])
    except Exception:
        return None

# ==============================
# Hàm lấy dữ liệu OHLCV (để vẽ chart/trendline)
# ==============================
def get_ohlcv(symbol, timeframe="4h", limit=200):
    try:
        ohlcv = binance.fetch_ohlcv(symbol.replace("USDT", "/USDT"), timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception:
        return None

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="AI Crypto Trading Dashboard", layout="wide")
st.title("📊 AI Crypto Trading Dashboard (Paper-trade)")

# Sidebar settings
st.sidebar.header("Settings / Controls")
symbols_input = st.sidebar.text_area(
    "Symbols (comma separated)",
    "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT,TIAUSDT,KAVAUSDT,RENDERUSDT,AVAXUSDT,NEARUSDT,SEIUSDT"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval = st.sidebar.selectbox("Chart interval (for detail)", ["1h", "4h", "1d"])
auto_scan = st.sidebar.checkbox("Auto-scan market for setups", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", 60, 3600, 900)
max_suggestions = st.sidebar.number_input("Max suggestions per scan", 1, 20, 6)
sl_max = st.sidebar.number_input("SL max %", 1.0, 20.0, 5.0)
tp_min = st.sidebar.number_input("TP min %", 1.0, 50.0, 10.0)

# Hiển thị thời gian VN
vn_time = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
st.sidebar.markdown(f"**Time (VN):** {vn_time.strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================
# Market Watch
# ==============================
market_data = []
for sym in symbols:
    price = get_price(sym)
    market_data.append({"Symbol": sym, "Price": price if price else "err"})

df_market = pd.DataFrame(market_data)
st.subheader("Market Watch")
st.table(df_market)

# ==============================
# Chart & Trendline
# ==============================
st.markdown("---")
st.subheader("Chart & Trendline")

selected_symbol = st.selectbox("Select symbol", symbols)
df_chart = get_ohlcv(selected_symbol, timeframe=interval, limit=200)

if df_chart is not None and not df_chart.empty:
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_chart["time"],
                open=df_chart["open"],
                high=df_chart["high"],
                low=df_chart["low"],
                close=df_chart["close"],
                name=selected_symbol,
            )
        ]
    )
    fig.update_layout(
        title=f"{selected_symbol} {interval} Chart",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        autosize=True,
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"❌ Cannot fetch data for {selected_symbol}")

# ==============================
# Scan / Actions
# ==============================
st.markdown("---")
st.subheader("Scan / Actions")

if st.button("Manual Scan now"):
    st.success("✅ Scan complete (placeholder logic).")
