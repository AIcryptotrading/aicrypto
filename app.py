import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# =========================
# Binance REST API functions
# =========================
BINANCE_API_URL = "https://api.binance.com/api/v3"

def get_price(symbol="BTCUSDT"):
    """Get latest price of a symbol"""
    url = f"{BINANCE_API_URL}/ticker/price?symbol={symbol}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data["price"])
    except Exception:
        return None

def get_klines(symbol="BTCUSDT", interval="4h", limit=200):
    """Get historical candlestick data"""
    url = f"{BINANCE_API_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception:
        return None

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="AI Crypto Trading Dashboard", layout="wide")

st.title("📊 AI Crypto Trading Dashboard (Paper-trade)")

# Sidebar controls
st.sidebar.header("Settings / Controls")

symbols = st.sidebar.text_area(
    "Symbols (comma separated)",
    "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT"
).replace(" ", "").split(",")

interval = st.sidebar.selectbox("Chart interval (for detail)", ["1m", "5m", "15m", "1h", "4h", "1d"], index=4)
auto_scan = st.sidebar.checkbox("Auto-scan market for setups", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", min_value=30, value=300, step=30)

sl_max = st.sidebar.number_input("SL max %", value=5.0)
tp_min = st.sidebar.number_input("TP min %", value=10.0)

st.sidebar.write("Time (VN):", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# =========================
# Market Watch
# =========================
st.subheader("Market Watch")

market_data = []
for sym in symbols:
    price = get_price(sym)
    market_data.append({"Symbol": sym, "Price": price if price else "err"})

df_prices = pd.DataFrame(market_data)
st.dataframe(df_prices, use_container_width=True)

# =========================
# Chart & Trendline
# =========================
st.subheader("Chart & Trendline")

selected_symbol = st.selectbox("Select symbol", symbols, index=0)

df = get_klines(selected_symbol, interval=interval, limit=200)
if df is None:
    st.error(f"Cannot fetch data for {selected_symbol}")
else:
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Candles"
    )])

    # Simple trendline (last 50 closes)
    closes = df["close"].tail(50)
    times = df["time"].tail(50)
    if len(closes) > 1:
        # Linear regression for trendline
        trend_df = pd.DataFrame({"x": range(len(closes)), "y": closes})
        coef = pd.Series(trend_df["y"]).rolling(window=2).mean()
        fig.add_trace(go.Scatter(
            x=times,
            y=coef,
            mode="lines",
            line=dict(color="red", width=2),
            name="Trendline"
        ))

    fig.update_layout(
        title=f"{selected_symbol} Price Chart ({interval})",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Auto refresh (if enabled)
# =========================
if auto_scan:
    st.experimental_autorefresh(interval=scan_interval * 1000, key="auto_scan")
