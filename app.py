import streamlit as st
import plotly.graph_objects as go
from utils import get_klines, compute_indicators
from telegram import send_telegram_message

st.set_page_config(page_title="AI Crypto Trading", layout="wide")

st.title("📈 AI Crypto Trading Dashboard")

symbol = st.sidebar.text_input("Symbol", "BTCUSDT")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","1h","4h","1d"], index=3)

# Lấy dữ liệu
df = get_klines(symbol=symbol, interval=interval)
df = compute_indicators(df)

# Hiển thị chart
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Candles"
))
fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], line=dict(color="blue"), name="EMA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], line=dict(color="red"), name="EMA50"))

st.plotly_chart(fig, use_container_width=True)

# Gửi tín hiệu AI đơn giản
last = df.iloc[-1]
if last["ema20"] > last["ema50"]:
    signal = f"✅ BUY Signal cho {symbol} (EMA20 > EMA50)"
else:
    signal = f"❌ SELL Signal cho {symbol} (EMA20 < EMA50)"

st.success(signal)
send_telegram_message(signal)
