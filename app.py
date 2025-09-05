import requests
import pandas as pd
import numpy as np
import datetime
import time
import streamlit as st
import plotly.graph_objects as go

# ============ CONFIG ============
BASE_URL = "https://api.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]  # Top coin
INTERVAL = "15m"
LIMIT = 200
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10

# Giả lập danh sách lệnh
open_trades = []
closed_trades = []

# ============ API ===============
def get_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume","close_time","qav","num_trades",
        "taker_base_vol","taker_quote_vol","ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df

def get_price(symbol):
    url = f"{BASE_URL}/api/v3/ticker/price"
    params = {"symbol": symbol}
    r = requests.get(url, params=params).json()
    return float(r["price"])

# ============ ANALYSIS ==========
def detect_trend_break(df):
    """
    Rất đơn giản: nếu close vượt MA20 => tín hiệu LONG
    Nếu close dưới MA20 => tín hiệu SHORT
    """
    df["MA20"] = df["close"].rolling(20).mean()
    last_close = df["close"].iloc[-1]
    last_ma = df["MA20"].iloc[-1]

    if last_close > last_ma:
        return "LONG"
    elif last_close < last_ma:
        return "SHORT"
    else:
        return None

def simulate_trade(symbol, signal, price):
    global open_trades, closed_trades
    trade = {
        "symbol": symbol,
        "signal": signal,
        "entry": price,
        "tp": price * (1 + TAKE_PROFIT_PCT if signal == "LONG" else 1 - TAKE_PROFIT_PCT),
        "sl": price * (1 - STOP_LOSS_PCT if signal == "LONG" else 1 + STOP_LOSS_PCT),
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "OPEN"
    }
    open_trades.append(trade)
    return trade

def update_trades():
    global open_trades, closed_trades
    updated = []
    for trade in open_trades:
        price = get_price(trade["symbol"])
        if trade["signal"] == "LONG":
            if price >= trade["tp"]:
                trade["status"] = "TP"
                trade["exit"] = price
                closed_trades.append(trade)
            elif price <= trade["sl"]:
                trade["status"] = "SL"
                trade["exit"] = price
                closed_trades.append(trade)
            else:
                updated.append(trade)
        elif trade["signal"] == "SHORT":
            if price <= trade["tp"]:
                trade["status"] = "TP"
                trade["exit"] = price
                closed_trades.append(trade)
            elif price >= trade["sl"]:
                trade["status"] = "SL"
                trade["exit"] = price
                closed_trades.append(trade)
            else:
                updated.append(trade)
    open_trades = updated

# ============ STREAMLIT =========
st.set_page_config(layout="wide")
st.title("📈 AI Crypto Trader (Binance Realtime)")

# Market Watch
cols = st.columns(len(SYMBOLS))
for i, sym in enumerate(SYMBOLS):
    price = get_price(sym)
    cols[i].metric(sym, f"{price:,.2f} USDT")

# Chart + Signal
selected = st.selectbox("Chọn coin để xem chart:", SYMBOLS)
df = get_klines(selected, INTERVAL, LIMIT)
signal = detect_trend_break(df)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="Price"
))
fig.add_trace(go.Scatter(x=df["time"], y=df["MA20"], mode="lines", name="MA20"))
st.plotly_chart(fig, use_container_width=True)

if signal:
    st.success(f"Tín hiệu hiện tại: {signal}")
    if st.button(f"Vào lệnh {signal} {selected}"):
        trade = simulate_trade(selected, signal, df["close"].iloc[-1])
        st.write("✅ Đã vào lệnh:", trade)

# Cập nhật trạng thái lệnh
update_trades()
st.subheader("Lệnh đang mở")
st.dataframe(pd.DataFrame(open_trades))
st.subheader("Lệnh đã đóng")
st.dataframe(pd.DataFrame(closed_trades))
