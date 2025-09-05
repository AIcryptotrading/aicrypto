import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# =========================
# Binance REST API (spot)
# =========================
BASE_URL = "https://api.binance.com"

def get_price(symbol):
    url = f"{BASE_URL}/api/v3/ticker/price"
    params = {"symbol": symbol}
    r = requests.get(url, params=params).json()
    if "price" in r:
        return float(r["price"])
    else:
        print(f"⚠️ Error get_price for {symbol}: {r}")
        return None

def get_klines(symbol, interval="4h", limit=200):
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params).json()
    if isinstance(r, list):
        df = pd.DataFrame(r, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    else:
        print(f"⚠️ Error get_klines for {symbol}: {r}")
        return pd.DataFrame()

# =========================
# Trading Logic
# =========================
def detect_signal(df):
    """Cơ bản: breakout + MA trend"""
    if df.empty: 
        return None
    
    df["MA20"] = df["close"].rolling(20).mean()
    last_close = df["close"].iloc[-1]
    last_ma = df["MA20"].iloc[-1]

    if last_close > last_ma:
        return "BUY"
    elif last_close < last_ma:
        return "SELL"
    return None

# =========================
# Paper Trade
# =========================
portfolio = {"USDT": 10000}
open_orders = []

def place_order(symbol, side, price, qty):
    order = {
        "symbol": symbol,
        "side": side,
        "price": price,
        "qty": qty,
        "time": datetime.utcnow()
    }
    open_orders.append(order)
    return order

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="AI Crypto Trader", layout="wide")
st.title("🤖 AI Crypto Trader (Binance Realtime)")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
symbols = st.sidebar.text_input("Symbols (comma separated)", "BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT")
symbols = [s.strip().upper() for s in symbols.split(",")]
interval = st.sidebar.selectbox("Chart interval", ["1m","5m","15m","1h","4h","1d"], index=4)
auto_scan = st.sidebar.checkbox("Auto-scan", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", 60, 3600, 300)

# Market Watch
st.subheader("📊 Market Watch")
cols = st.columns(5)
for i, sym in enumerate(symbols):
    price = get_price(sym)
    if price:
        cols[i % 5].metric(sym, f"{price:,.2f} USDT")
    else:
        cols[i % 5].metric(sym, "N/A")

# Scan Suggestions
st.subheader("🔍 Trade Suggestions")
suggestions = []
for sym in symbols:
    df = get_klines(sym, interval=interval, limit=100)
    sig = detect_signal(df)
    if sig:
        suggestions.append({"symbol": sym, "signal": sig, "price": df["close"].iloc[-1]})

if suggestions:
    st.table(pd.DataFrame(suggestions))
else:
    st.info("No trade signals found.")

# Chart
st.subheader("📈 Chart & Analysis")
chart_symbol = st.selectbox("Choose symbol", symbols, index=0)
df = get_klines(chart_symbol, interval=interval, limit=150)

if not df.empty:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], 
        low=df["low"], close=df["close"], name="Candles"
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["close"].rolling(20).mean(), 
                             line=dict(color="blue", width=1.5), name="MA20"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("⚠️ Cannot fetch chart data.")

# Paper Orders
st.subheader("📒 Paper Orders")
if st.button("Place Test Order"):
    price = get_price(chart_symbol)
    if price:
        order = place_order(chart_symbol, "BUY", price, qty=0.01)
        st.success(f"Order placed: {order}")

if open_orders:
    st.table(pd.DataFrame(open_orders))
else:
    st.info("No paper orders yet.")

# Reports
st.subheader("📑 Reports")
st.write("Daily PnL report will be generated here (placeholder).")
