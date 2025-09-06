import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
import talib
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
import os

# ================== CONFIG ==================
st.set_page_config(page_title="AI Crypto Trading Pro", layout="wide")

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = None
if API_KEY and API_SECRET:
    client = Client(API_KEY, API_SECRET)

BASE_URL = "https://api.binance.com"

# ================== FUNCTIONS ==================
def get_symbols():
    url = BASE_URL + "/api/v3/exchangeInfo"
    data = requests.get(url).json()
    return [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT']

def get_klines(symbol, interval="4h", limit=200):
    url = BASE_URL + f"/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "time","o","h","l","c","v","close_time","qv","trades","tb_base","tb_quote","ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
    return df

def detect_trendline_breakout(df):
    closes = df["c"].values
    highs = df["h"].values
    lows = df["l"].values
    
    # Trendline (simple: connect highest + lowest)
    max_idx = np.argmax(highs)
    min_idx = np.argmin(lows)
    x = np.array([min_idx, max_idx])
    y = np.array([lows[min_idx], highs[max_idx]])
    
    slope = (y[1]-y[0])/(x[1]-x[0]+1e-9)
    trendline = y[0] + slope*(len(closes)-1 - x[0])
    
    breakout = closes[-1] > trendline
    return breakout, (x, y, trendline)

def generate_signal(df):
    close = df["c"].values
    rsi = talib.RSI(close, timeperiod=14)
    ma50 = talib.SMA(close, timeperiod=50)
    ma200 = talib.SMA(close, timeperiod=200)
    
    breakout, trendline_info = detect_trendline_breakout(df)
    signals = []
    
    if breakout:
        signals.append("Breakout trendline ↑")
    if rsi[-1] < 30:
        signals.append("RSI oversold (BUY)")
    if rsi[-1] > 70:
        signals.append("RSI overbought (SELL)")
    if ma50[-1] > ma200[-1]:
        signals.append("Golden Cross (Bullish)")
    if ma50[-1] < ma200[-1]:
        signals.append("Death Cross (Bearish)")
    
    return signals, trendline_info

def place_order(symbol, side, qty, test=True):
    if not client:
        return {"status": "error", "msg": "No API key provided. Running paper trade."}
    try:
        if test:
            return {"status": "success", "msg": f"Paper order: {side} {qty} {symbol}"}
        else:
            order = client.create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty
            )
            return {"status": "success", "msg": str(order)}
    except BinanceAPIException as e:
        return {"status": "error", "msg": str(e)}

# ================== UI ==================
st.title("🤖 AI Crypto Trading — Professional Edition")

# Sidebar
st.sidebar.header("⚙️ Settings")
symbols = st.sidebar.text_input("Symbols (comma)", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT").split(",")
interval = st.sidebar.selectbox("Interval", ["15m","1h","4h","1d"], index=2)
auto_scan = st.sidebar.checkbox("Enable auto-scan", value=True)
paper_trade = st.sidebar.checkbox("Paper-trade mode", value=True)
telegram_alert = st.sidebar.text_input("Telegram Bot Token + Chat ID (opt)", "")

# Market Watch
st.subheader("📊 Market Watch")
cols = st.columns(len(symbols))
for i, sym in enumerate(symbols):
    try:
        url = BASE_URL + f"/api/v3/ticker/price?symbol={sym.strip()}"
        price = float(requests.get(url).json()["price"])
        cols[i].metric(sym.strip(), price)
    except:
        cols[i].write("Error")

# Scan
st.subheader("🔍 Signal Scanner")
run = st.button("Run Scan Now")

if run or auto_scan:
    for sym in symbols:
        sym = sym.strip()
        try:
            df = get_klines(sym, interval=interval, limit=200)
            signals, trendline_info = generate_signal(df)
            
            if signals:
                st.write(f"### {sym} — Signals")
                st.write(", ".join(signals))
                
                # Chart
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df["time"], df["c"], label="Close")
                ax.plot(df["time"], talib.SMA(df["c"], timeperiod=50), label="MA50")
                ax.plot(df["time"], talib.SMA(df["c"], timeperiod=200), label="MA200")
                ax.legend()
                st.pyplot(fig)
                
                # Auto-trade
                if any("BUY" in s or "Bullish" in s or "Breakout" in s for s in signals):
                    res = place_order(sym, "BUY", qty=0.01, test=paper_trade)
                    st.info(res["msg"])
                elif any("SELL" in s or "Bearish" in s for s in signals):
                    res = place_order(sym, "SELL", qty=0.01, test=paper_trade)
                    st.warning(res["msg"])
        except Exception as e:
            st.error(f"{sym} — {e}")
