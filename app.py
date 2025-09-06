# app.py
import streamlit as st
from utils import fetch_klines, add_indicators, last_price
from ai_model import train_supervised, load_supervised, train_rl, load_rl
from trade_manager import place_paper_order, list_paper_orders
from telegram import send_telegram
import matplotlib.pyplot as plt
import pandas as pd
import os

st.set_page_config(page_title="AI Crypto Trading — Pro", layout="wide")
st.title("🤖 AI Crypto Trading — Pro (Paper-trade)")

# Sidebar - settings
st.sidebar.header("Settings")
symbols_txt = st.sidebar.text_input("Symbols (comma)", "BTCUSDT,ETHUSDT,SOLUSDT")
interval = st.sidebar.selectbox("Interval", ["15m","1h","4h","1d"], index=2)
auto_scan = st.sidebar.checkbox("Auto-scan", value=False)
paper_trade = st.sidebar.checkbox("Paper-trade (default)", value=True)
tg_token = st.sidebar.text_input("Telegram token", "")
tg_chat = st.sidebar.text_input("Telegram chat id", "")

symbols = [s.strip().upper() for s in symbols_txt.split(",") if s.strip()]

col1, col2 = st.columns([2,1])
with col2:
    st.subheader("Models")
    if st.button("Train supervised model (quick)"):
        # train on first symbol
        s = symbols[0]
        df = fetch_klines(s, interval=interval, limit=1000)
        df = add_indicators(df)
        train_supervised(df)
        st.success("Supervised model trained & saved.")
    if st.button("Train RL (PPO) quick (local)"):
        s = symbols[0]
        df = fetch_klines(s, interval=interval, limit=1000)
        df = add_indicators(df)
        st.info("Training RL, this may take long. Use local/colab with GPU for faster.")
        model = train_rl(df, timesteps=5000)
        st.success("RL model trained.")

with col1:
    st.subheader("Market Watch")
    # Show current price metrics
    cols = st.columns(len(symbols))
    prices = {}
    for i,sym in enumerate(symbols):
        try:
            p = last_price(sym)
            prices[sym] = p
            cols[i].metric(sym, p)
        except Exception as e:
            cols[i].write("err")

st.markdown("---")
st.subheader("Scanner & AI suggestion")

if st.button("Run scan now") or auto_scan:
    sup = load_supervised()
    rl = load_rl()
    suggestions = []
    for sym in symbols:
        try:
            df = fetch_klines(sym, interval=interval, limit=500)
            df = add_indicators(df)
            # supervised inference
            if sup:
                feat = df[["rsi","ema20","ema50","atr"]].fillna(0).values
                pred = sup.predict(feat[-1].reshape(1,-1))[0]
                suggestion = "BUY" if pred==1 else "HOLD/SELL"
            else:
                suggestion = "Model not found"
            st.write(f"**{sym}** → {suggestion}")
            # chart
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(df.index[-200:], df["close"].values[-200:], label="close")
            ax.plot(df.index[-200:], df["ema20"].values[-200:], label="ema20")
            ax.plot(df.index[-200:], df["ema50"].values[-200:], label="ema50")
            ax.legend()
            st.pyplot(fig)
            # optional RL policy action
            if rl:
                obs_env = None
                # we need a proper env, skipping here; real env predict uses low-level API
            # auto place paper order
            if suggestion == "BUY" and paper_trade:
                price = df["close"].iloc[-1]
                ord = place_paper_order(sym, "BUY", price, size=1.0, note="AI sup suggestion")
                st.info(f"Placed paper order: {ord}")
                if tg_token and tg_chat:
                    send_telegram(tg_token, tg_chat, f"Paper BUY {sym} @ {price}")
            suggestions.append((sym, suggestion))
        except Exception as e:
            st.error(f"{sym} error: {e}")

st.markdown("---")
st.subheader("Paper Orders")
ords = list_paper_orders()
if ords:
    st.table(pd.DataFrame(ords))
else:
    st.info("No paper orders yet.")
