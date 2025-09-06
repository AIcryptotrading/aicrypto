import streamlit as st
import matplotlib.pyplot as plt
from utils import get_klines, compute_indicators
from ai_model import train_ai_model, load_ai_model, predict_signal
from trade_manager import place_order, get_orders
from telegram import send_telegram_message

st.set_page_config(page_title="AI Crypto Trading Pro", layout="wide")
st.title("🤖 AI Crypto Trading — Professional Edition")

symbols = st.sidebar.text_input("Symbols", "BTCUSDT,ETHUSDT").split(",")
interval = st.sidebar.selectbox("Interval", ["15m","1h","4h","1d"], index=2)
train_ai = st.sidebar.button("Train AI Model")
tg_bot = st.sidebar.text_input("Telegram Bot Token", "")
tg_chat = st.sidebar.text_input("Telegram Chat ID", "")

if train_ai:
    for sym in symbols:
        df = get_klines(sym.strip(), interval=interval, limit=500)
        df = compute_indicators(df)
        train_ai_model(df)
    st.success("✅ AI Model trained!")

model = load_ai_model()

if st.button("Scan Market"):
    for sym in symbols:
        df = get_klines(sym.strip(), interval=interval, limit=200)
        df = compute_indicators(df)

        if model:
            signal = predict_signal(df, model)
            st.write(f"### {sym}: AI Suggestion → {signal}")
            place_order(sym, signal, df['c'].iloc[-1])

            # Vẽ chart
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(df["time"], df["c"], label="Close")
            ax.plot(df["time"], df["ma50"], label="MA50")
            ax.plot(df["time"], df["ma200"], label="MA200")
            ax.legend()
            st.pyplot(fig)

            # Gửi Telegram
            if tg_bot and tg_chat:
                send_telegram_message(tg_bot, tg_chat, f"{sym}: {signal}")

st.subheader("📑 Orders")
st.dataframe(get_orders())
