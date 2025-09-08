# app.py - Streamlit Paper Trading AI (safe)
import streamlit as st
import pandas as pd, numpy as np, time
from utils import fetch_ohlcv, add_indicators, last_price
from ai_model import train_quick, load_model, predict
from trade_manager import place_order, load_orders
from backtest import run_backtest
from telegram import send_telegram
st.set_page_config(layout='wide', page_title='AI Paper Trader')

st.title('🤖 AI Paper Trader — Paper Mode (Safe)')

# Sidebar / settings
st.sidebar.header('Settings')
symbols = st.sidebar.text_input('Symbols (comma)', 'BTCUSDT,ETHUSDT,BNBUSDT').split(',')
interval = st.sidebar.selectbox('Interval', ['1h','4h','1d'], index=1)
paper_balance = st.sidebar.number_input('Paper balance (USD)', value=1000.0)
risk_pct = st.sidebar.number_input('Risk per trade (%)', value=1.0, min_value=0.1, max_value=10.0)
max_open = st.sidebar.number_input('Max open orders', value=5, min_value=1, max_value=20)
tg_token = st.sidebar.text_input('Telegram bot token (opt)', value='')
tg_chat = st.sidebar.text_input('Telegram chat id (opt)', value='')

st.sidebar.markdown('---')
st.sidebar.subheader('AI Model')
if st.sidebar.button('Train quick supervised model'):
    with st.spinner('Training model...'):
        s = symbols[0].strip()
        df = fetch_ohlcv(s, interval=interval, limit=1000)
        if df is None or df.empty:
            st.error('No data to train for '+s)
        else:
            df = add_indicators(df)
            try:
                train_quick(df)
                st.success('Model trained and saved (models/supervised_rf.pkl)')
            except Exception as e:
                st.error('Train failed: '+str(e))

st.sidebar.markdown('---')
mode = st.sidebar.radio('Mode', ['Paper (Bot)','Paper (Manual)'])

# Main layout
col1, col2 = st.columns([3,1])

with col2:
    st.subheader('Paper Orders')
    orders = load_orders()
    if orders:
        st.write(pd.DataFrame(orders).sort_values('time', ascending=False).head(10))
    else:
        st.info('No paper orders yet.')

with col1:
    st.subheader('Market Watch & Scanner')
    if st.button('Run scan now'):
        results = []
        for sym in symbols:
            s = sym.strip()
            df = fetch_ohlcv(s, interval=interval, limit=500)
            if df is None or df.empty:
                st.warning(f'No data for {s}')
                continue
            df = add_indicators(df)
            sig = predict(df)
            price = last_price(df)
            # decide entry sizing
            size_usd = (paper_balance * (risk_pct/100.0))
            qty = round(size_usd / price, 6) if price>0 else 0
            # check open orders limit
            orders = load_orders()
            open_orders = [o for o in orders if o['status']=='open'] if orders else []
            if sig == 1 and len(open_orders) < max_open:
                o = place_order(s, 'BUY', qty, price, note='AI signal')
                results.append({'symbol':s,'side':'BUY','price':price,'qty':qty})
                if tg_token and tg_chat:
                    send_telegram(tg_token, tg_chat, f"AI placed PAPER BUY {s} @ {price}")
            elif sig == 0:
                results.append({'symbol':s,'side':'HOLD','price':price})
        st.write('Scan results:')
        st.write(pd.DataFrame(results))

    st.markdown('---')
    st.subheader('Chart (select symbol)')
    sym_choice = st.selectbox('Symbol', [s.strip() for s in symbols])
    dfc = fetch_ohlcv(sym_choice, interval=interval, limit=1000)
    if dfc is None or dfc.empty:
        st.error('No chart data')
    else:
        dfc = add_indicators(dfc)
        st.line_chart(dfc.set_index('time')['close'].tail(200))
        st.write(dfc[['time','close','ema9','ema25','rsi14']].tail(5))

st.markdown('---')
st.subheader('Backtest (quick)')
if st.button('Run quick backtest on first symbol'):
    s = symbols[0].strip()
    df = fetch_ohlcv(s, interval=interval, limit=2000)
    if df is None or df.empty:
        st.error('No data')
    else:
        df = add_indicators(df)
        def strategy(d): return 'LONG' if (d['close'].iloc[-1] > d['ema25'].iloc[-1] and d['rsi14'].iloc[-1]<70) else None
        trades, final = run_backtest(df, strategy, initial_balance=paper_balance)
        st.write('Trades:', len(trades), 'Final balance:', final)
        st.write(trades)
