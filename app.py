import streamlit as st
import yaml, os, time
from utils import fetch_klines, add_indicators, last_price
from ai_models import train_supervised, load_supervised
from trade_manager import place_paper_order, list_paper_orders, clear_paper_orders
from telegram import send_telegram

st.set_page_config(page_title='AI Crypto Trader (Pro)', layout='wide')
st.title('🤖 AI Crypto Trader — Pro (Paper-trade)')

# Load config if present
if os.path.exists('config.yaml'):
    cfg = yaml.safe_load(open('config.yaml'))
else:
    cfg = {}

st.sidebar.header('Settings')
symbols_text = st.sidebar.text_input('Symbols (comma)', ','.join(cfg.get('default_symbols', ['BTCUSDT','ETHUSDT'])))
symbols = [s.strip().upper() for s in symbols_text.split(',') if s.strip()]
interval = st.sidebar.selectbox('Interval', ['15m','1h','4h','1d'], index=2)
auto_scan = st.sidebar.checkbox('Auto-scan', value=cfg.get('auto_scan', False))
scan_interval = st.sidebar.number_input('Auto-scan interval (sec)', value=cfg.get('auto_scan_interval_sec',900), step=60)
paper_trade = st.sidebar.checkbox('Paper-trade (enabled)', value=cfg.get('paper_trade', True))
tg_token = st.sidebar.text_input('Telegram token', value=cfg.get('telegram_token',''))
tg_chat = st.sidebar.text_input('Telegram chat id', value=cfg.get('telegram_chat_id',''))

st.sidebar.markdown('---')
st.sidebar.subheader('Models (local)')
train_sup_btn = st.sidebar.button('Train supervised (quick)')

st.subheader('Market Watch')
cols = st.columns(min(len(symbols), 6))
prices = {}
for i, sym in enumerate(symbols):
    try:
        p = last_price(sym)
        prices[sym] = p
        cols[i % len(cols)].metric(sym, f"{p:.8f}")
    except Exception as e:
        cols[i % len(cols)].write(f"{sym}: err")

st.markdown('---')
st.subheader('Scanner & AI suggestions')
colL, colR = st.columns([3,1])
with colR:
    if st.button('Run scan now'):
        run_scan = True
    else:
        run_scan = False
    if train_sup_btn:
        st.info('Training supervised model (this may take a minute)...')
        try:
            s = symbols[0]
            df = fetch_klines(s, interval=interval, limit=1500)
            df = add_indicators(df)
            train_supervised(df)
            st.success('Supervised model trained & saved.')
        except Exception as e:
            st.error(f'Train supervised error: {e}')
with colL:
    sup = load_supervised()
    if auto_scan:
        last_run = st.session_state.get('last_auto_scan', 0)
        if time.time() - last_run > scan_interval:
            run_scan = True
            st.session_state['last_auto_scan'] = time.time()
    if run_scan:
        for sym in symbols:
            with st.expander(f"{sym}"):
                try:
                    df = fetch_klines(sym, interval=interval, limit=1000)
                    df = add_indicators(df)
                    st.write(f"Latest close: {df['close'].iloc[-1]:.8f}")
                    st.line_chart(df['close'].tail(200))
                    if sup:
                        X = df[['rsi14','ema20','ema50','atr14','vol_ma20']].fillna(0).values
                        pred = sup.predict(X[-1].reshape(1,-1))[0]
                        suggestion = 'BUY' if pred==1 else 'HOLD/SELL'
                    else:
                        suggestion = 'No model'
                    st.markdown(f"**Suggestion (supervised):** {suggestion}")
                    if suggestion == 'BUY' and paper_trade:
                        price = float(df['close'].iloc[-1])
                        ord = place_paper_order(sym, 'BUY', price, size=1.0, note='AI sup')
                        st.success(f"Placed paper order: {ord}")
                        if tg_token and tg_chat:
                            send_telegram(tg_token, tg_chat, f"Paper BUY {sym} @ {price}")
                except Exception as e:
                    st.error(f"{sym} scan error: {e}")

st.markdown('---')
st.subheader('Paper Orders / History')
ords = list_paper_orders()
if ords:
    st.table(ords[::-1])
    if st.button('Clear paper orders'):
        clear_paper_orders()
        st.experimental_rerun()
else:
    st.info('No paper orders yet.')

st.markdown('---')
st.subheader('Manual Chart & Trendline (single symbol)')
symbol_for_chart = st.selectbox('Choose symbol', symbols)
if symbol_for_chart:
    try:
        df = fetch_klines(symbol_for_chart, interval=interval, limit=1000)
        df = add_indicators(df)
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Candlestick(x=df.index[-200:], open=df['open'].iloc[-200:], high=df['high'].iloc[-200:], low=df['low'].iloc[-200:], close=df['close'].iloc[-200:])])
        fig.add_trace(go.Scatter(x=df.index[-200:], y=df['ema20'].iloc[-200:], name='EMA20'))
        fig.add_trace(go.Scatter(x=df.index[-200:], y=df['ema50'].iloc[-200:], name='EMA50'))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f'Chart error: {e}')

st.markdown('---')
st.caption('Paper-trade only. Live trading requires secure key management and extra checks.')
