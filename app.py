import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
import time
import plotly.graph_objects as go

# --- Config
VTZ = timezone(timedelta(hours=7))
BINANCE_BASE = "https://api.binance.com"
DEFAULT_SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOTUSDT",
    "TIAUSDT","KAVAUSDT","RENDERUSDT","AVAXUSDT","NEARUSDT","SEIUSDT"
]
AUTO_SCAN_INTERVAL_DEFAULT = 900
SL_MAX_PCT = 5.0
TP_MIN_PCT = 10.0
HEADERS = {"User-Agent": "Mozilla/5.0 (AI-Trading-Dashboard)"}

st.set_page_config(page_title="AI Crypto Trading Dashboard", layout="wide")

# --- Utils
def now_vn_str():
    return datetime.now(VTZ).strftime("%Y-%m-%d %H:%M:%S")

def fetch_price(symbol):
    try:
        url = f"{BINANCE_BASE}/api/v3/ticker/price"
        r = requests.get(url, params={"symbol": symbol}, headers=HEADERS, timeout=8)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None

def fetch_klines(symbol, interval="4h", limit=200):
    try:
        url = f"{BINANCE_BASE}/api/v3/klines"
        r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, headers=HEADERS, timeout=12)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","taker_base","taker_quote","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception:
        return None

def linear_fit_y(y):
    if len(y) < 2:
        return 0, float(y.iloc[-1]) if len(y) else 0
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y.values, 1)
    return slope, intercept

def calc_trendlines(df):
    if df is None or df.empty:
        return (0,0),(0,0)
    N = min(60, len(df))
    highs = df["high"].tail(N).reset_index(drop=True)
    lows  = df["low"].tail(N).reset_index(drop=True)
    return linear_fit_y(highs), linear_fit_y(lows)

def predict_line(slope, intercept, idx):
    return slope * idx + intercept

def propose_trade(symbol, price, high_t, low_t):
    s_h, i_h = high_t
    action, signal, note, sl, tp, rr = "WAIT","WAIT","",None,None,None
    try:
        pred_high = predict_line(s_h, i_h, 59)
    except Exception:
        pred_high = None
    if s_h < 0 and price and pred_high and price > pred_high:
        action="Long"; signal="BreakDownTrend"; note="Break trend giảm H4"
        sl = round(price * (1 - SL_MAX_PCT/100), 8)
        tp = round(price * (1 + TP_MIN_PCT/100), 8)
        rr = round((tp-price)/max(price-sl,1e-9),2)
    return {"Symbol":symbol,"Action":action,"Signal":signal,"SL":sl,"TP":tp,"RR":rr,"Note":note}

# --- State init
if "orders" not in st.session_state: st.session_state.orders=[]
if "last_scan_ts" not in st.session_state: st.session_state.last_scan_ts=0
if "scan_results" not in st.session_state: st.session_state.scan_results=[]

# --- Sidebar
st.sidebar.header("Settings / Controls")
symbols_input = st.sidebar.text_area("Symbols", value=",".join(DEFAULT_SYMBOLS))
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
interval_choice = st.sidebar.selectbox("Chart interval (detail)", ["1h","4h","1d"], index=1)
auto_scan = st.sidebar.checkbox("Auto-scan", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", min_value=60, value=AUTO_SCAN_INTERVAL_DEFAULT, step=60)
st.sidebar.write("SL max %:", SL_MAX_PCT, "TP min %:", TP_MIN_PCT)
st.sidebar.write("Time (VN):", now_vn_str())

# --- Market Watch
st.title("📈 AI Crypto Trading Dashboard (Paper-trade)")
col1,col2 = st.columns([2,1])
with col1:
    st.subheader("Market Watch")
    prices=[{"Symbol":s,"Price":fetch_price(s) or "err"} for s in symbols]
    st.table(pd.DataFrame(prices))
with col2:
    st.subheader("Scan / Actions")
    if st.button("Manual Scan now"):
        st.session_state.scan_results=[]
        st.session_state.last_scan_ts=time.time()
        for s in symbols:
            df=fetch_klines(s,"4h",200)
            if df is None: continue
            high_t,low_t=calc_trendlines(df)
            price=fetch_price(s)
            st.session_state.scan_results.append(propose_trade(s,price,high_t,low_t))
        st.success("Manual scan done: "+now_vn_str())
    if auto_scan and time.time()-st.session_state.last_scan_ts>scan_interval:
        st.session_state.scan_results=[]
        st.session_state.last_scan_ts=time.time()
        for s in symbols:
            df=fetch_klines(s,"4h",120)
            if df is None: continue
            high_t,low_t=calc_trendlines(df)
            price=fetch_price(s)
            st.session_state.scan_results.append(propose_trade(s,price,high_t,low_t))
        st.rerun()

    if st.session_state.scan_results:
        df_scan=pd.DataFrame(st.session_state.scan_results)
        df_pick=df_scan[df_scan["Action"]!="WAIT"].sort_values("RR",ascending=False)
        st.table(df_pick)

# --- Chart
st.markdown("---")
st.subheader("Chart & Trendline")
sel=st.selectbox("Select symbol", symbols, index=0)
df=fetch_klines(sel,interval_choice,200)
if df is not None:
    fig=go.Figure(data=[go.Candlestick(x=df["open_time"],open=df["open"],high=df["high"],low=df["low"],close=df["close"])])
    try:
        (s_h,i_h),(s_l,i_l)=calc_trendlines(df)
        xs=[df["open_time"].iloc[-60],df["open_time"].iloc[-1]]
        fig.add_trace(go.Scatter(x=xs,y=[predict_line(s_h,i_h,0),predict_line(s_h,i_h,59)],mode="lines",line=dict(color="blue",dash="dash"),name="Trend High"))
        fig.add_trace(go.Scatter(x=xs,y=[predict_line(s_l,i_l,0),predict_line(s_l,i_l,59)],mode="lines",line=dict(color="orange",dash="dash"),name="Trend Low"))
    except: pass
    st.plotly_chart(fig,use_container_width=True)
else:
    st.error("Cannot fetch data for "+sel)
