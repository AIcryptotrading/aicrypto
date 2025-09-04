# app.py - AI Crypto Trading Dashboard (full, paper-trade)
import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import datetime
import plotly.graph_objects as go
import time

# -----------------------------
# Config & init
# -----------------------------
st.set_page_config(page_title="AI Crypto Trading Dashboard", layout="wide")
VTZ = datetime.timezone(datetime.timedelta(hours=7))
binance = ccxt.binance()  # using public market endpoints (spot)
binance.load_markets()

# defaults
DEFAULT_SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOTUSDT",
    "TIAUSDT","KAVAUSDT","RENDERUSDT","AVAXUSDT","NEARUSDT","SEIUSDT"
]
AUTO_SCAN_INTERVAL_DEFAULT = 900  # sec
SL_MAX_DEFAULT = 5.0
TP_MIN_DEFAULT = 10.0

# -----------------------------
# Utilities
# -----------------------------
def now_vn_str():
    return datetime.datetime.now(VTZ).strftime("%Y-%m-%d %H:%M:%S")

def symbol_ccxt(sym: str):
    """Convert BTCUSDT -> BTC/USDT for ccxt queries."""
    s = sym.strip().upper()
    if "/" not in s:
        if s.endswith("USDT"):
            return s.replace("USDT", "/USDT")
        else:
            return s
    return s

def safe_fetch_ticker(sym: str):
    try:
        return binance.fetch_ticker(symbol_ccxt(sym))
    except Exception:
        return None

def fetch_price(sym: str):
    t = safe_fetch_ticker(sym)
    if t:
        return float(t.get("last", None) or 0.0)
    return None

def fetch_ohlcv(sym: str, timeframe="4h", limit=200):
    try:
        bars = binance.fetch_ohlcv(symbol_ccxt(sym), timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception:
        return None

# indicators
def add_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    df = df.copy()
    # MA
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    # RSI simple
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# simple trendline fit (linear on highs/lows)
def calc_trendlines(df: pd.DataFrame, lookback=60):
    if df is None or df.empty:
        return (0,0),(0,0)
    N = min(lookback, len(df))
    highs = df["high"].tail(N).reset_index(drop=True)
    lows  = df["low"].tail(N).reset_index(drop=True)
    x = np.arange(len(highs))
    if len(x) < 2:
        return (0, highs.iloc[-1] if len(highs) else 0), (0, lows.iloc[-1] if len(lows) else 0)
    s_high, i_high = np.polyfit(x, highs.values, 1)
    s_low, i_low = np.polyfit(x, lows.values, 1)
    return (s_high, i_high), (s_low, i_low)

def predict_line(slope, intercept, idx):
    return slope*idx + intercept

# -----------------------------
# Session state init
# -----------------------------
if "orders" not in st.session_state:
    # order structure: dict time,symbol,side,entry,sl,tp,size,leverage,status,created_at,closed_at,current_price,pnl_pct,note
    st.session_state.orders = []
if "last_scan_ts" not in st.session_state:
    st.session_state.last_scan_ts = 0
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

# -----------------------------
# Sidebar: settings
# -----------------------------
st.sidebar.header("Settings / Controls")
symbols_in = st.sidebar.text_area("Symbols (comma separated)", value=",".join(DEFAULT_SYMBOLS), height=160)
symbols = [s.strip().upper() for s in symbols_in.split(",") if s.strip()]
chart_interval = st.sidebar.selectbox("Chart interval (detail)", ["1h","4h","1d"], index=1)
auto_scan = st.sidebar.checkbox("Auto-scan market for setups", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", min_value=60, value=AUTO_SCAN_INTERVAL_DEFAULT, step=60)
max_suggest = st.sidebar.number_input("Max suggestions per scan", min_value=1, value=6, step=1)
sl_max_pct = st.sidebar.number_input("SL max %", min_value=0.5, value=SL_MAX_DEFAULT, step=0.5)
tp_min_pct = st.sidebar.number_input("TP min %", min_value=1.0, value=TP_MIN_DEFAULT, step=1.0)
st.sidebar.markdown("Time (VN): " + now_vn_str())

# -----------------------------
# Top: Market Watch + Manual Scan
# -----------------------------
st.title("📈 AI Crypto Trading Dashboard — Paper-trade")
col_left, col_right = st.columns([2,1])

with col_left:
    st.subheader("Market Watch")
    prices_list = []
    for s in symbols:
        p = fetch_price(s)
        prices_list.append({"Symbol": s, "Price": round(p,8) if p else "err"})
    df_prices = pd.DataFrame(prices_list)
    st.table(df_prices)

with col_right:
    st.subheader("Scan / Actions")
    if st.button("Manual Scan now"):
        st.session_state.scan_results = []
        st.session_state.last_scan_ts = time.time()
        for s in symbols:
            df = fetch_ohlcv(s, timeframe="4h", limit=200)
            if df is None: continue
            (s_h,i_h),(s_l,i_l) = calc_trendlines(df)
            df = add_indicators(df)
            price = fetch_price(s)
            # simple signals
            signal = "WAIT"
            note = ""
            # RSI breakout
            rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and not df["RSI"].isna().all() else None
            if rsi is not None and rsi < 30:
                signal="LONG"; note="RSI oversold"
            if rsi is not None and rsi > 70:
                signal="SHORT"; note="RSI overbought"
            # MA crossover
            if "MA20" in df.columns and "MA50" in df.columns:
                if df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
                    signal="LONG"; note="MA20>MA50"
                elif df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
                    signal="SHORT"; note="MA20<MA50"
            # breakout high/low (20)
            recent_high = df["high"].iloc[-20:].max()
            recent_low = df["low"].iloc[-20:].min()
            if df["close"].iloc[-1] > recent_high:
                signal="LONG"; note="Breakout recent high"
            elif df["close"].iloc[-1] < recent_low:
                signal="SHORT"; note="Breakdown recent low"

            if signal != "WAIT":
                # compute conservative SL/TP
                entry = price or df["close"].iloc[-1]
                if signal == "LONG":
                    sl = round(entry * (1 - min(0.03, sl_max_pct/100.0)), 8)
                    tp = round(entry * (1 + max(tp_min_pct/100.0, 0.10)), 8)
                else:
                    sl = round(entry * (1 + min(0.03, sl_max_pct/100.0)), 8)
                    tp = round(entry * (1 - max(tp_min_pct/100.0, 0.10)), 8)
                rr = (tp - entry) / max((entry - sl) if signal=="LONG" else (sl - entry), 1e-9)
                st.session_state.scan_results.append({
                    "Symbol": s, "Action": signal, "Price": entry, "SL": sl, "TP": tp, "RR": round(rr,2), "Note": note
                })
        st.success("Manual scan finished: " + now_vn_str())

    st.markdown("**Auto-scan:** " + ("ON" if auto_scan else "OFF"))
    if auto_scan and (time.time() - st.session_state.last_scan_ts > scan_interval):
        # perform lightweight scan
        st.session_state.scan_results = []
        st.session_state.last_scan_ts = time.time()
        for s in symbols:
            df = fetch_ohlcv(s, timeframe="4h", limit=120)
            if df is None: continue
            df = add_indicators(df)
            price = fetch_price(s)
            signal = "WAIT"; note=""
            try:
                rsi = float(df["RSI"].iloc[-1])
            except Exception:
                rsi = None
            if rsi is not None and rsi < 30:
                signal="LONG"; note="RSI oversold"
            if rsi is not None and rsi > 70:
                signal="SHORT"; note="RSI overbought"
            if "MA20" in df.columns and "MA50" in df.columns:
                if df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
                    signal="LONG"; note="MA20>MA50"
                elif df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
                    signal="SHORT"; note="MA20<MA50"
            recent_high = df["high"].iloc[-20:].max()
            recent_low = df["low"].iloc[-20:].min()
            if df["close"].iloc[-1] > recent_high:
                signal="LONG"; note="Breakout recent high"
            elif df["close"].iloc[-1] < recent_low:
                signal="SHORT"; note="Breakdown recent low"

            if signal != "WAIT":
                entry = price or df["close"].iloc[-1]
                if signal=="LONG":
                    sl = round(entry * (1 - min(0.03, sl_max_pct/100.0)), 8)
                    tp = round(entry * (1 + max(tp_min_pct/100.0, 0.10)), 8)
                else:
                    sl = round(entry * (1 + min(0.03, sl_max_pct/100.0)), 8)
                    tp = round(entry * (1 - max(tp_min_pct/100.0, 0.10)), 8)
                rr = (tp - entry) / max((entry - sl) if signal=="LONG" else (sl - entry), 1e-9)
                st.session_state.scan_results.append({
                    "Symbol": s, "Action": signal, "Price": entry, "SL": sl, "TP": tp, "RR": round(rr,2), "Note": note
                })
        # refresh UI once after building scan results
        st.experimental_set_query_params(_t=int(time.time()))
        st.rerun()

    # show suggestions table
    if st.session_state.scan_results:
        df_sugg = pd.DataFrame(st.session_state.scan_results)
        if not df_sugg.empty:
            st.write("Suggestions (top):")
            st.table(df_sugg.head(int(max_suggest)))
            # quick add buttons
            for i, row in df_sugg.head(int(max_suggest)).iterrows():
                c1, c2 = st.columns([3,1])
                with c1:
                    st.write(f"**{row['Symbol']}** → {row['Action']} | RR={row['RR']} | {row['Note']}")
                with c2:
                    if st.button(f"Add {row['Symbol']}_{i}"):
                        # add paper order
                        order = {
                            "created_at": now_vn_str(),
                            "symbol": row["Symbol"],
                            "side": row["Action"],
                            "entry": float(row["Price"]),
                            "sl": float(row["SL"]),
                            "tp": float(row["TP"]),
                            "size": 1.0,
                            "leverage": 10,
                            "status": "OPEN",
                            "closed_at": None,
                            "current_price": float(row["Price"]),
                            "pnl_pct": 0.0,
                            "note": row["Note"]
                        }
                        st.session_state.orders.append(order)
                        st.success("Order added: " + row["Symbol"])

# -----------------------------
# Orders manager (paper-trade)
# -----------------------------
st.markdown("---")
st.subheader("Order Manager (Paper-trade)")

# Function: update orders with current price and check TP/SL
def update_orders():
    if not st.session_state.orders:
        return
    uniq = sorted(set([o["symbol"] for o in st.session_state.orders]))
    prices = {s: fetch_price(s) for s in uniq}
    for i, ord in enumerate(st.session_state.orders):
        cp = prices.get(ord["symbol"]) or ord["current_price"] or ord["entry"]
        st.session_state.orders[i]["current_price"] = round(cp,8)
        # compute pnl (percent)
        if ord["side"].upper() == "LONG":
            pnl_pct = (cp - ord["entry"]) / ord["entry"] * 100
        else:
            pnl_pct = (ord["entry"] - cp) / ord["entry"] * 100
        st.session_state.orders[i]["pnl_pct"] = round(pnl_pct,4)
        # check TP/SL automatic close
        if ord["status"] == "OPEN":
            if ord["side"].upper() == "LONG":
                if cp >= ord["tp"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = now_vn_str()
                elif cp <= ord["sl"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = now_vn_str()
            else:  # SHORT
                if cp <= ord["tp"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = now_vn_str()
                elif cp >= ord["sl"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = now_vn_str()

# call update_orders on each load
update_orders()

# manual add order form
with st.form("manual_order"):
    c1,c2,c3 = st.columns(3)
    sym_in = c1.text_input("Coin", value=(symbols[0] if symbols else "BTCUSDT"))
    side_in = c2.selectbox("Side", ["Long","Short"])
    entry_in = c3.number_input("Entry price", value=float(fetch_price(sym_in) or 0.0), format="%.8f")
    c4,c5 = st.columns(2)
    size_in = c4.number_input("Size (units)", value=1.0, min_value=0.0001)
    lev_in  = c5.number_input("Leverage", value=10, min_value=1)
    tp_in = st.number_input("TP (price)", value=round(entry_in*(1.10 if side_in=="Long" else 0.90),8), format="%.8f")
    sl_in = st.number_input("SL (price)", value=round(entry_in*(0.97 if side_in=="Long" else 1.03),8), format="%.8f")
    note_in = st.text_input("Note","")
    add_btn = st.form_submit_button("Add manual order")
    if add_btn:
        st.session_state.orders.append({
            "created_at": now_vn_str(),
            "symbol": sym_in.upper(),
            "side": side_in,
            "entry": float(entry_in),
            "sl": float(sl_in),
            "tp": float(tp_in),
            "size": float(size_in),
            "leverage": int(lev_in),
            "status": "OPEN",
            "closed_at": None,
            "current_price": float(entry_in),
            "pnl_pct": 0.0,
            "note": note_in
        })
        st.success("Manual order added")

# display orders table
if st.session_state.orders:
    df_orders = pd.DataFrame(st.session_state.orders)
    st.dataframe(df_orders[["created_at","symbol","side","entry","current_price","pnl_pct","sl","tp","status","closed_at","note"]], use_container_width=True)

    # per-order actions
    for idx, ord in enumerate(list(st.session_state.orders)):
        cols = st.columns([3,1,1,1])
        cols[0].write(f"{ord['symbol']} | {ord['side']} | Entry {ord['entry']} | Status {ord['status']}")
        if cols[1].button(f"Close {idx}"):
            st.session_state.orders[idx]["status"]="CLOSED"
            st.session_state.orders[idx]["closed_at"]=now_vn_str()
            st.experimental_rerun()
        if cols[2].button(f"Delete {idx}"):
            st.session_state.orders.pop(idx)
            st.experimental_rerun()
        if cols[3].button(f"Edit {idx}"):
            st.info("Edit via GitHub file or delete & re-add (simple editor not provided)")

    # download csv
    csv = pd.DataFrame(st.session_state.orders).to_csv(index=False).encode("utf-8")
    st.download_button("Download orders CSV", csv, file_name=f"orders_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv")

# -----------------------------
# Summary & notes
# -----------------------------
st.markdown("---")
st.subheader("Daily Summary (on-demand)")
if st.button("Generate daily report"):
    dfc = pd.DataFrame(st.session_state.orders)
    closed = dfc[dfc["status"]=="CLOSED"] if not dfc.empty else pd.DataFrame()
    wins = closed[closed["pnl_pct"]>0] if not closed.empty else pd.DataFrame()
    losses = closed[closed["pnl_pct"]<=0] if not closed.empty else pd.DataFrame()
    total_pnl = closed["pnl_pct"].sum() if not closed.empty else 0
    summary = {
        "time": now_vn_str(),
        "total_orders": len(dfc),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "total_pnl_pct": round(total_pnl,4)
    }
    st.json(summary)
    st.success("Report ready — download CSV above for full log.")

st.info("This is paper-trading only. The app does NOT place real trades. Use at your own risk.")
