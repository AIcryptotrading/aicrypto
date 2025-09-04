# app.py
# AI Crypto Trading Dashboard — REST Binance + improved signals (Fibonacci, volume filter, pivot/trendline)
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

# ----------------------------
# Config
# ----------------------------
BINANCE_API = "https://api.binance.com/api/v3"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT,TIAUSDT,KAVAUSDT,RENDERUSDT,AVAXUSDT,NEARUSDT,SEIUSDT"
AUTO_SCAN_DEFAULT = 900
SL_MAX_DEFAULT = 5.0
TP_MIN_DEFAULT = 10.0

st.set_page_config(page_title="AI Crypto Trading (Fibo+Vol+Pivot)", layout="wide")

# ----------------------------
# Binance REST helpers
# ----------------------------
def bn_price(symbol):
    try:
        r = requests.get(f"{BINANCE_API}/ticker/price", params={"symbol": symbol}, timeout=6)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None

def bn_klines(symbol, interval="4h", limit=500):
    try:
        r = requests.get(f"{BINANCE_API}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=8)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","tbq","tbq2","ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception:
        return None

# ----------------------------
# Indicators & utilities
# ----------------------------
def add_indicators(df):
    if df is None or df.empty: 
        return df
    df = df.copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.rolling(14).mean()
    avg_down = down.rolling(14).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df

def find_pivots(df, left=3, right=3):
    """Return list of pivots: (index, type, price) type: 'H' or 'L'"""
    pivots = []
    n = len(df)
    for i in range(left, n-right):
        is_high = True
        is_low = True
        for j in range(1, left+1):
            if df['high'].iloc[i] <= df['high'].iloc[i-j]:
                is_high = False
            if df['high'].iloc[i] <= df['high'].iloc[i+j]:
                is_high = False
            if df['low'].iloc[i] >= df['low'].iloc[i-j]:
                is_low = False
            if df['low'].iloc[i] >= df['low'].iloc[i+j]:
                is_low = False
        if is_high:
            pivots.append((i, 'H', df['high'].iloc[i]))
        if is_low:
            pivots.append((i, 'L', df['low'].iloc[i]))
    return pivots

def last_swing(df, lookback=200):
    piv = find_pivots(df, left=3, right=3)
    if not piv:
        return None, None
    # find last low then last high sequence
    lows = [p for p in piv if p[1]=='L']
    highs = [p for p in piv if p[1]=='H']
    last_low = lows[-1] if lows else None
    last_high = highs[-1] if highs else None
    # If last_low occurs before last_high (up swing), return low->high; else high->low
    if last_low and last_high:
        if last_low[0] < last_high[0]:
            return ('L','H', (last_low, last_high))
        else:
            return ('H','L', (last_high, last_low))
    return None, None

def fib_levels(low, high):
    diff = high - low
    levels = {
        "0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "100%": low
    }
    return levels

def linear_trend_slope(series):
    x = np.arange(len(series))
    if len(x) < 2:
        return 0
    coef = np.polyfit(x, series.values, 1)
    return coef[0]  # slope

# ----------------------------
# Session state init
# ----------------------------
if "orders" not in st.session_state: st.session_state.orders = []
if "scan_results" not in st.session_state: st.session_state.scan_results = []
if "last_scan" not in st.session_state: st.session_state.last_scan = 0

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Settings")
symbols = st.sidebar.text_area("Symbols (comma separated)", DEFAULT_SYMBOLS, height=140).replace(" ", "").split(",")
chart_interval = st.sidebar.selectbox("Chart interval (detail)", ["15m","1h","4h","1d"], index=2)
auto_scan = st.sidebar.checkbox("Auto-scan", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", min_value=30, value=AUTO_SCAN_DEFAULT, step=30)
volume_factor = st.sidebar.number_input("Volume multiplier (breakout filter)", min_value=1.0, value=1.5, step=0.1)
score_threshold = st.sidebar.number_input("Signal score threshold (0-10)", min_value=1, max_value=10, value=4, step=1)
sl_max_pct = st.sidebar.number_input("SL max %", min_value=0.5, value=SL_MAX_DEFAULT, step=0.5)
tp_min_pct = st.sidebar.number_input("TP min %", min_value=1.0, value=TP_MIN_DEFAULT, step=1.0)

st.sidebar.markdown("Time (system): " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC")

# ----------------------------
# Top layout: Market Watch + Scan
# ----------------------------
st.title("AI Crypto Trading — Fibo + Vol + Pivot signals (Paper-trade)")
colA, colB = st.columns([2,1])

with colA:
    st.subheader("Market Watch")
    price_rows = []
    for s in symbols:
        p = bn_price(s)
        price_rows.append({"symbol": s, "price": round(p,8) if p else "err"})
    st.table(pd.DataFrame(price_rows))

with colB:
    st.subheader("Scan / Actions")
    if st.button("Manual Scan Now"):
        st.session_state.scan_results = []
        st.session_state.last_scan = time.time()
        st.success("Manual scan started")
        # perform scan below outside UI button logic

# ----------------------------
# Core scanner logic (improved)
# ----------------------------
def scan_all():
    results = []
    for s in symbols:
        df = bn_klines(s, interval=chart_interval, limit=500)
        if df is None or df.empty:
            continue
        df = add_indicators(df)
        last = df.iloc[-1]
        price_now = bn_price(s) or last["close"]

        score = 0
        reasons = []

        # 1) RSI extreme
        rsi = last.get("RSI", None)
        if rsi is not None:
            if rsi < 28:
                score += 2; reasons.append("RSI < 28 (oversold)")
            elif rsi > 72:
                score += 2; reasons.append("RSI > 72 (overbought)")

        # 2) MA crossover (momentum)
        if "MA20" in df.columns and "MA50" in df.columns and not np.isnan(df["MA20"].iloc[-2]):
            prev_ma20 = df["MA20"].iloc[-2]; prev_ma50 = df["MA50"].iloc[-2]
            cur_ma20 = df["MA20"].iloc[-1]; cur_ma50 = df["MA50"].iloc[-1]
            if prev_ma20 < prev_ma50 and cur_ma20 > cur_ma50:
                score += 2; reasons.append("MA20 crossed above MA50")
            elif prev_ma20 > prev_ma50 and cur_ma20 < cur_ma50:
                score += 2; reasons.append("MA20 crossed below MA50")

        # 3) Pivot swing + Fibonacci
        piv = find_pivots(df, left=3, right=3)
        last_low = None; last_high = None
        lows = [p for p in piv if p[1]=='L']
        highs = [p for p in piv if p[1]=='H']
        if lows and highs:
            last_low = lows[-1]
            last_high = highs[-1]
            # determine swing direction
            if last_low[0] < last_high[0]:
                swing_low = last_low; swing_high = last_high
            else:
                swing_low = last_high; swing_high = last_low
            # ensure low < high in numeric
            low_price = min(swing_low[2], swing_high[2])
            high_price = max(swing_low[2], swing_high[2])
            fibs = fib_levels(low_price, high_price)
            # if price broke above recent high (breakout)
            recent_high = df["high"].iloc[-21:-1].max()
            recent_low = df["low"].iloc[-21:-1].min()
            breakout = last["close"] > recent_high
            breakdown = last["close"] < recent_low
            if breakout:
                # check volume
                vol = last["volume"]; vol_ma = last.get("vol_ma20", None)
                vol_ok = (vol_ma is not None) and (vol > vol_ma * volume_factor)
                if vol_ok:
                    score += 2; reasons.append("Breakout with volume spike")
                else:
                    score += 1; reasons.append("Breakout but volume weak")
            if breakdown:
                vol = last["volume"]; vol_ma = last.get("vol_ma20", None)
                vol_ok = (vol_ma is not None) and (vol > vol_ma * volume_factor)
                if vol_ok:
                    score += 2; reasons.append("Breakdown with volume spike")
                else:
                    score += 1; reasons.append("Breakdown but volume weak")
            # fib retest check: close is near 38.2/50/61.8
            for k in ["38.2%","50%","61.8%"]:
                lvl = fibs[k]
                if abs(last["close"] - lvl) / lvl < 0.01:  # within 1%
                    score += 1; reasons.append(f"Price touching Fibonacci {k}")
        else:
            # fallback: trend slope check using last 60 closes
            slope = linear_trend_slope(df["close"].tail(60))
            if slope > 0:
                reasons.append("Uptrend slope positive")
            elif slope < 0:
                reasons.append("Downtrend slope negative")

        # 4) volume surge independent
        vol = last["volume"]; vol_ma = last.get("vol_ma20", None)
        if vol_ma and vol > vol_ma * (volume_factor + 0.2):
            score += 1; reasons.append("Strong volume surge")

        # define action based on signs:
        action = None
        if "Breakout" in ";".join(reasons) or ("MA20 crossed above MA50" in reasons):
            action = "LONG"
        if "Breakdown" in ";".join(reasons) or ("MA20 crossed below MA50" in reasons):
            action = "SHORT"

        if score >= score_threshold and action:
            # compute suggestion SL/TP
            entry = price_now
            if action == "LONG":
                sl = round(entry * (1 - min(0.03, sl_max_pct / 100.0)), 8)
                tp = round(entry * (1 + max(tp_min_pct / 100.0, 0.10)), 8)
            else:
                sl = round(entry * (1 + min(0.03, sl_max_pct / 100.0)), 8)
                tp = round(entry * (1 - max(tp_min_pct / 100.0, 0.10)), 8)
            results.append({
                "symbol": s,
                "action": action,
                "price": round(entry,8),
                "sl": sl,
                "tp": tp,
                "score": score,
                "reasons": "; ".join(reasons)
            })
    return results

# If manual scan button pressed earlier or autoscans
if st.button("Run manual scan (full)"):
    st.session_state.scan_results = scan_all()
    st.session_state.last_scan = time.time()
    st.success(f"Scan done: {len(st.session_state.scan_results)} suggestions")

# Auto-scan on load if enabled and interval passed
if auto_scan and (time.time() - st.session_state.last_scan > scan_interval):
    st.session_state.scan_results = scan_all()
    st.session_state.last_scan = time.time()
    if st.session_state.scan_results:
        st.toast = st.success(f"Auto-scan found {len(st.session_state.scan_results)} suggestions at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# ----------------------------
# Display suggestions with details & allow add paper order
# ----------------------------
st.markdown("---")
st.subheader("Suggestions (Fibo+Vol+Pivot)")
if st.session_state.scan_results:
    df_s = pd.DataFrame(st.session_state.scan_results)
    st.table(df_s[["symbol","action","price","sl","tp","score"]])
    for i, row in df_s.iterrows():
        cols = st.columns([3,1])
        with cols[0]:
            st.write(f"**{row['symbol']}** → {row['action']} | price {row['price']} | score {row['score']}")
            st.caption(row["reasons"])
        with cols[1]:
            if st.button(f"Add paper order {i}"):
                # avoid duplicates
                exists = any(o for o in st.session_state.orders if o["symbol"]==row["symbol"] and o["status"]=="OPEN" and o["side"]==row["action"])
                if exists:
                    st.warning("Open similar order exists.")
                else:
                    st.session_state.orders.append({
                        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": row["symbol"],
                        "side": row["action"],
                        "entry": float(row["price"]),
                        "sl": float(row["sl"]),
                        "tp": float(row["tp"]),
                        "size": 1.0,
                        "leverage": 10,
                        "status": "OPEN",
                        "closed_at": None,
                        "current_price": float(row["price"]),
                        "pnl_pct": 0.0,
                        "notes": row["reasons"]
                    })
                    st.success("Paper order added.")
else:
    st.info("No suggestions at the moment. Try manual scan or wait for auto-scan.")

# ----------------------------
# Chart (detailed) with fib and pivots
# ----------------------------
st.markdown("---")
st.subheader("Chart & Analysis")

sel = st.selectbox("Choose symbol for chart", symbols, index=0)
d = bn_klines(sel, interval=chart_interval, limit=500)
if d is None:
    st.error("Cannot fetch chart data.")
else:
    d = add_indicators(d)
    pivots = find_pivots(d, left=3, right=3)
    fig = go.Figure(data=[go.Candlestick(x=d["open_time"], open=d["open"], high=d["high"], low=d["low"], close=d["close"], name=sel)])
    if "MA20" in d.columns:
        fig.add_trace(go.Scatter(x=d["open_time"], y=d["MA20"], name="MA20"))
    if "MA50" in d.columns:
        fig.add_trace(go.Scatter(x=d["open_time"], y=d["MA50"], name="MA50"))
    # draw pivots
    for p in pivots[-30:]:
        idx, typ, price = p
        t = d["open_time"].iloc[idx]
        if typ == 'H':
            fig.add_trace(go.Scatter(x=[t], y=[price], mode="markers", marker=dict(color="red",size=8), name="Pivot H", showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[t], y=[price], mode="markers", marker=dict(color="green",size=8), name="Pivot L", showlegend=False))
    # draw fib if available
    # take last clear swing (low->high or high->low)
    if pivots:
        lows = [p for p in pivots if p[1]=='L']
        highs = [p for p in pivots if p[1]=='H']
        if lows and highs:
            last_low = lows[-1][2] if lows else None
            last_high = highs[-1][2] if highs else None
            if last_low and last_high and last_low < last_high:
                fibs = fib_levels(last_low, last_high)
                for k,v in fibs.items():
                    fig.add_hline(y=v, line=dict(color="yellow" if k in ["50%","38.2%","61.8%"] else "gray", width=1), annotation_text=k, annotation_position="top left")
    fig.update_layout(height=650, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    # RSI small chart
    if "RSI" in d.columns:
        st.line_chart(d.set_index("open_time")["RSI"].tail(200))

# ----------------------------
# Paper order manager
# ----------------------------
st.markdown("---")
st.subheader("Paper Orders")
def refresh_orders():
    for i,o in enumerate(st.session_state.orders):
        if o["status"] == "CLOSED": continue
        cp = bn_price(o["symbol"]) or o.get("current_price", o["entry"])
        st.session_state.orders[i]["current_price"] = round(cp,8)
        if o["side"] == "LONG":
            pnl = (cp - o["entry"]) / o["entry"] * 100
        else:
            pnl = (o["entry"] - cp) / o["entry"] * 100
        st.session_state.orders[i]["pnl_pct"] = round(pnl,4)
        # auto close by TP/SL
        if o["status"] == "OPEN":
            if o["side"] == "LONG":
                if cp >= o["tp"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f"Order TP hit (paper): {o['symbol']}")
                elif cp <= o["sl"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    st.warning(f"Order SL hit (paper): {o['symbol']}")
            else:
                if cp <= o["tp"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f"Order TP hit (paper): {o['symbol']}")
                elif cp >= o["sl"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    st.warning(f"Order SL hit (paper): {o['symbol']}")

refresh_orders()
if st.session_state.orders:
    df_o = pd.DataFrame(st.session_state.orders)
    st.dataframe(df_o[["created_at","symbol","side","entry","current_price","pnl_pct","sl","tp","status","closed_at","notes"]], use_container_width=True)
    # actions
    for idx, ord in enumerate(list(st.session_state.orders)):
        c1,c2,c3 = st.columns([3,1,1])
        c1.write(f"{ord['symbol']} | {ord['side']} | Entry {ord['entry']} | Status {ord['status']}")
        if c2.button(f"Close {idx}"):
            st.session_state.orders[idx]["status"] = "CLOSED"
            st.session_state.orders[idx]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            st.experimental_rerun()
        if c3.button(f"Delete {idx}"):
            st.session_state.orders.pop(idx)
            st.experimental_rerun()
    csv = pd.DataFrame(st.session_state.orders).to_csv(index=False).encode("utf-8")
    st.download_button("Download orders CSV", csv, file_name=f"paper_orders_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv")
else:
    st.info("No paper orders yet.")

# ----------------------------
# Reports
# ----------------------------
st.markdown("---")
st.subheader("Reports")
if st.button("Daily closed summary"):
    closed = [o for o in st.session_state.orders if o["status"]=="CLOSED"]
    dfc = pd.DataFrame(closed) if closed else pd.DataFrame()
    total = dfc["pnl_pct"].sum() if not dfc.empty else 0
    wins = dfc[dfc["pnl_pct"]>0].shape[0] if not dfc.empty else 0
    losses = dfc[dfc["pnl_pct"]<=0].shape[0] if not dfc.empty else 0
    st.json({
        "time_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "closed": len(closed),
        "wins": wins, "losses": losses, "total_pnl_pct": round(total,4)
    })
st.info("Paper-trade only. For live trading integrate secure APIs separately (not included).")
