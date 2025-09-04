# app.py
# AI Crypto Trading Dashboard — Auto-scan + Auto-execute (paper-trade)
# Uses Binance public REST API (requests). Paper-trade only by default.

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# ---------------------------
# Config
# ---------------------------
BINANCE_API = "https://api.binance.com/api/v3"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT,TIAUSDT,KAVAUSDT,RENDERUSDT,AVAXUSDT,NEARUSDT,SEIUSDT"
AUTO_SCAN_DEFAULT = 900  # seconds
SL_MAX_DEFAULT = 5.0
TP_MIN_DEFAULT = 10.0
PAPER_LEVERAGE_DEFAULT = 10

st.set_page_config(page_title="AI Crypto Trading — Auto Paper Trading", layout="wide")

# ---------------------------
# Utilities: BN API (requests)
# ---------------------------
def bn_price(symbol):
    """Return last price (float) or None."""
    try:
        resp = requests.get(f"{BINANCE_API}/ticker/price", params={"symbol": symbol}, timeout=6)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        return None

def bn_klines(symbol, interval="4h", limit=200):
    """Return DataFrame of klines or None."""
    try:
        resp = requests.get(f"{BINANCE_API}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception:
        return None

# indicators
def add_indicators(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    return df

# trendline simple fit on highs/lows
def fit_trendline(df, lookback=60):
    if df is None or df.empty or len(df) < 2:
        return (0,0),(0,0)
    N = min(lookback, len(df))
    highs = df["high"].tail(N).reset_index(drop=True)
    lows = df["low"].tail(N).reset_index(drop=True)
    x = np.arange(len(highs))
    s_h, i_h = np.polyfit(x, highs.values, 1)
    s_l, i_l = np.polyfit(x, lows.values, 1)
    return (s_h, i_h), (s_l, i_l)

def predict_val(slope, intercept, idx):
    return slope*idx + intercept

# ---------------------------
# Session state init
# ---------------------------
if "orders" not in st.session_state:
    st.session_state.orders = []   # list of dicts for paper trades
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "last_scan" not in st.session_state:
    st.session_state.last_scan = 0
if "auto_execute" not in st.session_state:
    st.session_state.auto_execute = False

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Settings & Controls")
symbols_text = st.sidebar.text_area("Symbols (comma sep)", DEFAULT_SYMBOLS, height=120)
symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
chart_interval = st.sidebar.selectbox("Chart interval (detail)", ["15m","1h","4h","1d"], index=2)
auto_scan = st.sidebar.checkbox("Auto-scan", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", min_value=30, value=AUTO_SCAN_DEFAULT, step=30)
max_suggestions = st.sidebar.number_input("Max suggestions per scan", min_value=1, value=6, step=1)
sl_max_pct = st.sidebar.number_input("SL max %", min_value=0.5, value=SL_MAX_DEFAULT, step=0.5)
tp_min_pct = st.sidebar.number_input("TP min %", min_value=1.0, value=TP_MIN_DEFAULT, step=1.0)
paper_leverage = st.sidebar.number_input("Paper leverage", min_value=1, value=PAPER_LEVERAGE_DEFAULT)

st.sidebar.markdown("---")
st.sidebar.markdown("**Execution mode**")
col_e1, col_e2 = st.sidebar.columns([3,1])
with col_e1:
    if st.sidebar.checkbox("Auto-execute signals (paper-trade)", value=False):
        st.session_state.auto_execute = True
    else:
        st.session_state.auto_execute = False
with col_e2:
    live_mode = st.sidebar.checkbox("Enable LIVE trading (DISABLED)", value=False, disabled=True)
    # disabled — explained in label
st.sidebar.caption("Live trading disabled in app for safety. Contact developer for secure integration.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"Time (System): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)")

# ---------------------------
# Top: Market Watch + Manual Scan
# ---------------------------
st.title("🤖 AI Crypto Trading — Auto Paper Trading")
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Market Watch")
    price_rows = []
    for s in symbols:
        p = bn_price(s)
        price_rows.append({"symbol": s, "price": round(p,8) if p else "err"})
    st.table(pd.DataFrame(price_rows))

with col2:
    st.subheader("Scan / Auto-execute")
    if st.button("Manual scan now"):
        st.session_state.scan_results = []
        st.session_state.last_scan = time.time()
        # perform scan
        for s in symbols:
            df = bn_klines(s, interval="4h", limit=200)
            if df is None:
                continue
            df = add_indicators(df)
            # simple signals
            last = df.iloc[-1]
            signal = None
            notes = []
            # RSI extremes
            rsi = last.get("RSI", None)
            if rsi is not None:
                if rsi < 30:
                    signal = "LONG"; notes.append("RSI oversold")
                elif rsi > 70:
                    signal = "SHORT"; notes.append("RSI overbought")
            # MA crossover (last two candles)
            if "MA20" in df.columns and "MA50" in df.columns and not df["MA20"].isna().all():
                if df["MA20"].iloc[-2] < df["MA50"].iloc[-2] and df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
                    signal = "LONG"; notes.append("MA20 crossed above MA50")
                if df["MA20"].iloc[-2] > df["MA50"].iloc[-2] and df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
                    signal = "SHORT"; notes.append("MA20 crossed below MA50")
            # breakout recent high/low
            recent_high = df["high"].iloc[-21:-1].max()
            recent_low  = df["low"].iloc[-21:-1].min()
            if last["close"] > recent_high:
                signal = "LONG"; notes.append("Breakout recent high")
            if last["close"] < recent_low:
                signal = "SHORT"; notes.append("Breakdown recent low")

            if signal:
                price_now = bn_price(s) or last["close"]
                # compute SL/TP conservatively
                if signal == "LONG":
                    sl = round(price_now * (1 - min(0.03, sl_max_pct/100.0)), 8)
                    tp = round(price_now * (1 + max(tp_min_pct/100.0, 0.10)), 8)
                else:
                    sl = round(price_now * (1 + min(0.03, sl_max_pct/100.0)), 8)
                    tp = round(price_now * (1 - max(tp_min_pct/100.0, 0.10)), 8)
                rr = None
                try:
                    if signal == "LONG":
                        rr = round((tp - price_now) / max(price_now - sl, 1e-9), 2)
                    else:
                        rr = round((price_now - tp) / max(sl - price_now, 1e-9), 2)
                except Exception:
                    rr = None
                result = {
                    "symbol": s, "signal": signal, "price": round(price_now,8),
                    "sl": sl, "tp": tp, "rr": rr, "notes": "; ".join(notes)
                }
                st.session_state.scan_results.append(result)

        st.success("Manual scan completed.")

    st.markdown(f"Auto-scan: {'ON' if auto_scan else 'OFF'}  |  Auto-execute paper-trade: {'ON' if st.session_state.auto_execute else 'OFF'}")
    st.markdown(f"Last scan: {datetime.utcfromtimestamp(st.session_state.last_scan).strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_scan else 'Never'}")

# ---------------------------
# Auto-scan (triggered on page load if time elapsed)
# ---------------------------
def perform_auto_scan():
    st.session_state.scan_results = []
    st.session_state.last_scan = time.time()
    for s in symbols:
        df = bn_klines(s, interval="4h", limit=200)
        if df is None:
            continue
        df = add_indicators(df)
        last = df.iloc[-1]
        signal = None
        notes = []
        rsi = last.get("RSI", None)
        if rsi is not None:
            if rsi < 30:
                signal = "LONG"; notes.append("RSI oversold")
            elif rsi > 70:
                signal = "SHORT"; notes.append("RSI overbought")
        if "MA20" in df.columns and "MA50" in df.columns and not df["MA20"].isna().all():
            if df["MA20"].iloc[-2] < df["MA50"].iloc[-2] and df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
                signal = "LONG"; notes.append("MA20>MA50 crossover")
            if df["MA20"].iloc[-2] > df["MA50"].iloc[-2] and df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
                signal = "SHORT"; notes.append("MA20<MA50 crossover")
        recent_high = df["high"].iloc[-21:-1].max()
        recent_low  = df["low"].iloc[-21:-1].min()
        if last["close"] > recent_high:
            signal = "LONG"; notes.append("Breakout recent high")
        if last["close"] < recent_low:
            signal = "SHORT"; notes.append("Breakdown recent low")
        if signal:
            price_now = bn_price(s) or last["close"]
            if signal == "LONG":
                sl = round(price_now * (1 - min(0.03, sl_max_pct/100.0)), 8)
                tp = round(price_now * (1 + max(tp_min_pct/100.0, 0.10)), 8)
            else:
                sl = round(price_now * (1 + min(0.03, sl_max_pct/100.0)), 8)
                tp = round(price_now * (1 - max(tp_min_pct/100.0, 0.10)), 8)
            rr = None
            try:
                if signal == "LONG":
                    rr = round((tp - price_now) / max(price_now - sl, 1e-9), 2)
                else:
                    rr = round((price_now - tp) / max(sl - price_now, 1e-9), 2)
            except Exception:
                rr = None
            st.session_state.scan_results.append({
                "symbol": s, "signal": signal, "price": round(price_now,8),
                "sl": sl, "tp": tp, "rr": rr, "notes": "; ".join(notes)
            })

# auto trigger when page loads and interval passed
if auto_scan and (time.time() - st.session_state.last_scan > scan_interval):
    perform_auto_scan()
    # if auto-execute enabled, create paper orders automatically
    if st.session_state.auto_execute and st.session_state.scan_results:
        for r in st.session_state.scan_results:
            # avoid duplicate same open order for same symbol+side
            exists = False
            for o in st.session_state.orders:
                if o["symbol"] == r["symbol"] and o["status"] == "OPEN" and o["side"] == r["signal"]:
                    exists = True; break
            if exists:
                continue
            order = {
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": r["symbol"],
                "side": r["signal"],
                "entry": float(r["price"]),
                "sl": float(r["sl"]),
                "tp": float(r["tp"]),
                "size": 1.0,
                "leverage": int(paper_leverage),
                "status": "OPEN",
                "closed_at": None,
                "current_price": float(r["price"]),
                "pnl_pct": 0.0,
                "notes": r["notes"]
            }
            st.session_state.orders.append(order)
            st.success(f"Auto paper order created: {r['symbol']} {r['signal']} @ {r['price']} (SL {r['sl']} TP {r['tp']})")

# ---------------------------
# Show scan results & let user add manually
# ---------------------------
st.markdown("---")
st.subheader("Scan Results / Suggestions")
if st.session_state.scan_results:
    df_s = pd.DataFrame(st.session_state.scan_results)
    st.table(df_s.head(int(max_suggestions)))
    # allow manual add
    for i, row in df_s.head(int(max_suggestions)).iterrows():
        colA, colB = st.columns([4,1])
        with colA:
            st.write(f"**{row['symbol']}** → {row['signal']} | Price {row['price']} | RR {row['rr']} | {row['notes']}")
        with colB:
            if st.button(f"Add paper order {i}"):
                # add to orders
                already = any(o for o in st.session_state.orders if o["symbol"]==row["symbol"] and o["status"]=="OPEN" and o["side"]==row["signal"])
                if already:
                    st.warning("Open order with same symbol+side already exists.")
                else:
                    st.session_state.orders.append({
                        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": row["symbol"],
                        "side": row["signal"],
                        "entry": float(row["price"]),
                        "sl": float(row["sl"]),
                        "tp": float(row["tp"]),
                        "size": 1.0,
                        "leverage": int(paper_leverage),
                        "status": "OPEN",
                        "closed_at": None,
                        "current_price": float(row["price"]),
                        "pnl_pct": 0.0,
                        "notes": row["notes"]
                    })
                    st.success("Paper order added.")

else:
    st.info("No scan results yet. Click 'Manual scan' or enable Auto-scan.")

# ---------------------------
# Chart area (candles + MA + RSI)
# ---------------------------
st.markdown("---")
st.subheader("Chart / Detailed view")
sel = st.selectbox("Select symbol for chart", symbols, index=0)
dfc = bn_klines(sel, interval=chart_interval, limit=500)
if dfc is None:
    st.error(f"Cannot fetch data for {sel}")
else:
    dfc = add_indicators(dfc)
    fig = go.Figure(data=[go.Candlestick(x=dfc["open_time"], open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name=sel)])
    if "MA20" in dfc.columns:
        fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA20"], mode="lines", name="MA20"))
    if "MA50" in dfc.columns:
        fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA50"], mode="lines", name="MA50"))
    fig.update_layout(height=600, template="plotly_dark", xaxis_title="Time", yaxis_title="Price (USDT)")
    st.plotly_chart(fig, use_container_width=True)
    # RSI small
    if "RSI" in dfc.columns:
        st.line_chart(dfc[["open_time","RSI"]].set_index("open_time").tail(200), height=150)

# ---------------------------
# Orders manager (paper)
# ---------------------------
st.markdown("---")
st.subheader("Paper Order Manager")
# Update orders: refresh current price & pnl; auto-close TP/SL
def refresh_orders():
    for i, o in enumerate(st.session_state.orders):
        if o["status"] == "CLOSED":
            continue
        cp = bn_price(o["symbol"])
        if cp is None:
            cp = o.get("current_price", o["entry"])
        st.session_state.orders[i]["current_price"] = round(cp,8)
        if o["side"] == "LONG":
            pnl = (cp - o["entry"]) / o["entry"] * 100
        else:
            pnl = (o["entry"] - cp) / o["entry"] * 100
        st.session_state.orders[i]["pnl_pct"] = round(pnl,4)
        # auto TP/SL
        if o["status"] == "OPEN":
            if o["side"] == "LONG":
                if cp >= o["tp"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f"Order TP hit (paper): {o['symbol']} LONG at {cp}")
                elif cp <= o["sl"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    st.warning(f"Order SL hit (paper): {o['symbol']} LONG at {cp}")
            else:
                if cp <= o["tp"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f"Order TP hit (paper): {o['symbol']} SHORT at {cp}")
                elif cp >= o["sl"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    st.warning(f"Order SL hit (paper): {o['symbol']} SHORT at {cp}")

# allow manual add form
with st.form("manual_add"):
    c1,c2,c3 = st.columns(3)
    sym_in = c1.text_input("Symbol", value=(symbols[0] if symbols else "BTCUSDT"))
    side_in = c2.selectbox("Side", ["LONG","SHORT"])
    entry_in = c3.number_input("Entry price", value=float(bn_price(sym_in) or 0.0), format="%.8f")
    tp_in = st.number_input("TP price", value=round(entry_in*(1.10 if side_in=="LONG" else 0.90),8), format="%.8f")
    sl_in = st.number_input("SL price", value=round(entry_in*(0.97 if side_in=="LONG" else 1.03),8), format="%.8f")
    size_in = st.number_input("Size (units)", value=1.0, step=0.01)
    add_btn = st.form_submit_button("Add paper order")
    if add_btn:
        st.session_state.orders.append({
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": sym_in.upper(),
            "side": side_in,
            "entry": float(entry_in),
            "sl": float(sl_in),
            "tp": float(tp_in),
            "size": float(size_in),
            "leverage": int(paper_leverage),
            "status": "OPEN",
            "closed_at": None,
            "current_price": float(entry_in),
            "pnl_pct": 0.0,
            "notes": "manual"
        })
        st.success("Paper order added.")

refresh_orders()

if st.session_state.orders:
    dfo = pd.DataFrame(st.session_state.orders)
    st.dataframe(dfo[["created_at","symbol","side","entry","current_price","pnl_pct","sl","tp","status","closed_at","notes"]], use_container_width=True)

    # action buttons per order
    for idx, ord in enumerate(list(st.session_state.orders)):
        cols = st.columns([3,1,1,1])
        cols[0].write(f"{ord['symbol']} | {ord['side']} | Entry {ord['entry']} | Status {ord['status']}")
        if cols[1].button(f"Close {idx}"):
            st.session_state.orders[idx]["status"] = "CLOSED"
            st.session_state.orders[idx]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            st.experimental_rerun()
        if cols[2].button(f"Delete {idx}"):
            st.session_state.orders.pop(idx)
            st.experimental_rerun()
        if cols[3].button(f"Edit {idx}"):
            st.info("Edit: remove & re-add (simple editor not implemented)")

    # CSV download
    csv = pd.DataFrame(st.session_state.orders).to_csv(index=False).encode("utf-8")
    st.download_button("Download orders CSV", csv, file_name=f"paper_orders_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv")

else:
    st.info("No paper orders yet.")

# ---------------------------
# Reports
# ---------------------------
st.markdown("---")
st.subheader("Reports")
if st.button("Daily summary (closed orders)"):
    dfc = pd.DataFrame([o for o in st.session_state.orders if o["status"]=="CLOSED"])
    total = dfc["pnl_pct"].sum() if not dfc.empty else 0
    wins = dfc[dfc["pnl_pct"]>0].shape[0] if not dfc.empty else 0
    losses = dfc[dfc["pnl_pct"]<=0].shape[0] if not dfc.empty else 0
    st.json({
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "closed_orders": len(dfc),
        "wins": wins,
        "losses": losses,
        "total_pnl_pct": round(total,4)
    })

st.info("This app performs PAPER trading only unless you integrate real exchange API and implement secure authentication. Use at your own risk.")
