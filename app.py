# app.py
"""
AI Crypto Trading — Robust Streamlit app using Binance public REST (data-api) for paper-trade.
Features:
- Market watch (many symbols)
- Auto-scan with robust requests (retry/backoff + fallback endpoint)
- Signal engine: RSI / MA crossover / Pivot+Fibonacci breakout+retest + Volume filter
- Signal scoring, suggestions, auto paper-execute (create paper orders)
- Paper order manager with auto TP/SL, CSV export
- Optional Telegram notifications (configure via st.secrets)
- Safe: NO live trading (disabled by default)
"""
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import io
import math
import logging

# ------------- CONFIG -------------
# Use Binance data-api (mirror) which is more cloud-friendly
BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3",  # mirror endpoint
    "https://api.binance.com/api/v3"           # fallback
]

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,NEARUSDT,SEIUSDT"
DEFAULT_INTERVAL = "4h"
AUTO_SCAN_DEFAULT = 900  # seconds

# ------------- Logging -------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aicrypto")

# ------------- HTTP session with retries -------------
def make_session(total_retries=4, backoff_factor=0.8, status_forcelist=(429,500,502,503,504)):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor, status_forcelist=list(status_forcelist))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()

# ------------- Helpers: Binance REST with fallback -------------
def bn_get(path, params=None, timeout=10):
    """GET with fallback over BINANCE_ENDPOINTS"""
    last_err = None
    for base in BINANCE_ENDPOINTS:
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            logger.debug(f"bn_get failed for {url} -> {e}")
            time.sleep(0.2)
    logger.error(f"bn_get all endpoints failed: {last_err}")
    return None

def get_price(symbol):
    res = bn_get("ticker/price", params={"symbol": symbol})
    if res and "price" in res:
        try:
            return float(res["price"])
        except:
            return None
    return None

def get_klines(symbol, interval="4h", limit=500):
    res = bn_get("klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    if not res:
        return None
    # columns per Binance API
    df = pd.DataFrame(res, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------- Indicators & analysis helpers -------------
def add_indicators(df):
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
    """Return list of (idx, 'H'/'L', price)"""
    pivots = []
    n = len(df)
    for i in range(left, n-right):
        is_high = True
        is_low = True
        for j in range(1, left+1):
            if not (df['high'].iloc[i] > df['high'].iloc[i-j]): is_high=False
            if not (df['high'].iloc[i] > df['high'].iloc[i+j]): is_high=False
            if not (df['low'].iloc[i] < df['low'].iloc[i-j]): is_low=False
            if not (df['low'].iloc[i] < df['low'].iloc[i+j]): is_low=False
        if is_high: pivots.append((i,'H',df['high'].iloc[i]))
        if is_low: pivots.append((i,'L',df['low'].iloc[i]))
    return pivots

def fib_levels(low, high):
    diff = high - low
    return {
        "0%": high,
        "23.6%": high - 0.236*diff,
        "38.2%": high - 0.382*diff,
        "50%": high - 0.5*diff,
        "61.8%": high - 0.618*diff,
        "100%": low
    }

def slope_of_last(series, window=60):
    s = series.tail(window)
    if len(s) < 2: return 0.0
    x = np.arange(len(s))
    m, b = np.polyfit(x, s.values, 1)
    return float(m)

# ------------- Session state init -------------
if "orders" not in st.session_state:
    st.session_state.orders = []  # list of dicts
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "last_scan_at" not in st.session_state:
    st.session_state.last_scan_at = 0
if "auto_execute" not in st.session_state:
    st.session_state.auto_execute = False
if "telegram_enabled" not in st.session_state:
    st.session_state.telegram_enabled = False

# ------------- Streamlit UI -------------
st.set_page_config(page_title="AI Crypto Trading (Robust)", layout="wide")
st.title("AI Crypto Trading — Robust Paper-trade (Binance REST)")

# Sidebar controls
st.sidebar.header("Settings")
symbols_text = st.sidebar.text_area("Symbols (comma separated)", DEFAULT_SYMBOLS, height=160)
symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
chart_interval = st.sidebar.selectbox("Chart interval (detail)", ["15m","1h","4h","1d"], index=2)
auto_scan = st.sidebar.checkbox("Auto-scan", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", min_value=30, value=AUTO_SCAN_DEFAULT, step=30)
score_threshold = st.sidebar.number_input("Signal score threshold (1-10)", min_value=1, max_value=10, value=4)
volume_factor = st.sidebar.number_input("Volume multiplier (breakout filter)", min_value=1.0, value=1.5, step=0.1)
sl_max_pct = st.sidebar.number_input("SL max %", min_value=0.5, value=5.0, step=0.5)
tp_min_pct = st.sidebar.number_input("TP min %", min_value=1.0, value=10.0, step=1.0)
st.sidebar.markdown("---")

# Telegram (optional)
st.sidebar.subheader("Telegram notify (optional)")
tg_token = st.secrets.get("telegram_bot_token") if "telegram_bot_token" in st.secrets else st.sidebar.text_input("Bot token (or save in Secrets)", value="")
tg_chat = st.secrets.get("telegram_chat_id") if "telegram_chat_id" in st.secrets else st.sidebar.text_input("Chat ID (or save in Secrets)", value="")
if tg_token and tg_chat:
    st.session_state.telegram_enabled = True
else:
    st.session_state.telegram_enabled = False

# Execution mode toggles
st.sidebar.markdown("---")
st.sidebar.subheader("Execution mode")
auto_exec_checkbox = st.sidebar.checkbox("Auto-execute suggestions (paper-trade)", value=False)
st.session_state.auto_execute = auto_exec_checkbox
st.sidebar.markdown("LIVE trading is disabled in this app for safety. To enable real orders you must implement secure backend and not store API keys in public cloud.")

# Time display
st.sidebar.text("System UTC: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

# Top area: market watch
st.subheader("Market Watch")
cols = st.columns([3,1])
with cols[0]:
    # fetch prices
    price_rows = []
    for s in symbols:
        try:
            p = get_price(s)
            price_rows.append({"symbol": s, "price": round(p,8) if p else "err"})
        except Exception:
            price_rows.append({"symbol": s, "price": "err"})
    st.table(pd.DataFrame(price_rows))

with cols[1]:
    if st.button("Manual scan now"):
        st.session_state.scan_results = []
        st.session_state.last_scan_at = 0  # force scan immediately in next block
        st.experimental_rerun()

# ------------- Signal scanner -------------
def signal_engine_for_symbol(symbol, interval=chart_interval):
    """Return suggestion dict or None"""
    df = get_klines(symbol, interval=interval, limit=500)
    if df is None or df.empty:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    price_now = get_price(symbol) or float(last["close"])
    score = 0
    notes = []
    # RSI extremes
    rsi = float(last.get("RSI") if not pd.isna(last.get("RSI")) else np.nan)
    if not math.isnan(rsi):
        if rsi < 28:
            score += 2; notes.append("RSI <28 (oversold)")
        elif rsi > 72:
            score += 2; notes.append("RSI >72 (overbought)")
    # MA crossover
    if "MA20" in df.columns and "MA50" in df.columns and not pd.isna(df["MA20"].iloc[-2]):
        prev_ma20, prev_ma50 = df["MA20"].iloc[-2], df["MA50"].iloc[-2]
        cur_ma20, cur_ma50 = df["MA20"].iloc[-1], df["MA50"].iloc[-1]
        if prev_ma20 < prev_ma50 and cur_ma20 > cur_ma50:
            score += 2; notes.append("MA20>MA50 crossover")
            ma_signal = "LONG"
        elif prev_ma20 > prev_ma50 and cur_ma20 < cur_ma50:
            score += 2; notes.append("MA20<MA50 crossover")
            ma_signal = "SHORT"
        else:
            ma_signal = None
    else:
        ma_signal = None
    # pivot/fib & breakout
    pivots = find_pivots(df, left=3, right=3)
    # compute recent high/low
    recent_high = df["high"].iloc[-21:-1].max()
    recent_low = df["low"].iloc[-21:-1].min()
    breakout = last["close"] > recent_high
    breakdown = last["close"] < recent_low
    vol = float(last["volume"])
    vol_ma = float(last.get("vol_ma20") if not pd.isna(last.get("vol_ma20")) else np.nan)
    vol_ok = False
    if not math.isnan(vol_ma) and vol_ma > 0 and vol > vol_ma * volume_factor:
        vol_ok = True
        score += 1; notes.append("Volume spike")
    # breakout scoring
    if breakout:
        if vol_ok:
            score += 2; notes.append("Breakout above recent high with vol")
        else:
            score += 1; notes.append("Breakout above recent high (low vol)")
        breakout_signal = "LONG"
    elif breakdown:
        if vol_ok:
            score += 2; notes.append("Breakdown below recent low with vol")
        else:
            score += 1; notes.append("Breakdown below recent low (low vol)")
        breakout_signal = "SHORT"
    else:
        breakout_signal = None
    # fib retest if swing found
    # find last low/high pair
    swings = {"L":[], "H":[]}
    for idx, t, price in pivots:
        swings[t].append((idx, price))
    if swings["L"] and swings["H"]:
        last_low_idx, last_low_price = swings["L"][-1]
        last_high_idx, last_high_price = swings["H"][-1]
        if last_low_idx < last_high_idx:
            lowp, highp = last_low_price, last_high_price
        else:
            lowp, highp = last_high_price, last_low_price
        if lowp < highp:
            fibs = fib_levels(lowp, highp)
            # if close near key fib levels
            for key in ("38.2%","50%","61.8%"):
                lvl = fibs[key]
                if abs(price_now - lvl)/lvl < 0.01:
                    score += 1; notes.append(f"Retest fib {key}")
    # slope trend
    slope = slope_of_last(df["close"].tail(120), window=60) if len(df)>=60 else slope_of_last(df["close"])
    if slope > 0:
        notes.append("Trend slope +")
    elif slope < 0:
        notes.append("Trend slope -")
    # decide final action
    action = None
    # combine MA and breakout: require either breakout or MA crossover plus score threshold
    if score >= score_threshold:
        # prefer breakout if present
        if breakout_signal:
            action = breakout_signal
        elif ma_signal:
            action = ma_signal
        else:
            # fallback: if RSI strong extreme:
            if rsi < 28:
                action = "LONG"
            elif rsi > 72:
                action = "SHORT"
    # prepare SL/TP conservative
    if action:
        entry = price_now
        if action == "LONG":
            sl = round(entry * (1 - min(0.03, sl_max_pct/100.0)), 8)
            tp = round(entry * (1 + max(tp_min_pct/100.0, 0.10)), 8)
        else:
            sl = round(entry * (1 + min(0.03, sl_max_pct/100.0)), 8)
            tp = round(entry * (1 - max(tp_min_pct/100.0, 0.10)), 8)
        rr = None
        try:
            if action == "LONG":
                rr = round((tp - entry) / max(entry - sl, 1e-9), 2)
            else:
                rr = round((entry - tp) / max(sl - entry, 1e-9), 2)
        except:
            rr = None
        return {
            "symbol": symbol,
            "action": action,
            "price": round(entry,8),
            "sl": sl,
            "tp": tp,
            "score": score,
            "rr": rr,
            "notes": "; ".join(notes)
        }
    return None

# ------------- Scanning orchestration -------------
def run_scan_all():
    results = []
    for s in symbols:
        try:
            r = signal_engine_for_symbol(s, interval=chart_interval)
            if r:
                results.append(r)
        except Exception as e:
            logger.exception("scan symbol error: %s %s", s, e)
    return results

# auto-scan trigger based on interval and on manual button
if (auto_scan and (time.time() - st.session_state.last_scan_at > scan_interval)) or st.session_state.last_scan_at == 0:
    st.session_state.last_scan_at = time.time()
    st.session_state.scan_results = run_scan_all()
    if st.session_state.scan_results:
        # auto-execute if enabled
        if st.session_state.auto_execute:
            for sug in st.session_state.scan_results:
                # avoid duplicate open paper orders same symbol & side
                exists = any(o for o in st.session_state.orders if o["symbol"]==sug["symbol"] and o["side"]==sug["action"] and o["status"]=="OPEN")
                if exists:
                    continue
                order = {
                    "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sug["symbol"],
                    "side": sug["action"],
                    "entry": float(sug["price"]),
                    "sl": float(sug["sl"]),
                    "tp": float(sug["tp"]),
                    "size": 1.0,
                    "status": "OPEN",
                    "closed_at": None,
                    "current_price": float(sug["price"]),
                    "pnl_pct": 0.0,
                    "notes": sug["notes"]
                }
                st.session_state.orders.append(order)
                # notify telegram if enabled
                if st.session_state.telegram_enabled and tg_token and tg_chat:
                    try:
                        txt = f"Paper order created: {order['symbol']} {order['side']} @ {order['entry']} SL {order['sl']} TP {order['tp']}"
                        SESSION.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", json={"chat_id":tg_chat,"text":txt}, timeout=5)
                    except Exception as e:
                        logger.warning("Telegram notify failed: %s", e)

# ------------- Show suggestions -------------
st.markdown("---")
st.subheader("Suggestions")
if st.session_state.scan_results:
    df_sug = pd.DataFrame(st.session_state.scan_results)
    st.table(df_sug[["symbol","action","price","sl","tp","score","rr"]])
    for i, r in enumerate(st.session_state.scan_results):
        cols = st.columns([3,1])
        with cols[0]:
            st.write(f"**{r['symbol']}** → {r['action']} @ {r['price']}  |  score {r['score']}  |  RR {r['rr']}")
            st.caption(r["notes"])
        with cols[1]:
            if st.button(f"Add paper {i}"):
                exists = any(o for o in st.session_state.orders if o["symbol"]==r["symbol"] and o["side"]==r["action"] and o["status"]=="OPEN")
                if exists:
                    st.warning("Similar open order exists")
                else:
                    st.session_state.orders.append({
                        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": r["symbol"],
                        "side": r["action"],
                        "entry": float(r["price"]),
                        "sl": float(r["sl"]),
                        "tp": float(r["tp"]),
                        "size": 1.0,
                        "status": "OPEN",
                        "closed_at": None,
                        "current_price": float(r["price"]),
                        "pnl_pct": 0.0,
                        "notes": r["notes"]
                    })
                    st.success("Paper order added")

else:
    st.info("No suggestions found (try manual scan or adjust threshold/volume filter).")

# ------------- Chart & analysis -------------
st.markdown("---")
st.subheader("Chart & Analysis")
chart_sym = st.selectbox("Select symbol", symbols, index=0)
dfc = get_klines(chart_sym, interval=chart_interval, limit=500)
if dfc is None:
    st.error("Cannot fetch chart data for " + chart_sym)
else:
    dfc = add_indicators(dfc)
    fig = go.Figure(data=[go.Candlestick(x=dfc["open_time"], open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name=chart_sym)])
    if "MA20" in dfc.columns:
        fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA20"], name="MA20"))
    if "MA50" in dfc.columns:
        fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA50"], name="MA50"))
    pivs = find_pivots(dfc, left=3, right=3)
    for p in pivs[-40:]:
        idx, t, price = p
        tm = dfc["open_time"].iloc[idx]
        if t == 'H':
            fig.add_trace(go.Scatter(x=[tm], y=[price], mode="markers", marker=dict(color="red", size=7), name="PivotH", showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[tm], y=[price], mode="markers", marker=dict(color="green", size=7), name="PivotL", showlegend=False))
    # draw last fib if possible
    lows = [p for p in pivs if p[1]=='L']
    highs = [p for p in pivs if p[1]=='H']
    if lows and highs:
        last_low = lows[-1][2]
        last_high = highs[-1][2]
        lowp, highp = (last_low, last_high) if last_low < last_high else (last_high, last_low)
        if lowp < highp:
            fibs = fib_levels(lowp, highp)
            for k,v in fibs.items():
                fig.add_hline(y=v, line=dict(color="yellow" if k in ("38.2%","50%","61.8%") else "gray", width=1), annotation_text=k, annotation_position="top left")
    fig.update_layout(height=640, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ------------- Paper order manager -------------
st.markdown("---")
st.subheader("Paper Orders (manager)")
# refresh order prices and auto TP/SL
def refresh_orders():
    for i,o in enumerate(st.session_state.orders):
        if o["status"] == "CLOSED": continue
        cp = get_price(o["symbol"]) or o.get("current_price", o["entry"])
        st.session_state.orders[i]["current_price"] = round(cp,8)
        # PnL %
        if o["side"] == "LONG":
            pnl = (cp - o["entry"]) / o["entry"] * 100
        else:
            pnl = (o["entry"] - cp) / o["entry"] * 100
        st.session_state.orders[i]["pnl_pct"] = round(pnl,4)
        # auto close
        if o["status"] == "OPEN":
            if o["side"] == "LONG":
                if cp >= o["tp"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    if st.session_state.telegram_enabled and tg_token and tg_chat:
                        try:
                            SESSION.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", json={"chat_id":tg_chat,"text":f"TP hit (paper): {o['symbol']} LONG @ {cp}"}, timeout=5)
                        except: pass
                elif cp <= o["sl"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            else:
                if cp <= o["tp"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                elif cp >= o["sl"]:
                    st.session_state.orders[i]["status"] = "CLOSED"
                    st.session_state.orders[i]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

refresh_orders()

if st.session_state.orders:
    df_orders = pd.DataFrame(st.session_state.orders)
    st.dataframe(df_orders[["created_at","symbol","side","entry","current_price","pnl_pct","sl","tp","status","closed_at","notes"]], use_container_width=True)
    # actions
    for idx,o in enumerate(list(st.session_state.orders)):
        c1,c2,c3 = st.columns([3,1,1])
        c1.write(f"{o['symbol']} | {o['side']} | entry {o['entry']} | status {o['status']}")
        if c2.button(f"Close {idx}"):
            st.session_state.orders[idx]["status"] = "CLOSED"
            st.session_state.orders[idx]["closed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            st.experimental_rerun()
        if c3.button(f"Delete {idx}"):
            st.session_state.orders.pop(idx)
            st.experimental_rerun()
    # download CSV
    csv = pd.DataFrame(st.session_state.orders).to_csv(index=False).encode("utf-8")
    st.download_button("Download orders CSV", csv, file_name=f"paper_orders_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv")
else:
    st.info("No paper orders yet")

# ------------- Reports -------------
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
        "closed_count": len(closed),
        "wins": wins, "losses": losses, "total_pnl_pct": round(total,4)
    })

# auto refresh UI (scan interval)
if auto_scan:
    st.experimental_autorefresh(interval=scan_interval*1000, key="auto_scan")

st.info("App runs PAPER-trade only by default. For live trading integration you must implement secure backend & API key management (NOT recommended on Streamlit Cloud).")
