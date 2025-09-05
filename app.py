# app.py
# AI Crypto Trader - Upgraded (Streamlit)
# - Binance REST (public endpoints)
# - Indicators: MA, RSI, MACD, Vol MA
# - Trendline break detection (linear fit)
# - Fib & pivot checks
# - Signal scoring engine (combine signals)
# - Paper-trade manager (SQLite)
# - Telegram notify optional via Streamlit secrets
# - Chart with trendline & TP/SL markers (plotly)
# Usage: streamlit run app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sqlite3, time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------- CONFIG -----------------
BINANCE_REST = ["https://api.binance.com/api/v3", "https://data-api.binance.vision/api/v3"]
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,NEARUSDT,SEIUSDT"
DEFAULT_INTERVAL = "4h"
DB_PATH = "aicrypto.db"
# ------------------------------------------

# ---------- HTTP session with retry ----------
def make_session(total_retries=3, backoff=0.5):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

SESSION = make_session()

def bn_get(path, params=None, timeout=8):
    last_exc = None
    for base in BINANCE_REST:
        try:
            url = base.rstrip("/") + "/" + path.lstrip("/")
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            # continue to fallback
    # all failed
    st.warning(f"Binance REST fetch failed for {path}: {last_exc}")
    return None

# ------------- Data helpers -------------
def get_price(symbol):
    res = bn_get("ticker/price", params={"symbol": symbol})
    if not res: return None
    if isinstance(res, dict) and res.get("price"):
        try:
            return float(res["price"])
        except:
            return None
    # If invalid symbol, API often returns {"code": -1121, "msg": "..."}
    return None

def get_klines(symbol, interval="4h", limit=500):
    res = bn_get("klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    if not res or not isinstance(res, list):
        return pd.DataFrame()
    df = pd.DataFrame(res, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_vol","n_trades","taker_base","taker_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume","quote_vol","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------- Indicators -------------
def add_indicators(df):
    df = df.copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1/14, adjust=False).mean()
    loss = down.ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # Vol MA
    df["VOL20"] = df["volume"].rolling(20).mean()
    return df

# ------------- Trendline helpers -------------
def linear_fit_trend(series):
    # return slope, intercept for index-based linear fit
    y = np.array(series.dropna())
    if len(y) < 2:
        return 0.0, 0.0
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)

def detect_trendline_break(df, lookback=30, threshold_pct=0.005):
    # Fit trendline to closes over lookback, check last candle vs trendline
    if len(df) < lookback+1:
        return None
    seg = df["close"].tail(lookback)
    m, b = linear_fit_trend(seg)
    # compute expected last value
    last_idx = len(seg)-1
    expected = m*last_idx + b
    actual = seg.iloc[-1]
    pct = (actual - expected) / expected if expected != 0 else 0
    # Decide break direction
    if pct > threshold_pct:
        return "BREAK_UP"
    elif pct < -threshold_pct:
        return "BREAK_DOWN"
    return None

# ------------- Pivot & Fib (simple) -------------
def find_recent_swing(df, lookback=200):
    # simple: highest high and lowest low in lookback window
    seg = df.tail(lookback)
    if seg.empty:
        return None, None
    high = seg["high"].max()
    low = seg["low"].min()
    return low, high

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

# ------------- Signal Engine (scoring) -------------
def score_signal(df, symbol, cfg):
    # returns suggestion dict or None
    if df.empty or len(df) < 50:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    price = float(last["close"])
    score = 0
    notes = []

    # MA crossover bias
    if df["MA20"].iloc[-2] < df["MA50"].iloc[-2] and df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
        score += 2; notes.append("MA20 cross up")
        ma_sig = "LONG"
    elif df["MA20"].iloc[-2] > df["MA50"].iloc[-2] and df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
        score += 2; notes.append("MA20 cross down")
        ma_sig = "SHORT"
    else:
        ma_sig = None

    # RSI extremes
    rsi = last["RSI"]
    if not np.isnan(rsi):
        if rsi < cfg["rsi_oversold"]:
            score += 1; notes.append("RSI oversold")
        elif rsi > cfg["rsi_overbought"]:
            score += 1; notes.append("RSI overbought")

    # MACD momentum
    macd = last["MACD"]; macd_s = last["MACD_SIGNAL"]
    if macd > macd_s:
        score += 1; notes.append("MACD>signal")
        macd_sig = "LONG"
    else:
        macd_sig = "SHORT"

    # Volume spike
    vol = last["volume"]; vol20 = last["VOL20"]
    if not np.isnan(vol20) and vol > vol20 * cfg["vol_factor"]:
        score += 2; notes.append("Volume spike")

    # Trendline break
    tb = detect_trendline_break(df, lookback=cfg["trend_lookback"], threshold_pct=cfg["trend_threshold"])
    if tb == "BREAK_UP":
        score += 2; notes.append("Trendline break up")
        break_sig = "LONG"
    elif tb == "BREAK_DOWN":
        score += 2; notes.append("Trendline break down")
        break_sig = "SHORT"
    else:
        break_sig = None

    # Fibonacci proximity (retest)
    low, high = find_recent_swing(df, lookback=cfg["pivot_lookback"])
    fib_note = None
    if low is not None and high is not None and low < high:
        fibs = fib_levels(low, high)
        for k in ("38.2%","50%","61.8%"):
            lvl = fibs[k]
            if abs(price - lvl)/lvl < cfg["fib_tolerance"]:
                score += 1; notes.append(f"Retest {k}"); fib_note = k

    # Combine signals to decide action
    action = None
    if score >= cfg["score_threshold"]:
        # priority: break_sig > ma_sig > macd_sig
        if break_sig: action = break_sig
        elif ma_sig: action = "LONG" if ma_sig=="LONG" else "SHORT"
        else: action = "LONG" if macd_sig=="LONG" else "SHORT"

    if action:
        sl_pct = min(cfg["sl_max_pct"]/100.0, 0.2)
        tp_pct = max(cfg["tp_min_pct"]/100.0, 0.05)
        if action == "LONG":
            sl = round(price*(1 - sl_pct), cfg["price_precision"])
            tp = round(price*(1 + tp_pct), cfg["price_precision"])
        else:
            sl = round(price*(1 + sl_pct), cfg["price_precision"])
            tp = round(price*(1 - tp_pct), cfg["price_precision"])
        rr = None
        try:
            if action == "LONG":
                rr = round((tp - price) / max(price - sl, 1e-9), 2)
            else:
                rr = round((price - tp) / max(sl - price, 1e-9), 2)
        except:
            rr = None
        return {
            "symbol": symbol,
            "action": action,
            "price": price,
            "sl": sl,
            "tp": tp,
            "score": score,
            "rr": rr,
            "notes": "; ".join(notes),
            "fib": fib_note
        }
    return None

# ------------- Persistence: SQLite -------------
def init_db(path=DB_PATH):
    con = sqlite3.connect(path, check_same_thread=False)
    c = con.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            symbol TEXT,
            side TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            size REAL,
            status TEXT,
            closed_at TEXT,
            exit_price REAL,
            pnl_pct REAL,
            notes TEXT
        )
    """)
    con.commit()
    return con

DB = init_db(DB_PATH)

def db_insert(order):
    cur = DB.cursor()
    cur.execute("""
        INSERT INTO orders (created_at,symbol,side,entry,sl,tp,size,status,notes)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (order["created_at"], order["symbol"], order["side"], order["entry"], order["sl"], order["tp"], order.get("size",1.0), order.get("status","OPEN"), order.get("notes","")))
    DB.commit()
    return cur.lastrowid

def db_all_open():
    return pd.read_sql_query("SELECT * FROM orders WHERE status='OPEN' ORDER BY id DESC", DB)

def db_all():
    return pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC", DB)

def db_close(order_id, exit_price, pnl_pct):
    cur = DB.cursor()
    cur.execute("UPDATE orders SET status='CLOSED', closed_at=?, exit_price=?, pnl_pct=? WHERE id=?",
                (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), exit_price, pnl_pct, order_id))
    DB.commit()

# ------------- Telegram notify -------------
def tg_notify(msg):
    token = st.secrets.get("telegram_bot_token") if "telegram_bot_token" in st.secrets else st.sidebar.text_input("Telegram token (or store in secrets)", value="")
    chat = st.secrets.get("telegram_chat_id") if "telegram_chat_id" in st.secrets else st.sidebar.text_input("Telegram chat id", value="")
    if token and chat:
        try:
            SESSION.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id":chat,"text":msg}, timeout=5)
        except Exception as e:
            st.warning(f"Telegram notify failed: {e}")

# ------------- Streamlit UI -------------
st.set_page_config(layout="wide", page_title="AI Crypto Trading — Upgraded")
st.title("🤖 AI Crypto Trading — Upgraded (Paper-trade)")

# Sidebar controls
st.sidebar.header("Scan Settings")
symbols_text = st.sidebar.text_area("Symbols (comma separated)", DEFAULT_SYMBOLS, height=160)
symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
interval = st.sidebar.selectbox("Detail interval", ["15m","1h","4h","1d"], index=2)
auto_scan = st.sidebar.checkbox("Auto-scan", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", value=900, min_value=30)
score_threshold = st.sidebar.number_input("Score threshold", 1, 10, value=4)
vol_factor = st.sidebar.number_input("Volume multiplier", min_value=1.0, value=1.5, step=0.1)
sl_max_pct = st.sidebar.number_input("SL max %", min_value=0.5, value=5.0)
tp_min_pct = st.sidebar.number_input("TP min %", min_value=1.0, value=10.0)
rsi_oversold = st.sidebar.number_input("RSI oversold", min_value=1, max_value=50, value=30)
rsi_overbought = st.sidebar.number_input("RSI overbought", min_value=50, max_value=99, value=70)
trend_lookback = st.sidebar.number_input("Trendline lookback (bars)", min_value=10, value=30)
trend_thresh = st.sidebar.number_input("Trend break threshold %", min_value=0.001, value=0.005, step=0.001)
pivot_lookback = st.sidebar.number_input("Pivot lookback (bars)", min_value=20, value=200)
fib_tol = st.sidebar.number_input("Fib tolerance (pct)", min_value=0.001, value=0.012, step=0.001)
price_precision = st.sidebar.number_input("Price decimal precision", min_value=2, value=6)

# Put config into dict for engine
cfg = {
    "score_threshold": score_threshold,
    "vol_factor": vol_factor,
    "sl_max_pct": sl_max_pct,
    "tp_min_pct": tp_min_pct,
    "rsi_oversold": rsi_oversold,
    "rsi_overbought": rsi_overbought,
    "trend_lookback": trend_lookback,
    "trend_threshold": trend_thresh,
    "pivot_lookback": pivot_lookback,
    "fib_tolerance": fib_tol,
    "price_precision": int(price_precision)
}

st.sidebar.markdown("---")
st.sidebar.write("Telegram notify (optional) — you can also store in Streamlit Secrets.")
# Telegram text inputs shown earlier by tg_notify() if secrets absent.

# ---------- Market Watch ----------
st.subheader("Market Watch")
cols = st.columns(3)
mw = []
for i,sym in enumerate(symbols):
    price = get_price(sym)
    row = {"symbol": sym, "price": price if price else "N/A"}
    mw.append(row)
    col = cols[i % 3]
    if price:
        col.metric(label=sym, value=f"{price:.8f}")
    else:
        col.metric(label=sym, value="N/A")

# ---------- Scan (manual/auto) ----------
st.subheader("Scan / Suggestions")
if st.button("Run manual scan now"):
    st.session_state.last_scan = 0  # force scan

if "last_scan" not in st.session_state:
    st.session_state.last_scan = 0
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

now = time.time()
if auto_scan and (now - st.session_state.last_scan > scan_interval or st.session_state.last_scan == 0):
    st.session_state.last_scan = now
    suggestions = []
    for sym in symbols:
        df = get_klines(sym, interval=interval, limit=400)
        try:
            sug = score_signal(df, sym, cfg)
            if sug:
                suggestions.append(sug)
        except Exception as e:
            st.warning(f"Scan error {sym}: {e}")
    st.session_state.scan_results = suggestions
    if suggestions:
        for s in suggestions:
            # Auto create paper order if enabled and not duplicate
            if st.sidebar.checkbox("Auto-exec paper orders", value=False):
                existing = DB.execute("SELECT count(*) FROM orders WHERE symbol=? AND side=? AND status='OPEN'", (s["symbol"], s["action"])).fetchone()[0]
                if existing == 0:
                    ord = {
                        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": s["symbol"],
                        "side": s["action"],
                        "entry": s["price"],
                        "sl": s["sl"],
                        "tp": s["tp"],
                        "size": 1.0,
                        "status": "OPEN",
                        "notes": s.get("notes","")
                    }
                    oid = db_insert(ord)
                    tg_notify(f"Paper order created: {s['symbol']} {s['action']} @ {s['price']} SL {s['sl']} TP {s['tp']}")
    else:
        st.info("Scan done: no suggestions")

# show scan results
if st.session_state.scan_results:
    df_sug = pd.DataFrame(st.session_state.scan_results)
    st.table(df_sug[["symbol","action","price","sl","tp","score","rr","notes"]])
else:
    st.info("No current suggestions.")

# ---------- Chart & Trendline ----------
st.subheader("Chart & Trendline")
chart_sym = st.selectbox("Choose symbol for chart", symbols, index=0)
dfc = get_klines(chart_sym, interval=interval, limit=500)
if dfc.empty:
    st.error("Cannot fetch chart data for " + chart_sym)
else:
    dfc = add_indicators(dfc)
    fig = go.Figure(data=[go.Candlestick(x=dfc["open_time"], open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name=chart_sym)])
    # add MA lines
    fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA20"], name="MA20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA50"], name="MA50", line=dict(width=1)))
    # add pivots (simple)
    low, high = find_recent_swing(dfc, lookback=cfg["pivot_lookback"])
    if low is not None and high is not None and low < high:
        fibs = fib_levels(low, high)
        for k,v in fibs.items():
            color = "yellow" if k in ("38.2%","50%","61.8%") else "gray"
            fig.add_hline(y=v, line=dict(color=color, width=1), annotation_text=k, annotation_position="top left")
    # trendline (linear fit on last N)
    tb = detect_trendline_break(dfc, lookback=cfg["trend_lookback"], threshold_pct=cfg["trend_threshold"])
    st.write("Trendline check:", tb)
    # show chart
    fig.update_layout(height=650, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Paper Orders Manager ----------
st.subheader("Paper Orders Manager")
orders_df = db_all()
if not orders_df.empty:
    st.dataframe(orders_df)
else:
    st.info("No orders yet.")

# update open orders status: check price vs TP/SL
open_df = db_all_open()
if not open_df.empty:
    for idx, row in open_df.iterrows():
        current = get_price(row["symbol"]) or row["entry"]
        pnl = 0.0
        if row["side"] == "LONG":
            pnl = (current - row["entry"]) / row["entry"] * 100.0
            if current >= row["tp"]:
                db_close(row["id"], current, pnl)
                tg_notify(f"Paper TP hit: {row['symbol']} LONG @ {current}")
            elif current <= row["sl"]:
                db_close(row["id"], current, pnl)
                tg_notify(f"Paper SL hit: {row['symbol']} LONG @ {current}")
        else:
            pnl = (row["entry"] - current) / row["entry"] * 100.0
            if current <= row["tp"]:
                db_close(row["id"], current, pnl)
                tg_notify(f"Paper TP hit: {row['symbol']} SHORT @ {current}")
            elif current >= row["sl"]:
                db_close(row["id"], current, pnl)
                tg_notify(f"Paper SL hit: {row['symbol']} SHORT @ {current}")

# ---------- Export & Reports ----------
st.subheader("Export / Reports")
if st.button("Export orders to CSV"):
    all_df = db_all()
    csv = all_df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="aicrypto_orders.csv", mime="text/csv")

# Auto refresh when auto_scan enabled
if auto_scan:
    st.experimental_autorefresh(interval=scan_interval*1000, key="auto_scan")
