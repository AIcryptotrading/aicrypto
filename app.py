# app.py (FULL) - FIXED: no duplicate widgets, better auto-exec and improved signals
import streamlit as st
import requests, sqlite3, time, math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- Config ----------
BINANCE_REST = ["https://api.binance.com/api/v3", "https://data-api.binance.vision/api/v3"]
DB_PATH = "aicrypto.db"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,NEARUSDT,SEIUSDT"
DEFAULT_INTERVAL = "4h"

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
    # fallback failure
    st.session_state._last_fetch_error = str(last_exc)
    return None

# ---------- DB ----------
def init_db(path=DB_PATH):
    con = sqlite3.connect(path, check_same_thread=False)
    c = con.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT, symbol TEXT, side TEXT, entry REAL, sl REAL, tp REAL,
          size REAL, status TEXT, closed_at TEXT, exit_price REAL, pnl_pct REAL, notes TEXT
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
    """, (order["created_at"], order["symbol"], order["side"], order["entry"],
          order["sl"], order["tp"], order.get("size",1.0), order.get("status","OPEN"), order.get("notes","")))
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

# ---------- Data helpers ----------
def get_price(symbol):
    res = bn_get("ticker/price", params={"symbol": symbol})
    if not res:
        return None
    if isinstance(res, dict) and "price" in res:
        try:
            return float(res["price"])
        except:
            return None
    return None

def get_klines(symbol, interval="4h", limit=500):
    res = bn_get("klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    if not res or not isinstance(res, list):
        return pd.DataFrame()
    df = pd.DataFrame(res, columns=["open_time","open","high","low","close","volume","close_time","quote_vol","n_trades","tb_base","tb_quote","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume","quote_vol","tb_base","tb_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- Indicators ----------
def add_indicators(df):
    df = df.copy()
    if df.empty:
        return df
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # ATR (for SL sizing)
    df["tr0"] = df["high"] - df["low"]
    df["tr1"] = (df["high"] - df["close"].shift()).abs()
    df["tr2"] = (df["low"] - df["close"].shift()).abs()
    df["TR"] = df[["tr0","tr1","tr2"]].max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()
    # Vol MA
    df["VOL20"] = df["volume"].rolling(20).mean()
    return df

# ---------- Trendline helpers ----------
def linear_fit_trend(series):
    y = np.array(series.dropna())
    if len(y) < 2:
        return 0.0, 0.0
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)

def detect_trendline_break(df, lookback=30, threshold_pct=0.005):
    if len(df) < lookback+1:
        return None
    seg = df["close"].tail(lookback)
    m,b = linear_fit_trend(seg)
    idx = len(seg)-1
    expected = m*idx + b
    actual = seg.iloc[-1]
    if expected == 0: return None
    pct = (actual - expected)/expected
    if pct > threshold_pct: return "BREAK_UP"
    if pct < -threshold_pct: return "BREAK_DOWN"
    return None

# ---------- Fib / pivot ----------
def find_recent_swing(df, lookback=200):
    seg = df.tail(lookback)
    if seg.empty: return None, None
    return float(seg["low"].min()), float(seg["high"].max())

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

# ---------- Scoring engine ----------
def score_signal(df, symbol, cfg):
    if df.empty or len(df) < 60:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    price = float(last["close"])
    score = 0
    notes = []

    # MA crossover
    if not np.isnan(df["MA20"].iloc[-2]) and not np.isnan(df["MA50"].iloc[-2]):
        if df["MA20"].iloc[-2] < df["MA50"].iloc[-2] and df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
            score += 2; notes.append("MA20 cross up"); ma_sig="LONG"
        elif df["MA20"].iloc[-2] > df["MA50"].iloc[-2] and df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
            score += 2; notes.append("MA20 cross down"); ma_sig="SHORT"
        else:
            ma_sig=None
    else:
        ma_sig=None

    # RSI
    rsi = last["RSI"]
    if not np.isnan(rsi):
        if rsi < cfg["rsi_oversold"]:
            score += 1; notes.append("RSI oversold")
        elif rsi > cfg["rsi_overbought"]:
            score += 1; notes.append("RSI overbought")

    # MACD
    macd = last["MACD"]; macd_s = last["MACD_SIGNAL"]
    if not np.isnan(macd) and not np.isnan(macd_s):
        if macd > macd_s:
            score += 1; notes.append("MACD>signal"); macd_sig="LONG"
        else:
            macd_sig="SHORT"

    # Volume
    vol = last["volume"]; vol20 = last["VOL20"]
    if not np.isnan(vol20) and vol > vol20*cfg["vol_factor"]:
        score += 2; notes.append("Volume spike")

    # Trendline break
    tb = detect_trendline_break(df, lookback=cfg["trend_lookback"], threshold_pct=cfg["trend_threshold"])
    if tb == "BREAK_UP":
        score += 2; notes.append("Trendline break up"); break_sig="LONG"
    elif tb == "BREAK_DOWN":
        score += 2; notes.append("Trendline break down"); break_sig="SHORT"
    else:
        break_sig=None

    # Fib proximity
    low, high = find_recent_swing(df, lookback=cfg["pivot_lookback"])
    fib_note=None
    if low and high and low < high:
        fibs = fib_levels(low, high)
        for key in ("38.2%","50%","61.8%"):
            lvl = fibs[key]
            if abs(price-lvl)/lvl < cfg["fib_tolerance"]:
                score += 1; notes.append(f"Retest {key}"); fib_note=key

    # Multi-timeframe: check higher timeframe agrees (e.g., daily)
    multi_ok = False
    try:
        higher = get_klines(symbol, interval=cfg["higher_interval"], limit=200)
        if not higher.empty:
            higher = add_indicators(higher)
            # if higher MA20 > MA50 and our MA crossover is up -> confirm
            if ma_sig == "LONG" and higher["MA20"].iloc[-1] > higher["MA50"].iloc[-1]:
                score += 1; notes.append("HTF confirm LONG"); multi_ok=True
            if ma_sig == "SHORT" and higher["MA20"].iloc[-1] < higher["MA50"].iloc[-1]:
                score += 1; notes.append("HTF confirm SHORT"); multi_ok=True
    except Exception:
        pass

    # decide action
    action = None
    if score >= cfg["score_threshold"]:
        # priority break_sig > ma_sig > macd_sig
        if break_sig:
            action = break_sig
        elif ma_sig:
            action = ma_sig
        else:
            action = "LONG" if macd_sig=="LONG" else "SHORT"

    if action:
        # ATR-based sl sizing if requested
        atr = last.get("ATR14", None)
        if cfg["use_atr_sl"] and atr and not np.isnan(atr):
            sl_distance = atr * cfg["atr_multiplier"]
            if action=="LONG":
                sl = round(price - sl_distance, cfg["price_precision"])
                tp = round(price + max(cfg["tp_min_pct"]/100*price, sl_distance*cfg["tp_atr_mult"]), cfg["price_precision"])
            else:
                sl = round(price + sl_distance, cfg["price_precision"])
                tp = round(price - max(cfg["tp_min_pct"]/100*price, sl_distance*cfg["tp_atr_mult"]), cfg["price_precision"])
        else:
            sl_pct = min(cfg["sl_max_pct"]/100.0, 0.2)
            tp_pct = max(cfg["tp_min_pct"]/100.0, 0.05)
            if action == "LONG":
                sl = round(price*(1 - sl_pct), cfg["price_precision"])
                tp = round(price*(1 + tp_pct), cfg["price_precision"])
            else:
                sl = round(price*(1 + sl_pct), cfg["price_precision"])
                tp = round(price*(1 - tp_pct), cfg["price_precision"])

        # compute RR
        try:
            if action=="LONG":
                rr = round((tp-price)/max(price-sl,1e-9), 2)
            else:
                rr = round((price-tp)/max(sl-price,1e-9), 2)
        except:
            rr = None

        return {
            "symbol": symbol, "action": action, "price": price, "sl": sl, "tp": tp,
            "score": score, "rr": rr, "notes": "; ".join(notes), "fib": fib_note
        }
    return None

# ---------- Telegram notify (single place) ----------
def tg_notify(msg):
    token = st.secrets.get("telegram_bot_token") if "telegram_bot_token" in st.secrets else st.session_state.get("tg_token","")
    chat = st.secrets.get("telegram_chat_id") if "telegram_chat_id" in st.secrets else st.session_state.get("tg_chat","")
    if token and chat:
        try:
            SESSION.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id":chat,"text":msg}, timeout=5)
        except Exception as e:
            st.warning(f"Telegram notify failed: {e}")

# ---------- UI ----------
st.set_page_config(layout="wide", page_title="AI Crypto Trader — Fixed")
st.title("🤖 AI Crypto Trader — Fixed (no duplicate widgets)")

# Sidebar: create all widgets once (move them out of loops)
with st.sidebar:
    st.header("Scan Settings")
    symbols_text = st.text_area("Symbols (comma separated)", DEFAULT_SYMBOLS, height=170, key="symbols_input")
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    interval = st.selectbox("Detail interval", ["15m","1h","4h","1d"], index=2)
    higher_interval = st.selectbox("Higher TF for confirmation", ["1d","12h","4h","1h"], index=0)
    auto_scan = st.checkbox("Auto-scan", value=True, key="auto_scan")
    scan_interval = st.number_input("Auto-scan interval (sec)", value=900, min_value=30, key="scan_interval")
    st.markdown("---")
    st.header("Scoring")
    score_threshold = st.number_input("Score threshold", 1, 10, value=4, key="score_threshold")
    vol_factor = st.number_input("Volume multiplier", min_value=1.0, value=1.5, step=0.1, key="vol_factor")
    sl_max_pct = st.number_input("SL max %", min_value=0.5, value=5.0, key="sl_max_pct")
    tp_min_pct = st.number_input("TP min %", min_value=1.0, value=10.0, key="tp_min_pct")
    rsi_oversold = st.number_input("RSI oversold", min_value=1, max_value=50, value=30, key="rsi_oversold")
    rsi_overbought = st.number_input("RSI overbought", min_value=50, max_value=99, value=70, key="rsi_overbought")
    trend_lookback = st.number_input("Trendline lookback (bars)", min_value=10, value=30, key="trend_lookback")
    trend_threshold = st.number_input("Trend break threshold %", min_value=0.001, value=0.005, step=0.001, key="trend_threshold")
    pivot_lookback = st.number_input("Pivot lookback (bars)", min_value=20, value=200, key="pivot_lookback")
    fib_tol = st.number_input("Fib tolerance (pct)", min_value=0.001, value=0.012, step=0.001, key="fib_tol")
    price_precision = st.number_input("Price precision (decimals)", min_value=2, value=6, key="price_precision")
    st.markdown("---")
    st.header("ATR SL (optional)")
    use_atr_sl = st.checkbox("Use ATR for SL", value=False, key="use_atr_sl")
    atr_multiplier = st.number_input("ATR multiplier for SL", min_value=0.5, value=1.5, key="atr_mult")
    tp_atr_mult = st.number_input("TP min in ATR multiples", min_value=0.5, value=2.0, key="tp_atr_mult")
    st.markdown("---")
    st.header("Auto-execution & Alerts")
    # create this checkbox once (prevent duplicate elements)
    auto_exec = st.checkbox("Auto-exec paper orders", value=False, key="auto_exec")
    tg_token_input = st.text_input("Telegram bot token (or set in secrets)", value="", key="tg_token_input")
    tg_chat_input = st.text_input("Telegram chat id (or set in secrets)", value="", key="tg_chat_input")
    # store telegram token/chat into session_state (so tg_notify reads it)
    if tg_token_input:
        st.session_state["tg_token"] = tg_token_input
    if tg_chat_input:
        st.session_state["tg_chat"] = tg_chat_input

# Build cfg dict
cfg = {
    "score_threshold": int(st.session_state.get("score_threshold", score_threshold)),
    "vol_factor": float(st.session_state.get("vol_factor", vol_factor)),
    "sl_max_pct": float(st.session_state.get("sl_max_pct", sl_max_pct)),
    "tp_min_pct": float(st.session_state.get("tp_min_pct", tp_min_pct)),
    "rsi_oversold": int(st.session_state.get("rsi_oversold", rsi_oversold)),
    "rsi_overbought": int(st.session_state.get("rsi_overbought", rsi_overbought)),
    "trend_lookback": int(st.session_state.get("trend_lookback", trend_lookback)),
    "trend_threshold": float(st.session_state.get("trend_threshold", trend_threshold)),
    "pivot_lookback": int(st.session_state.get("pivot_lookback", pivot_lookback)),
    "fib_tolerance": float(st.session_state.get("fib_tol", fib_tol)),
    "price_precision": int(st.session_state.get("price_precision", price_precision)),
    "higher_interval": higher_interval,
    "vol_factor": float(vol_factor),
    "score_threshold": int(score_threshold),
    "use_atr_sl": bool(st.session_state.get("use_atr_sl", use_atr_sl)),
    "atr_multiplier": float(st.session_state.get("atr_mult", atr_multiplier)),
    "tp_atr_mult": float(st.session_state.get("tp_atr_mult", tp_atr_mult)),
}

# ---------- Market Watch ----------
st.subheader("Market Watch")
cols = st.columns(3)
mw = []
for i,sym in enumerate(symbols):
    price = get_price(sym)
    if price is None:
        display = "N/A"
    else:
        display = f"{price:.8f}"
    cols[i % 3].metric(sym, display)

# ---------- Scan: manual trigger ----------
st.subheader("Scan / Suggestions")
if "last_scan" not in st.session_state: st.session_state["last_scan"] = 0
if "scan_results" not in st.session_state: st.session_state["scan_results"] = []
if st.button("Run manual scan now", key="btn_manual_scan"):
    st.session_state["last_scan"] = 0  # force next loop

# Auto-scan runner (non-blocking via experimental_autorefresh)
now_ts = time.time()
if auto_scan and (now_ts - st.session_state["last_scan"] > float(st.session_state.get("scan_interval", scan_interval)) or st.session_state["last_scan"]==0):
    st.session_state["last_scan"] = now_ts
    st.session_state["scan_results"] = []
    # scan all symbols (careful with API rate)
    for sym in symbols:
        df = get_klines(sym, interval=interval, limit=400)
        try:
            sug = score_signal(df, sym, cfg)
            if sug:
                st.session_state["scan_results"].append(sug)
        except Exception as e:
            # don't create widgets here; only log
            st.warning(f"Scan error {sym}: {e}")

# Show scan results
if st.session_state["scan_results"]:
    st.table(pd.DataFrame(st.session_state["scan_results"]))
    # Auto-exec logic: if enabled, attempt to create paper order (no duplicate open same symbol+side)
    if auto_exec:
        for sug in st.session_state["scan_results"]:
            # check duplicate open of same symbol & side
            cnt = DB.execute("SELECT count(*) FROM orders WHERE symbol=? AND side=? AND status='OPEN'", (sug["symbol"], sug["action"])).fetchone()[0]
            if cnt == 0:
                order = {
                    "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sug["symbol"],
                    "side": sug["action"],
                    "entry": sug["price"],
                    "sl": sug["sl"],
                    "tp": sug["tp"],
                    "size": 1.0,
                    "status": "OPEN",
                    "notes": sug.get("notes","")
                }
                oid = db_insert(order)
                msg = f"[Paper] {sug['symbol']} {sug['action']} @ {sug['price']} SL:{sug['sl']} TP:{sug['tp']} score:{sug['score']}"
                tg_notify(msg)
                st.info(f"Auto-created paper order id={oid} for {sug['symbol']} {sug['action']}")
else:
    st.info("No suggestions found.")

# ---------- Chart & analysis ----------
st.subheader("Chart & Analysis")
chart_sym = st.selectbox("Chart symbol", symbols, index=0, key="chart_sym")
dfc = get_klines(chart_sym, interval=interval, limit=500)
if dfc.empty:
    st.error("Cannot fetch chart data for " + chart_sym)
else:
    dfc = add_indicators(dfc)
    fig = go.Figure(data=[go.Candlestick(x=dfc["open_time"], open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name=chart_sym)])
    fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA20"], name="MA20", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA50"], name="MA50", line=dict(color="orange", width=1)))
    st.plotly_chart(fig, use_container_width=True)

# ---------- Paper Orders Manager ----------
st.subheader("Paper Orders (DB)")
orders_df = db_all()
if not orders_df.empty:
    st.dataframe(orders_df)
else:
    st.info("No orders yet.")

# Periodically check open orders for TP/SL hits (safe small number of queries)
open_df = db_all_open()
if not open_df.empty:
    for idx, row in open_df.iterrows():
        current = get_price(row["symbol"]) or row["entry"]
        if row["side"] == "LONG":
            pnl = (current - row["entry"]) / row["entry"] * 100.0
            if current >= row["tp"]:
                db_close(row["id"], current, pnl)
                tg_notify(f"Paper TP hit: {row['symbol']} LONG @ {current:.8f}")
            elif current <= row["sl"]:
                db_close(row["id"], current, pnl)
                tg_notify(f"Paper SL hit: {row['symbol']} LONG @ {current:.8f}")
        else:
            pnl = (row["entry"] - current) / row["entry"] * 100.0
            if current <= row["tp"]:
                db_close(row["id"], current, pnl)
                tg_notify(f"Paper TP hit: {row['symbol']} SHORT @ {current:.8f}")
            elif current >= row["sl"]:
                db_close(row["id"], current, pnl)
                tg_notify(f"Paper SL hit: {row['symbol']} SHORT @ {current:.8f}")

# Export CSV
st.subheader("Export")
if st.button("Export orders to CSV"):
    all_df = db_all()
    csv = all_df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="aicrypto_orders.csv", mime="text/csv")

# Auto refresh
if auto_scan:
    st.experimental_autorefresh(interval=int(st.session_state.get("scan_interval", scan_interval))*1000, key="auto_refresh")
