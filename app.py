# app.py
"""
AI Crypto Trader — Streamlit app (Binance REST)
Features:
- Market watch (Binance REST)
- Candlestick chart with auto trendline, fib, volume (Plotly)
- Signal scanner: break trend, MA crossover, RSI, volume spike
- Paper-trade DB (SQLite), auto-exec paper orders optional
- Telegram alerts (optional via Streamlit Secrets or sidebar)
Note: Paper-trade only. For production/live trading add secure storage and order signing.
"""
import streamlit as st
import requests, time, math, sqlite3, io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil import tz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------
# Configuration
# -------------------------
BINANCE_BASES = ["https://api.binance.com/api/v3", "https://data-api.binance.vision/api/v3"]
DB_PATH = "aicrypto.db"

# -------------------------
# HTTP session with retry
# -------------------------
def make_session(total_retries=3, backoff=0.5):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

SESSION = make_session()

def bn_get(path, params=None, timeout=8):
    last_exc = None
    for base in BINANCE_BASES:
        try:
            url = base.rstrip("/") + "/" + path.lstrip("/")
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
    # record last error to session for debugging
    st.session_state["_last_fetch_error"] = str(last_exc)
    return None

# -------------------------
# DB (SQLite) for paper orders
# -------------------------
def init_db(path=DB_PATH):
    con = sqlite3.connect(path, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
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
    """, (order["created_at"], order["symbol"], order["side"], order["entry"],
          order["sl"], order["tp"], order.get("size",1.0), order.get("status","OPEN"), order.get("notes","")))
    DB.commit()
    return cur.lastrowid

def db_all():
    return pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC", DB)

def db_all_open():
    try:
        return pd.read_sql_query("SELECT * FROM orders WHERE status='OPEN' ORDER BY id DESC", DB)
    except Exception:
        return pd.DataFrame()

def db_close(order_id, exit_price, pnl_pct):
    cur = DB.cursor()
    cur.execute("UPDATE orders SET status='CLOSED', closed_at=?, exit_price=?, pnl_pct=? WHERE id=?",
                (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), exit_price, pnl_pct, order_id))
    DB.commit()

# -------------------------
# Binance helpers
# -------------------------
def get_price(symbol):
    r = bn_get("ticker/price", params={"symbol": symbol})
    if not r:
        return None
    try:
        return float(r.get("price", None))
    except:
        return None

def get_klines(symbol, interval="4h", limit=500):
    r = bn_get("klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    if not r or not isinstance(r, list):
        return pd.DataFrame()
    df = pd.DataFrame(r, columns=["open_time","open","high","low","close","volume","close_time","quote_vol","n_trades","tb_base","tb_quote","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume","quote_vol","tb_base","tb_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -------------------------
# Indicators & utilities
# -------------------------
def add_indicators(df):
    df = df.copy()
    if df.empty: return df
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
    # ATR
    df["tr0"] = df["high"] - df["low"]
    df["tr1"] = (df["high"] - df["close"].shift()).abs()
    df["tr2"] = (df["low"] - df["close"].shift()).abs()
    df["TR"] = df[["tr0","tr1","tr2"]].max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()
    df["VOL20"] = df["volume"].rolling(20).mean()
    return df

def linear_trend(series):
    y = np.array(series.dropna())
    if len(y) < 2:
        return 0.0, 0.0
    x = np.arange(len(y))
    m,b = np.polyfit(x,y,1)
    return float(m), float(b)

def detect_trend_break(df, lookback=30, threshold_pct=0.005):
    if df.shape[0] < lookback+1: return None
    seg = df["close"].tail(lookback)
    m,b = linear_trend(seg)
    idx = len(seg)-1
    expected = m*idx + b
    actual = seg.iloc[-1]
    if expected == 0: return None
    pct = (actual - expected)/expected
    if pct > threshold_pct: return "BREAK_UP"
    if pct < -threshold_pct: return "BREAK_DOWN"
    return None

def find_swing_low_high(df, lookback=200):
    seg = df.tail(lookback)
    if seg.empty: return None, None
    return float(seg["low"].min()), float(seg["high"].max())

def fib_levels(low, high):
    d = high - low
    return {
        "0%": high,
        "23.6%": high - 0.236*d,
        "38.2%": high - 0.382*d,
        "50%": high - 0.5*d,
        "61.8%": high - 0.618*d,
        "100%": low
    }

# -------------------------
# Scoring / signal engine
# -------------------------
def score_and_signal(df, symbol, cfg):
    """
    Return dict: {symbol, action(LONG/SHORT), entry, sl, tp, score, notes} or None
    """
    if df.empty or len(df) < 60:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    price = float(last["close"])
    score = 0
    notes = []

    # MA cross
    ma_sig = None
    if not np.isnan(df["MA20"].iloc[-2]) and not np.isnan(df["MA50"].iloc[-2]):
        prev20, prev50 = df["MA20"].iloc[-2], df["MA50"].iloc[-2]
        cur20, cur50 = df["MA20"].iloc[-1], df["MA50"].iloc[-1]
        if prev20 < prev50 and cur20 > cur50:
            score += 2; notes.append("MA20 x MA50 up"); ma_sig="LONG"
        elif prev20 > prev50 and cur20 < cur50:
            score += 2; notes.append("MA20 x MA50 down"); ma_sig="SHORT"

    # RSI
    rsi = last.get("RSI", None)
    if rsi is not None and not np.isnan(rsi):
        if rsi < cfg["rsi_oversold"]:
            score += 1; notes.append("RSI oversold")
        elif rsi > cfg["rsi_overbought"]:
            score += 1; notes.append("RSI overbought")

    # Volume spike
    vol = last["volume"]; vol20 = last["VOL20"]
    if not np.isnan(vol20) and vol > vol20 * cfg["vol_factor"]:
        score += 2; notes.append("Volume spike")

    # Trend break
    tb = detect_trend_break(df, lookback=cfg["trend_lookback"], threshold_pct=cfg["trend_threshold"])
    if tb == "BREAK_UP":
        score += 3; notes.append("Trend break up"); break_sig="LONG"
    elif tb == "BREAK_DOWN":
        score += 3; notes.append("Trend break down"); break_sig="SHORT"
    else:
        break_sig=None

    # Fib confluence
    low, high = find_swing_low_high(df, lookback=cfg["pivot_lookback"])
    fib_note = None
    if low is not None and high is not None and high > low:
        fibs = fib_levels(low, high)
        for key in ["38.2%","50%","61.8%"]:
            lvl = fibs[key]
            if abs(price-lvl)/lvl < cfg["fib_tolerance"]:
                score += 1
                notes.append(f"Fib {key} proximity")
                fib_note = key

    # final action decision
    action = None
    if score >= cfg["score_threshold"]:
        if break_sig:
            action = break_sig
        elif ma_sig:
            action = ma_sig
        else:
            action = "LONG" if (rsi is not None and rsi < 50) else "SHORT"

    if not action:
        return None

    # SL/TP: use ATR if enabled
    atr = last.get("ATR14", None)
    if cfg["use_atr_sl"] and atr and not np.isnan(atr):
        sl_dist = atr * cfg["atr_mult"]
        if action == "LONG":
            sl = round(price - sl_dist, cfg["price_precision"])
            tp = round(price + max(price*cfg["tp_min_pct"]/100, sl_dist*cfg["tp_atr_mult"]), cfg["price_precision"])
        else:
            sl = round(price + sl_dist, cfg["price_precision"])
            tp = round(price - max(price*cfg["tp_min_pct"]/100, sl_dist*cfg["tp_atr_mult"]), cfg["price_precision"])
    else:
        sl = round(price * (1 - cfg["sl_max_pct"]/100) , cfg["price_precision"]) if action=="LONG" else round(price * (1 + cfg["sl_max_pct"]/100), cfg["price_precision"])
        tp = round(price * (1 + cfg["tp_min_pct"]/100), cfg["price_precision"]) if action=="LONG" else round(price * (1 - cfg["tp_min_pct"]/100), cfg["price_precision"])

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

# -------------------------
# Telegram notify (optional)
# -------------------------
def tg_notify(msg):
    token = None
    chat = None
    if "telegram_bot_token" in st.secrets:
        token = st.secrets["telegram_bot_token"]
    else:
        token = st.session_state.get("tg_token", "")
    if "telegram_chat_id" in st.secrets:
        chat = st.secrets["telegram_chat_id"]
    else:
        chat = st.session_state.get("tg_chat", "")
    if token and chat:
        try:
            SESSION.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg}, timeout=5)
        except Exception as e:
            st.warning(f"Telegram notify failed: {e}")

# -------------------------
# Streamlit UI (single widget creation)
# -------------------------
st.set_page_config(layout="wide", page_title="AI Crypto Trader — Binance REST")
st.title("🤖 AI Crypto Trader — Binance REST (Paper-trade)")

# Sidebar widgets (create once)
with st.sidebar:
    st.header("Scan / Settings")
    symbols_text = st.text_area("Symbols (comma separated)", value="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,NEARUSDT,SEIUSDT", height=160, key="symbols")
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    interval = st.selectbox("Detail interval", options=["15m","1h","4h","1d"], index=2, key="interval")
    higher_interval = st.selectbox("Higher TF", options=["1d","12h","4h","1h"], index=0, key="high_int")
    auto_scan = st.checkbox("Auto-scan", value=True, key="auto_scan")
    scan_interval = st.number_input("Auto-scan interval (sec)", value=900, min_value=30, key="scan_interval")
    st.markdown("---")
    st.header("Signal filters / Risk")
    score_threshold = st.number_input("Score threshold", min_value=1, max_value=20, value=4, key="score_threshold")
    vol_factor = st.number_input("Volume multiplier", min_value=1.0, value=1.5, step=0.1, key="vol_factor")
    sl_max_pct = st.number_input("SL max %", min_value=0.5, value=5.0, key="sl_max")
    tp_min_pct = st.number_input("TP min %", min_value=1.0, value=10.0, key="tp_min")
    rsi_oversold = st.number_input("RSI oversold", min_value=1, max_value=50, value=30, key="rsi_oversold")
    rsi_overbought = st.number_input("RSI overbought", min_value=50, max_value=99, value=70, key="rsi_overbought")
    trend_lookback = st.number_input("Trendline lookback (bars)", min_value=10, value=30, key="trend_lookback")
    trend_threshold = st.number_input("Trend break threshold %", value=0.005, step=0.001, key="trend_threshold")
    pivot_lookback = st.number_input("Pivot lookback (bars)", min_value=20, value=200, key="pivot_lookback")
    fib_tol = st.number_input("Fib tolerance (pct)", min_value=0.001, value=0.012, step=0.001, key="fib_tol")
    price_precision = st.number_input("Price precision decimals", min_value=2, max_value=8, value=6, key="price_precision")
    st.markdown("---")
    st.header("ATR SL (optional)")
    use_atr_sl = st.checkbox("Use ATR SL", value=False, key="use_atr_sl")
    atr_mult = st.number_input("ATR multiplier (SL)", min_value=0.5, value=1.5, key="atr_mult")
    tp_atr_mult = st.number_input("TP in ATR multiples", min_value=1.0, value=2.0, key="tp_atr_mult")
    st.markdown("---")
    st.header("Auto-exec & Alerts")
    auto_exec = st.checkbox("Auto-exec paper orders", value=False, key="auto_exec")
    tg_token = st.text_input("Telegram bot token (optional)", value="", key="tg_token")
    tg_chat = st.text_input("Telegram chat id (optional)", value="", key="tg_chat")
    if tg_token:
        st.session_state["tg_token"] = tg_token
    if tg_chat:
        st.session_state["tg_chat"] = tg_chat
    st.markdown("---")
    st.write("Time (local): ", datetime.now(tz=tz.tzlocal()).strftime("%Y-%m-%d %H:%M:%S"))

# Build config dict from sidebar inputs
cfg = {
    "score_threshold": int(st.session_state.get("score_threshold", score_threshold)),
    "vol_factor": float(st.session_state.get("vol_factor", vol_factor)),
    "sl_max_pct": float(st.session_state.get("sl_max", sl_max_pct)),
    "tp_min_pct": float(st.session_state.get("tp_min", tp_min_pct)),
    "rsi_oversold": int(st.session_state.get("rsi_oversold", rsi_oversold)),
    "rsi_overbought": int(st.session_state.get("rsi_overbought", rsi_overbought)),
    "trend_lookback": int(st.session_state.get("trend_lookback", trend_lookback)),
    "trend_threshold": float(st.session_state.get("trend_threshold", trend_threshold)),
    "pivot_lookback": int(st.session_state.get("pivot_lookback", pivot_lookback)),
    "fib_tolerance": float(st.session_state.get("fib_tol", fib_tol)),
    "price_precision": int(st.session_state.get("price_precision", price_precision)),
    "higher_interval": higher_interval,
    "use_atr_sl": bool(st.session_state.get("use_atr_sl", use_atr_sl)),
    "atr_mult": float(st.session_state.get("atr_mult", atr_mult)),
    "tp_atr_mult": float(st.session_state.get("tp_atr_mult", tp_atr_mult)),
    "vol_factor": float(st.session_state.get("vol_factor", vol_factor))
}

# -------------------------
# Main dashboard layout
# -------------------------
# Top: Market Watch
st.subheader("Market Watch")
cols = st.columns(3)
for i, s in enumerate(symbols[:len(symbols)]):
    price = get_price(s)
    text = f"{price:.{cfg['price_precision']}f}" if price else "N/A"
    cols[i % 3].metric(label=s, value=text)

# Scan / suggestions
st.subheader("Scan & Suggestions")
if "last_scan" not in st.session_state: st.session_state["last_scan"] = 0
if "scan_results" not in st.session_state: st.session_state["scan_results"] = []
if st.button("Run manual scan now", key="manual_scan"):
    st.session_state["last_scan"] = 0  # force immediate scan

# Auto-scan
now_ts = time.time()
if auto_scan and (now_ts - st.session_state["last_scan"] > float(st.session_state.get("scan_interval", scan_interval)) or st.session_state["last_scan"]==0):
    st.session_state["last_scan"] = now_ts
    st.session_state["scan_results"] = []
    for sym in symbols:
        df = get_klines(sym, interval=interval, limit=500)
        try:
            sug = score_and_signal(df, sym, cfg)
            if sug:
                st.session_state["scan_results"].append(sug)
        except Exception as e:
            # log but continue
            st.write(f"Scan error {sym}: {e}")

# Show suggestions
if st.session_state["scan_results"]:
    df_sug = pd.DataFrame(st.session_state["scan_results"])
    st.table(df_sug)
    # Auto-exec paper orders if enabled
    if auto_exec:
        for sug in st.session_state["scan_results"]:
            # check duplicate open same symbol+side
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
                msg = f"[PaperOrder] {sug['symbol']} {sug['action']} @ {sug['price']} SL:{sug['sl']} TP:{sug['tp']} score:{sug['score']}"
                tg_notify(msg)
                st.success(f"Auto-created paper order id={oid}: {sug['symbol']} {sug['action']}")
else:
    st.info("No suggestions at the moment. Try manual scan or wait for auto-scan.")

# -------------------------
# Chart & analysis (Plotly)
# -------------------------
st.subheader("Chart & Trendline")
chart_symbol = st.selectbox("Select symbol for chart", symbols, index=0, key="chart_sym")
chart_df = get_klines(chart_symbol, interval=interval, limit=600)
if chart_df.empty:
    st.error("Cannot fetch chart data for " + chart_symbol + ". See _last_fetch_error in session state for details.")
else:
    chart_df = add_indicators(chart_df)
    # compute trendline fit on last N bars
    look_n = int(st.session_state.get("trend_lookback", trend_lookback))
    seg = chart_df.tail(look_n)
    m,b = linear_trend(seg["close"]) if len(seg) >= 2 else (0, seg["close"].iloc[-1] if not seg.empty else 0)
    # compute line values for plotting: extend to full timeframe for a nicer look
    xs = np.arange(len(chart_df))
    trend_y = m*xs + b

    # fib
    low, high = find_swing_low_high(chart_df, lookback=int(st.session_state.get("pivot_lookback", pivot_lookback)))
    fibs = fib_levels(low, high) if low is not None and high is not None and high>low else {}

    fig = go.Figure(data=[go.Candlestick(x=chart_df["open_time"],
                                         open=chart_df["open"], high=chart_df["high"],
                                         low=chart_df["low"], close=chart_df["close"], name=chart_symbol)])
    # overlay MA
    if "MA20" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["open_time"], y=chart_df["MA20"], name="MA20", line=dict(width=1, color="blue")))
    if "MA50" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["open_time"], y=chart_df["MA50"], name="MA50", line=dict(width=1, color="orange")))
    # overlay trendline
    fig.add_trace(go.Scatter(x=chart_df["open_time"], y=trend_y, name="AutoTrend", line=dict(width=1, dash="dash", color="green")))
    # overlay fibs
    for k,v in fibs.items():
        fig.add_hline(y=v, line=dict(dash="dot", width=1), annotation_text=k, annotation_position="top left")

    # volume as bar on secondary axis
    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    fig.update_yaxes(side="right")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Paper order manager
# -------------------------
st.subheader("Paper Orders & Manager")
orders = db_all()
if not orders.empty:
    st.dataframe(orders)
else:
    st.info("No paper orders yet.")

# Check open orders for TP/SL hits (light)
open_orders = db_all_open()
if not open_orders.empty:
    for idx, row in open_orders.iterrows():
        cur = get_price(row["symbol"]) or row["entry"]
        if row["side"] == "LONG":
            pnl = (cur - row["entry"])/row["entry"]*100.0
            if cur >= row["tp"]:
                db_close(row["id"], cur, pnl)
                tg_notify(f"[Paper TP] {row['symbol']} LONG TP hit @ {cur:.6f}")
            elif cur <= row["sl"]:
                db_close(row["id"], cur, pnl)
                tg_notify(f"[Paper SL] {row['symbol']} LONG SL hit @ {cur:.6f}")
        else:
            pnl = (row["entry"] - cur)/row["entry"]*100.0
            if cur <= row["tp"]:
                db_close(row["id"], cur, pnl)
                tg_notify(f"[Paper TP] {row['symbol']} SHORT TP hit @ {cur:.6f}")
            elif cur >= row["sl"]:
                db_close(row["id"], cur, pnl)
                tg_notify(f"[Paper SL] {row['symbol']} SHORT SL hit @ {cur:.6f}")

# -------------------------
# Export / Reporting
# -------------------------
st.subheader("Export & Reports")
if st.button("Export orders CSV"):
    df_export = db_all()
    csv = df_export.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="aicrypto_orders.csv", mime="text/csv")

# Auto refresh if auto_scan enabled
if auto_scan:
    st.experimental_autorefresh(interval=int(scan_interval)*1000, key="refresh")

# Footer: debug info
if st.checkbox("Show debug info (session)", value=False):
    st.write("Last fetch error:", st.session_state.get("_last_fetch_error", ""))
    st.write("Session keys:", list(st.session_state.keys()))
