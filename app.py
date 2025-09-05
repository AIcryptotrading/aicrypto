# app.py
"""
AI Crypto Trading — Robust Streamlit App (Paper-trade)
Features:
 - Binance REST via data-api mirror + retry/backoff
 - Pivot detection (configurable), Fibonacci retrace + multi-TP scaling
 - Volume filter, MA crossover, RSI, RSI divergence basic detection
 - Signal scoring + suggestions
 - Paper orders saved to SQLite (persistence)
 - Telegram notifications (configurable via Streamlit secrets or env)
 - Docker/systemd friendly
 - Safe: NO live trading; paper-execute only
"""
import streamlit as st
import requests, json, sqlite3, os, math, logging, time
import pandas as pd, numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

# ------------- SETTINGS -------------
APP_DB = os.environ.get("AICRYPTO_DB_PATH", "aicrypto.db")
BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3",
    "https://api.binance.com/api/v3"
]
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,NEARUSDT,SEIUSDT"
DEFAULT_INTERVAL = "4h"
AUTO_SCAN_DEFAULT = 900
LOG_LEVEL = logging.INFO

# ------------- Logging -------------
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aicrypto")

# ------------- HTTP session w retry -------------
def make_session(total_retries=4, backoff=0.6):
    s = requests.Session()
    r = Retry(total_retries, backoff_factor=backoff, status_forcelist=[429,500,502,503,504], allowed_methods=["GET","POST"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s
SESSION = make_session()

def bn_get(path, params=None, timeout=10):
    last_err = None
    for base in BINANCE_ENDPOINTS:
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            logger.debug("bn_get fail %s -> %s", url, e)
            time.sleep(0.2)
    logger.error("bn_get all fail: %s", last_err)
    return None

# ------------- Data helpers -------------
def get_price(symbol):
    res = bn_get("ticker/price", params={"symbol": symbol})
    if not res: return None
    try:
        return float(res.get("price"))
    except:
        return None

def get_klines(symbol, interval=DEFAULT_INTERVAL, limit=500):
    res = bn_get("klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    if not res:
        return None
    df = pd.DataFrame(res, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","tb1","tb2","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------- Indicators & analysis -------------
def add_indicators(df):
    df = df.copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    avg_up = up.rolling(14).mean(); avg_down = down.rolling(14).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df

def find_pivots(df, left=4, right=4):
    # stronger pivot detection (configurable left/right)
    pivots = []
    n = len(df)
    for i in range(left, n-right):
        is_high = True; is_low = True
        for j in range(1, left+1):
            if df['high'].iloc[i] <= df['high'].iloc[i-j]: is_high = False
            if df['high'].iloc[i] <= df['high'].iloc[i+j]: is_high = False
            if df['low'].iloc[i] >= df['low'].iloc[i-j]: is_low = False
            if df['low'].iloc[i] >= df['low'].iloc[i+j]: is_low = False
        if is_high: pivots.append((i,'H',df['high'].iloc[i]))
        if is_low: pivots.append((i,'L',df['low'].iloc[i]))
    return pivots

def fib_levels(low, high):
    diff = high - low
    return {"0%":high, "23.6%":high-0.236*diff, "38.2%":high-0.382*diff, "50%":high-0.5*diff, "61.8%":high-0.618*diff, "100%":low}

def slope(series):
    x = np.arange(len(series))
    if len(x) < 2: return 0.0
    m,b = np.polyfit(x, series.values, 1)
    return float(m)

def detect_rsi_divergence(df, lookback=40):
    # basic detection: compare last two price highs/lows vs RSI
    # returns 'bullish','bearish',None
    if len(df) < lookback: return None
    seg = df.tail(lookback)
    highs = seg['high']; lows = seg['low']; rsi = seg['RSI']
    # find top 2 highs
    idxs_high = highs.nlargest(4).index.tolist()
    if len(idxs_high) >= 2:
        i1, i2 = sorted(idxs_high[:2])
        if highs.loc[i1] < highs.loc[i2] and rsi.loc[i1] > rsi.loc[i2]:
            return 'bearish_div'
    idxs_low = lows.nsmallest(4).index.tolist()
    if len(idxs_low) >= 2:
        i1,i2 = sorted(idxs_low[:2])
        if lows.loc[i1] > lows.loc[i2] and rsi.loc[i1] < rsi.loc[i2]:
            return 'bullish_div'
    return None

# ------------- Persistence (SQLite) -------------
def init_db(path=APP_DB):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS orders (
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
        current_price REAL,
        pnl_pct REAL,
        notes TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS logs (ts TEXT, level TEXT, msg TEXT)""")
    conn.commit()
    return conn
DB = init_db(APP_DB)

def db_insert_order(order):
    c = DB.cursor()
    c.execute("INSERT INTO orders (created_at,symbol,side,entry,sl,tp,size,status,closed_at,current_price,pnl_pct,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
              (order["created_at"],order["symbol"],order["side"],order["entry"],order["sl"],order["tp"],order["size"],order["status"],order.get("closed_at"),order.get("current_price",0),order.get("pnl_pct",0),order.get("notes","")))
    DB.commit()
    return c.lastrowid

def db_update_order_status(order_id, status, closed_at=None):
    c = DB.cursor()
    c.execute("UPDATE orders SET status=?, closed_at=? WHERE id=?", (status, closed_at, order_id))
    DB.commit()

def db_list_orders():
    return pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC", DB)

def db_log(level, msg):
    c = DB.cursor()
    c.execute("INSERT INTO logs (ts,level,msg) VALUES (?,?,?)", (datetime.utcnow().isoformat(),level,msg))
    DB.commit()

# ------------- Telegram notify -------------
def tg_notify(token, chat_id, text):
    try:
        SESSION.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id":chat_id,"text":text}, timeout=5)
    except Exception as e:
        logger.warning("tg_notify failed: %s", e)

# ------------- Signal engine -------------
def signal_for_symbol(symbol, interval="4h", config=None):
    """
    Returns suggestion dict or None.
    config: dict {score_threshold, volume_factor, sl_max_pct, tp_min_pct}
    """
    cfg = config or {}
    df = get_klines(symbol, interval=interval, limit=700)
    if df is None or df.empty:
        return None
    df = add_indicators(df)
    last = df.iloc[-1]
    price_now = get_price(symbol) or float(last["close"])
    score = 0
    notes = []
    # RSI
    rsi = last.get("RSI", np.nan)
    if not math.isnan(rsi):
        if rsi < 30:
            score += 2; notes.append("RSI low")
        elif rsi > 70:
            score += 2; notes.append("RSI high")
    # MA cross
    if "MA20" in df.columns and "MA50" in df.columns and not math.isnan(df["MA20"].iloc[-2]):
        prev20, prev50 = df["MA20"].iloc[-2], df["MA50"].iloc[-2]
        cur20, cur50 = df["MA20"].iloc[-1], df["MA50"].iloc[-1]
        if prev20 < prev50 and cur20 > cur50:
            score += 2; notes.append("MA cross up")
            ma_sig = "LONG"
        elif prev20 > prev50 and cur20 < cur50:
            score += 2; notes.append("MA cross down")
            ma_sig = "SHORT"
        else:
            ma_sig = None
    else:
        ma_sig = None
    # pivot/fib
    pivs = find_pivots(df, left=4, right=4)
    lows = [p for p in pivs if p[1]=='L']; highs = [p for p in pivs if p[1]=='H']
    fib_note = None
    if lows and highs:
        last_low = lows[-1][2]; last_high = highs[-1][2]
        lowp, highp = (last_low, last_high) if last_low < last_high else (last_high, last_low)
        if lowp < highp:
            fibs = fib_levels(lowp, highp)
            # check retest near fibs
            for key in ("38.2%","50%","61.8%"):
                lvl = fibs[key]
                if abs(price_now - lvl)/lvl < 0.012:  # 1.2% tolerance
                    score += 1; notes.append(f"Retest {key}")
                    fib_note = key
    # recent breakout
    recent_high = df["high"].iloc[-21:-1].max()
    recent_low = df["low"].iloc[-21:-1].min()
    breakout = price_now > recent_high
    breakdown = price_now < recent_low
    vol = last["volume"]; vol_ma = last.get("vol_ma20", np.nan)
    vol_ok = False
    if not math.isnan(vol_ma) and vol > vol_ma * cfg.get("volume_factor",1.5):
        vol_ok = True; score += 1; notes.append("Vol spike")
    if breakout:
        score += 2 if vol_ok else 1
        notes.append("Breakout")
        break_sig = "LONG"
    elif breakdown:
        score += 2 if vol_ok else 1
        notes.append("Breakdown")
        break_sig = "SHORT"
    else:
        break_sig = None
    # RSI divergence detection
    div = detect_rsi_divergence(df, lookback=80)
    if div == 'bullish_div':
        score += 1; notes.append("Bullish RSI div")
    elif div == 'bearish_div':
        score += 1; notes.append("Bearish RSI div")
    # slope
    slp = slope(df["close"].tail(120))
    if slp > 0: notes.append("Up slope")
    if slp < 0: notes.append("Down slope")
    # final action decision
    action = None
    threshold = cfg.get("score_threshold", 4)
    if score >= threshold:
        # precedence: breakout > MA > RSI diverge
        if break_sig:
            action = break_sig
        elif ma_sig:
            action = ma_sig
        else:
            # fallback on RSI extremes
            if rsi < 30: action = "LONG"
            elif rsi > 70: action = "SHORT"
    if action:
        entry = price_now
        sl_pct = min(0.03, cfg.get("sl_max_pct",5.0)/100.0)
        tp_pct = max(cfg.get("tp_min_pct",10.0)/100.0, 0.10)
        if action == "LONG":
            sl = round(entry * (1 - sl_pct),8)
            tp1 = round(entry * (1 + tp_pct),8)
            # multi-target scaling: TP1, TP2 (fib extension idea)
            tp2 = round(entry * (1 + tp_pct*1.8),8)
            tp3 = round(entry * (1 + tp_pct*3.0),8)
        else:
            sl = round(entry * (1 + sl_pct),8)
            tp1 = round(entry * (1 - tp_pct),8)
            tp2 = round(entry * (1 - tp_pct*1.8),8)
            tp3 = round(entry * (1 - tp_pct*3.0),8)
        rr = None
        try:
            if action == "LONG": rr = round((tp1-entry)/max(entry-sl,1e-9),2)
            else: rr = round((entry-tp1)/max(sl-entry,1e-9),2)
        except:
            rr = None
        return {
            "symbol": symbol, "action": action, "price": round(entry,8),
            "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "score": score, "rr": rr, "notes": "; ".join(notes), "fib_note": fib_note
        }
    return None

# ------------- Session state -------------
if "last_scan" not in st.session_state: st.session_state.last_scan = 0
if "scan_results" not in st.session_state: st.session_state.scan_results = []
if "orders_cached" not in st.session_state: st.session_state.orders_cached = []

# ------------- UI -------------
st.set_page_config(page_title="AI Crypto Trading (Pro)", layout="wide")
st.title("AI Crypto Trading — Enhanced (Paper)")

# Sidebar
st.sidebar.header("Settings & Controls")
symbols = st.sidebar.text_area("Symbols (comma separated)", DEFAULT_SYMBOLS, height=150)
symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
interval = st.sidebar.selectbox("Detail interval", ["15m","1h","4h","1d"], index=2)
auto_scan = st.sidebar.checkbox("Auto-scan", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", min_value=30, value= AUTO_SCAN_DEFAULT, step=30)
score_threshold = st.sidebar.number_input("Score threshold", min_value=1, max_value=10, value=4)
volume_factor = st.sidebar.number_input("Volume factor", min_value=1.0, value=1.5)
sl_max_pct = st.sidebar.number_input("SL max %", min_value=0.5, value=5.0)
tp_min_pct = st.sidebar.number_input("TP min %", min_value=1.0, value=10.0)

# Telegram secrets (read from Streamlit secrets or env)
tg_token = st.secrets.get("telegram_bot_token") if "telegram_bot_token" in st.secrets else os.environ.get("TG_BOT_TOKEN","")
tg_chat = st.secrets.get("telegram_chat_id") if "telegram_chat_id" in st.secrets else os.environ.get("TG_CHAT_ID","")
if tg_token and tg_chat:
    tg_enabled = True
else:
    tg_enabled = False

auto_exec = st.sidebar.checkbox("Auto-execute suggestions (paper)", value=False)
st.sidebar.info("Paper-trade only. For live trading implement secure backend & never store API keys in plaintext on public hosting.")

# Top area: Market watch + quick scan
st.subheader("Market Watch")
col1, col2 = st.columns([2,1])
with col1:
    price_rows = []
    for s in symbols:
        p = get_price(s)
        price_rows.append({"symbol": s, "price": round(p,8) if p else "err"})
    st.table(pd.DataFrame(price_rows))
with col2:
    if st.button("Manual scan now"):
        st.session_state.last_scan = 0

# scan orchestration
now = time.time()
if (auto_scan and (now - st.session_state.last_scan > scan_interval)) or st.session_state.last_scan == 0:
    st.session_state.last_scan = now
    cfg = {"score_threshold": score_threshold, "volume_factor": volume_factor, "sl_max_pct": sl_max_pct, "tp_min_pct": tp_min_pct}
    out = []
    for s in symbols:
        try:
            r = signal_for_symbol(s, interval=interval, config=cfg)
            if r:
                out.append(r)
        except Exception as e:
            logger.exception("scan symbol error %s %s", s, e)
    st.session_state.scan_results = out
    # auto-execute as paper orders
    if auto_exec and out:
        for sug in out:
            # prevent duplicates same symbol+side open
            existing = DB.cursor().execute("SELECT count(*) FROM orders WHERE symbol=? AND side=? AND status='OPEN'", (sug["symbol"], sug["action"])).fetchone()[0]
            if existing > 0:
                continue
            order = {
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sug["symbol"], "side": sug["action"], "entry": sug["price"],
                "sl": sug["sl"], "tp": sug["tp1"], "size": 1.0, "status": "OPEN",
                "closed_at": None, "current_price": sug["price"], "pnl_pct": 0.0, "notes": sug["notes"]
            }
            db_insert_order(order)
            if tg_enabled:
                tg_notify(tg_token, tg_chat, f"[Paper order] {order['symbol']} {order['side']} @ {order['entry']} SL {order['sl']} TP {order['tp']}")
    # log scan
    db_log("INFO", f"Scan completed: {len(st.session_state.scan_results)} suggestions")
# show results
st.markdown("---")
st.subheader("Suggestions / Signals")
if st.session_state.scan_results:
    df_s = pd.DataFrame(st.session_state.scan_results)
    st.table(df_s[["symbol","action","price","sl","tp1","score","rr"]])
    for i,r in df_s.iterrows():
        c1,c2 = st.columns([3,1])
        with c1:
            st.write(f"**{r['symbol']}** — {r['action']} @ {r['price']} (score {r['score']})")
            st.caption(r.get("notes",""))
        with c2:
            if st.button(f"Add paper order {i}"):
                existing = DB.cursor().execute("SELECT count(*) FROM orders WHERE symbol=? AND side=? AND status='OPEN'", (r['symbol'], r['action'])).fetchone()[0]
                if existing > 0:
                    st.warning("Similar open order exists")
                else:
                    order = {
                        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": r['symbol'], "side": r['action'], "entry": r['price'],
                        "sl": r['sl'], "tp": r['tp1'], "size": 1.0, "status": "OPEN",
                        "closed_at": None, "current_price": r['price'], "pnl_pct": 0.0, "notes": r.get("notes","")
                    }
                    db_insert_order(order)
                    st.success("Paper order added")
else:
    st.info("No suggestions now")

# Chart & analysis
st.markdown("---")
st.subheader("Chart & Trendline")
chart_sym = st.selectbox("Chart symbol", symbols, index=0)
dfc = get_klines(chart_sym, interval=interval, limit=500)
if dfc is None:
    st.error("Cannot fetch chart data")
else:
    dfc = add_indicators(dfc)
    fig = go.Figure(data=[go.Candlestick(x=dfc["open_time"], open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name=chart_sym)])
    if "MA20" in dfc.columns: fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA20"], name="MA20"))
    if "MA50" in dfc.columns: fig.add_trace(go.Scatter(x=dfc["open_time"], y=dfc["MA50"], name="MA50"))
    pivs = find_pivots(dfc, left=4, right=4)
    for p in pivs[-50:]:
        idx, t, px = p
        tm = dfc["open_time"].iloc[idx]
        if t == 'H':
            fig.add_trace(go.Scatter(x=[tm], y=[px], mode="markers", marker=dict(color="red", size=7), showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[tm], y=[px], mode="markers", marker=dict(color="green", size=7), showlegend=False))
    # draw fib from last swing if found
    lows = [p for p in pivs if p[1]=='L']; highs = [p for p in pivs if p[1]=='H']
    if lows and highs:
        last_low = lows[-1][2]; last_high = highs[-1][2]
        lowp, highp = (last_low, last_high) if last_low < last_high else (last_high, last_low)
        if lowp < highp:
            fibs = fib_levels(lowp, highp)
            for k,v in fibs.items():
                fig.add_hline(y=v, line=dict(color="yellow" if k in ("38.2%","50%","61.8%") else "gray", width=1), annotation_text=k, annotation_position="top left")
    fig.update_layout(height=640, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Paper orders manager
st.markdown("---")
st.subheader("Paper Orders")
orders_df = db_list_orders()
if not orders_df.empty:
    st.dataframe(orders_df, use_container_width=True)
    # refresh prices & auto TP/SL
    for idx,row in orders_df.iterrows():
        if row['status'] == 'CLOSED': continue
        cp = get_price(row['symbol']) or row['current_price']
        pnl = 0.0
        if row['side'] == 'LONG': pnl = (cp - row['entry'])/row['entry']*100
        else: pnl = (row['entry'] - cp)/row['entry']*100
        # update DB row: (for simplicity we update via SQL)
        DB.cursor().execute("UPDATE orders SET current_price=?, pnl_pct=? WHERE id=?", (cp, round(pnl,4), row['id']))
        DB.commit()
        # auto-close
        if row['status']=='OPEN':
            if row['side']=='LONG':
                if cp >= row['tp']:
                    DB.cursor().execute("UPDATE orders SET status='CLOSED', closed_at=? WHERE id=?", (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), row['id']))
                    DB.commit()
                    if tg_enabled: tg_notify(tg_token, tg_chat, f"[TP Hit] {row['symbol']} LONG paper @ {cp}")
                elif cp <= row['sl']:
                    DB.cursor().execute("UPDATE orders SET status='CLOSED', closed_at=? WHERE id=?", (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), row['id']))
                    DB.commit()
                    if tg_enabled: tg_notify(tg_token, tg_chat, f"[SL Hit] {row['symbol']} LONG paper @ {cp}")
            else:
                if cp <= row['tp']:
                    DB.cursor().execute("UPDATE orders SET status='CLOSED', closed_at=? WHERE id=?", (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), row['id']))
                    DB.commit()
                    if tg_enabled: tg_notify(tg_token, tg_chat, f"[TP Hit] {row['symbol']} SHORT paper @ {cp}")
                elif cp >= row['sl']:
                    DB.cursor().execute("UPDATE orders SET status='CLOSED', closed_at=? WHERE id=?", (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), row['id']))
                    DB.commit()
                    if tg_enabled: tg_notify(tg_token, tg_chat, f"[SL Hit] {row['symbol']} SHORT paper @ {cp}")
    # refresh view
    st.experimental_rerun()
else:
    st.info("No paper orders yet")

# Reports & export
st.markdown("---")
st.subheader("Reports & Export")
if st.button("Export orders CSV"):
    df = db_list_orders()
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name=f"paper_orders_{datetime.utcnow().strftime('%Y%m%d')}.csv")

st.info("Paper-trade only. For production/live trading: use secure backend, never store real API keys in this UI.")
