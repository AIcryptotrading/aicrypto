# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
import math
import io
import plotly.graph_objects as go
import time

st.set_page_config(page_title="AI Crypto Trading Dashboard", layout="wide")
VTZ = timezone(timedelta(hours=7))

# -----------------------
# CONFIG
# -----------------------
DEFAULT_SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOTUSDT",
    "TIAUSDT","KAVAUSDT","RENDERUSDT","AVAXUSDT","NEARUSDT","SEIUSDT"
]
BINANCE_BASE = "https://api.binance.com"
AUTO_SCAN_INTERVAL_DEFAULT = 900  # 15 minutes (seconds)
SL_MAX_PCT = 5.0
TP_MIN_PCT = 10.0

# -----------------------
# Utils
# -----------------------
def now_vn():
    return datetime.now(VTZ).strftime("%Y-%m-%d %H:%M:%S")

def fetch_price(symbol):
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        return None

def fetch_klines(symbol, interval="4h", limit=200):
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
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
    except Exception as e:
        return None

def linear_fit_y(y):
    x = np.arange(len(y))
    if len(y) < 2:
        return 0.0, float(y.iloc[-1]) if len(y)>0 else 0.0
    slope, intercept = np.polyfit(x, y.values, 1)
    return slope, intercept

def calc_trendlines(h4_df):
    # Use last N highs/lows to estimate trendlines
    N = min(60, len(h4_df))
    highs = h4_df["high"].tail(N).reset_index(drop=True)
    lows  = h4_df["low"].tail(N).reset_index(drop=True)
    s_h, i_h = linear_fit_y(highs)
    s_l, i_l = linear_fit_y(lows)
    return (s_h, i_h), (s_l, i_l)

def predict_line(slope, intercept, idx):
    return slope*idx + intercept

def propose_trade_from_signal(symbol, price, pred_high_slope_intercept, pred_low_slope_intercept, rsi_h4):
    s_h, i_h = pred_high_slope_intercept
    # index current is last index
    # We'll simply check if price > predicted descending high and slope<0
    signal = "WAIT"; action="WAIT"; sl=None; tp=None; rr=None; note=""
    pred_high = predict_line(s_h, i_h, 59) if s_h is not None else None
    if s_h is not None and s_h < 0 and price is not None and price > pred_high and rsi_h4 is not None and rsi_h4 > 45:
        action="LONG"
        # conservative SL at 3% under entry but enforce SL_MAX_PCT
        sl = round(price*(1 - min(0.03, SL_MAX_PCT/100)), 8)
        tp = round(price*(1 + TP_MIN_PCT/100), 8)
        rr = (tp - price) / max(price - sl, 1e-9)
        signal = "BreakDownTrend"
        note = "Break trend giảm H4 + RSI>45"
    # simple double-top quick detection (price close to recent high)
    # not exhaustive, used as SHORT signal
    return {"Symbol":symbol,"Action":action,"Signal":signal,"SL":sl,"TP":tp,"RR":round(rr,2) if rr else None,"Note":note}

# -----------------------
# Session State init
# -----------------------
if "orders" not in st.session_state:
    st.session_state.orders = []  # each order: dict with fields below
if "market_cache" not in st.session_state:
    st.session_state.market_cache = {}  # symbol -> latest price
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

# -----------------------
# Layout: sidebar controls
# -----------------------
st.sidebar.header("Settings / Controls")
symbols = st.sidebar.text_area("Symbols (comma separated)", value=",".join(DEFAULT_SYMBOLS), height=120)
symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]

interval_choice = st.sidebar.selectbox("Chart interval (for detail)", ["1h","4h","1d"], index=1)
auto_scan = st.sidebar.checkbox("Auto-scan market for setups", value=True)
scan_interval = st.sidebar.number_input("Auto-scan interval (sec)", min_value=60, value=AUTO_SCAN_INTERVAL_DEFAULT, step=60)
max_suggest = st.sidebar.number_input("Max suggestions per scan", min_value=1, value=6, step=1)
st.sidebar.write("SL max %:", SL_MAX_PCT, "  TP min %:", TP_MIN_PCT)
st.sidebar.write("Time (VN):", now_vn())

# -----------------------
# Top: Market Watch + scan button
# -----------------------
st.title("📈 AI Crypto Trading Dashboard (Paper-trade)")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Market Watch")
    # live list of prices
    prices_table = []
    for s in symbols:
        p = fetch_price(s)
        if p is None and s in st.session_state.market_cache:
            p = st.session_state.market_cache[s]
        if p is not None:
            st.session_state.market_cache[s] = p
        prices_table.append({"Symbol":s, "Price": p if p is not None else "err"})
    st.table(pd.DataFrame(prices_table))
with col2:
    st.subheader("Scan / Actions")
    if st.button("Manual Scan now"):
        st.session_state.scan_results = []
        st.session_state.last_scan = now_vn()
        # perform scan now
        for s in symbols:
            h4 = fetch_klines(s, interval="4h", limit=200)
            if h4 is None: continue
            (s_h,i_h),(s_l,i_l) = calc_trendlines(h4)
            # approximate rsi by momentum simple
            closes = h4["close"]
            delta = closes.diff().fillna(0)
            up = delta.clip(lower=0).rolling(14).mean()
            down = -delta.clip(upper=0).rolling(14).mean()
            rsi = 100 - 100/(1 + (up/(down.replace(0,1e-9))))
            rsi_h4 = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None
            price = fetch_price(s)
            st.session_state.scan_results.append(propose_trade_from_signal(s, price, (s_h,i_h),(s_l,i_l), rsi_h4))
        st.success("Scan completed at " + st.session_state.last_scan)
    if auto_scan:
        # check interval
        if st.session_state.last_scan is None or (time.time() - (pd.to_datetime(st.session_state.last_scan).timestamp() if st.session_state.last_scan else 0)) > scan_interval:
            # perform automatic scan (lightweight)
            st.session_state.scan_results = []
            st.session_state.last_scan = now_vn()
            for s in symbols:
                h4 = fetch_klines(s, interval="4h", limit=120)
                if h4 is None: continue
                (s_h,i_h),(s_l,i_l) = calc_trendlines(h4)
                closes = h4["close"]
                delta = closes.diff().fillna(0)
                up = delta.clip(lower=0).rolling(14).mean()
                down = -delta.clip(upper=0).rolling(14).mean()
                rsi = 100 - 100/(1 + (up/(down.replace(0,1e-9))))
                rsi_h4 = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None
                price = fetch_price(s)
                st.session_state.scan_results.append(propose_trade_from_signal(s, price, (s_h,i_h),(s_l,i_l), rsi_h4))
            st.experimental_rerun()

    # show top suggestions
    if st.session_state.scan_results:
        df_scan = pd.DataFrame(st.session_state.scan_results)
        df_pick = df_scan[df_scan["Action"]!="WAIT"].sort_values(by="RR", ascending=False).head(max_suggest)
        st.write("Suggestions (auto):")
        if not df_pick.empty:
            st.table(df_pick[["Symbol","Action","Signal","RR","SL","TP","Note"]])
            # quick-buttons to add suggested orders
            for _, row in df_pick.iterrows():
                colA, colB = st.columns([2,1])
                with colA:
                    st.write(f"**{row['Symbol']}** → {row['Action']} | RR={row['RR']} | {row['Note']}")
                with colB:
                    if st.button(f"Add {row['Symbol']}", key=f"add_{row['Symbol']}"):
                        # append order
                        entry = fetch_price(row["Symbol"]) or row["TP"] or 0
                        sl = row["SL"]
                        tp = row["TP"]
                        order = {
                            "time": now_vn(),
                            "symbol": row["Symbol"],
                            "side": row["Action"],
                            "entry": round(entry,8),
                            "sl": sl,
                            "tp": tp,
                            "size": 1.0,
                            "leverage": 10,
                            "status": "OPEN",
                            "note": row["Note"]
                        }
                        st.session_state.orders.append(order)
                        st.success("Order added: " + row["Symbol"])

# -----------------------
# Main: Chart + Order Manager
# -----------------------
st.markdown("---")
left, right = st.columns([2,1])
with left:
    st.subheader("Chart & Trendline")
    sel_symbol = st.selectbox("Select symbol", symbols, index=0)
    data_interval = st.selectbox("Chart Interval", ["1h","4h","1d"], index=1)
    df = fetch_klines(sel_symbol, interval=data_interval, limit=200)
    if df is None:
        st.error("Cannot load klines for " + sel_symbol)
    else:
        # compute trendline markers (simple: connecting lowest recent lows and highest highs)
        lows = df.nsmallest(3, "low").sort_values("open_time")
        highs = df.nlargest(3, "high").sort_values("open_time")
        # candlestick
        fig = go.Figure(data=[go.Candlestick(
            x=df["open_time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="green", decreasing_line_color="red"
        )])
        # draw simple support/resistance lines (fit last 50 bars)
        try:
            (s_h,i_h),(s_l,i_l) = calc_trendlines(df)
            idx_last = min(59, len(df)-1)
            # generate line points (map to timestamps)
            xs = [df["open_time"].iloc[max(0, idx_last-59)], df["open_time"].iloc[idx_last]]
            y_high = [predict_line(s_h,i_h,0), predict_line(s_h,i_h,59)]
            y_low  = [predict_line(s_l,i_l,0), predict_line(s_l,i_l,59)]
            fig.add_trace(go.Scatter(x=xs, y=y_high, mode="lines", line=dict(color="blue",dash="dash"), name="Trend High"))
            fig.add_trace(go.Scatter(x=xs, y=y_low, mode="lines", line=dict(color="orange",dash="dash"), name="Trend Low"))
        except Exception:
            pass
        fig.update_layout(height=600, margin={"t":20})
        st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Order Manager (Paper)")
    # add manual order form
    with st.form("manual"):
        c1,c2,c3 = st.columns(3)
        symbol_in = c1.text_input("Coin", sel_symbol)
        side_in = c2.selectbox("Side", ["Long","Short"])
        entry_in = c3.number_input("Entry price", value=float(fetch_price(sel_symbol) or 0.0), format="%.8f")
        c4,c5 = st.columns(2)
        size_in = c4.number_input("Size (units)", value=1.0, min_value=0.0001)
        lev_in  = c5.number_input("Leverage", value=10, min_value=1)
        tp_in = st.number_input("TP (price)", value=round(entry_in*(1.10 if side_in=="Long" else 0.90),8), format="%.8f")
        sl_in = st.number_input("SL (price)", value=round(entry_in*(0.97 if side_in=="Long" else 1.03),8), format="%.8f")
        note_in = st.text_input("Note","")
        submit = st.form_submit_button("Add Order")
        if submit:
            st.session_state.orders.append({
                "time": now_vn(),
                "symbol": symbol_in.upper(),
                "side": side_in,
                "entry": float(entry_in),
                "sl": float(sl_in),
                "tp": float(tp_in),
                "size": float(size_in),
                "leverage": int(lev_in),
                "status": "OPEN",
                "note": note_in
            })
            st.success("Order added")

    # show orders table
    if st.session_state.orders:
        df_orders = pd.DataFrame(st.session_state.orders)
        # update current price & PnL
        current_prices = {s:fetch_price(s) for s in set(df_orders["symbol"].tolist())}
        pnl_list = []
        for i,row in df_orders.iterrows():
            cp = current_prices.get(row["symbol"], None) or row["entry"]
            if row["side"]=="Long":
                pnl_pct = (cp - row["entry"]) / row["entry"] * 100
            else:
                pnl_pct = (row["entry"] - cp) / row["entry"] * 100
            pnl_list.append(round(pnl_pct,3))
            st.session_state.orders[i]["current_price"]=round(cp,8)
            st.session_state.orders[i]["pnl_pct"]=round(pnl_pct,3)
        df_orders = pd.DataFrame(st.session_state.orders)
        st.dataframe(df_orders[["time","symbol","side","entry","current_price","pnl_pct","sl","tp","status","note"]], use_container_width=True)
        # actions per order
        for idx, ord in enumerate(st.session_state.orders):
            colA,colB,colC = st.columns([2,1,2])
            with colA:
                st.write(f"{ord['symbol']} | {ord['side']} | Entry {ord['entry']}")
            with colB:
                if st.button(f"Close {idx}", key=f"close_{idx}"):
                    st.session_state.orders[idx]["status"]="CLOSED"
                    st.session_state.orders[idx]["closed_time"]=now_vn()
                    st.success("Closed order "+ord["symbol"])
            with colC:
                if st.button(f"Delete {idx}", key=f"del_{idx}"):
                    st.session_state.orders.pop(idx)
                    st.experimental_rerun()

    # download log
    if st.session_state.orders:
        csv = pd.DataFrame(st.session_state.orders).to_csv(index=False).encode("utf-8")
        st.download_button("Download orders CSV", csv, file_name=f"orders_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

# -----------------------
# Bottom: Reports & Summary
# -----------------------
st.markdown("---")
st.subheader("Daily Summary & Reports")
if st.button("Generate daily summary now"):
    df = pd.DataFrame(st.session_state.orders)
    wins = df[(df["status"]=="CLOSED") & (df["pnl_pct"]>0)]
    losses = df[(df["status"]=="CLOSED") & (df["pnl_pct"]<=0)]
    total_pnl = df[(df["status"]=="CLOSED")]["pnl_pct"].sum() if not df.empty else 0
    summary = {
        "time": now_vn(),
        "total_orders": len(df),
        "closed": len(df[df["status"]=="CLOSED"]),
        "wins": len(wins),
        "losses": len(losses),
        "total_pnl_pct": round(total_pnl,3)
    }
    st.json(summary)
    st.success("Summary generated. You can download full CSV above.")

st.markdown("### Notes / Disclaimer")
st.info("This app is for **paper trading / analysis** and educational purposes. It does NOT execute real trades. Use at your own risk.")

# end of app
