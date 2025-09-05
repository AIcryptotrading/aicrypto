import os
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # để render chart không cần GUI
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
from datetime import datetime

app = Flask(__name__)

BINANCE_API = "https://api.binance.com/api/v3/klines"

# ====== Lấy dữ liệu giá từ Binance ======
def get_klines(symbol="BTCUSDT", interval="1h", limit=200):
    url = f"{BINANCE_API}?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url)
    data = res.json()
    df = pd.DataFrame(data, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_base_vol","taker_quote_vol","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df

# ====== Vẽ chart với trendline ======
def plot_chart(df, symbol):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["timestamp"], df["close"], label="Close Price", color="blue")

    # trendline cơ bản: nối 2 đáy gần nhất và 2 đỉnh gần nhất
    lows = df.nsmallest(2, "close")
    highs = df.nlargest(2, "close")
    ax.plot(lows["timestamp"], lows["close"], color="green", linestyle="--", label="Support Trendline")
    ax.plot(highs["timestamp"], highs["close"], color="red", linestyle="--", label="Resistance Trendline")

    ax.set_title(f"{symbol} Trendline Chart")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USDT)")
    ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

# ====== Đánh giá cơ bản: Long hay Short ======
def evaluate_signal(df):
    recent = df.iloc[-1]
    prev = df.iloc[-2]

    # logic đơn giản: nếu close > open (nến xanh) và trên trendline → Long
    if recent["close"] > prev["close"]:
        return "LONG"
    elif recent["close"] < prev["close"]:
        return "SHORT"
    else:
        return "NEUTRAL"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/market")
def market():
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "1h")
    df = get_klines(symbol, interval)
    signal = evaluate_signal(df)
    price = df.iloc[-1]["close"]
    return jsonify({
        "symbol": symbol,
        "interval": interval,
        "latest_price": price,
        "signal": signal,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/chart")
def chart():
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "1h")
    df = get_klines(symbol, interval)
    buf = plot_chart(df, symbol)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
