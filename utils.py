import requests
import pandas as pd
import numpy as np
from datetime import datetime

BASE_URL = "https://api.binance.com"

def fetch_klines(symbol: str, interval: str = "4h", limit: int = 500):
    try:
        url = f"{BASE_URL}/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('open_time')
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        raise RuntimeError(f"fetch_klines error for {symbol}: {e}")

def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_indicators(df: pd.DataFrame):
    df = df.copy()
    df['rsi14'] = compute_rsi(df['close'], period=14)
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ma200'] = df['close'].rolling(window=200, min_periods=1).mean()
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14, min_periods=1).mean()
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=1).mean()
    return df

def last_price(symbol: str):
    try:
        url = BASE_URL + "/api/v3/ticker/price"
        r = requests.get(url, params={"symbol": symbol.upper()}, timeout=5)
        r.raise_for_status()
        return float(r.json()['price'])
    except Exception as e:
        raise RuntimeError(f"last_price error {symbol}: {e}")
