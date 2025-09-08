# utils.py - data fetch and indicators (lightweight)
import pandas as pd, numpy as np

def fetch_ohlcv_yf(symbol='BTC-USD', interval='1h', period='60d'):
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={'Datetime':'time','Date':'time','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        df['time'] = pd.to_datetime(df['time'])
        df = df[['time','open','high','low','close','volume']]
        return df
    except Exception:
        return pd.DataFrame()

def fetch_ohlcv_ccxt(symbol='BTC/USDT', timeframe='1h', limit=500):
    try:
        import ccxt
        ex = ccxt.binance({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except Exception:
        return pd.DataFrame()

def fetch_ohlcv(symbol='BTCUSDT', interval='1h', limit=500):
    # try ccxt/binance first (local), else yfinance fallback
    df = fetch_ohlcv_ccxt(symbol.replace('USDT','/USDT'), timeframe=interval, limit=limit)
    if df is None or df.empty:
        df = fetch_ohlcv_yf(symbol.replace('USDT','-USD'), interval=interval, period='120d')
    return df

def add_indicators(df):
    df = df.copy()
    if df.empty:
        return df
    df['close'] = df['close'].astype(float)
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema25'] = df['close'].ewm(span=25, adjust=False).mean()
    # RSI simple
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - 100/(1+rs)
    df['volume_ma20'] = df['volume'].rolling(window=20).mean().fillna(0)
    return df

def last_price(df):
    if df is None or df.empty:
        return None
    return float(df['close'].iloc[-1])
