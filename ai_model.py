# ai_model.py - lightweight AI for paper trading
# Provides: train_quick (RandomForest) and predict function (supervised). If sklearn not available, use rule-based fallback.
import os, joblib, numpy as np

MODEL_FILE = 'models/supervised_rf.pkl'
os.makedirs('models', exist_ok=True)

def featurize(df):
    df = df.copy().dropna()
    X = []
    for i in range(len(df)):
        row = df.iloc[i]
        X.append([row.get('close',0), row.get('ema9',0), row.get('ema25',0), row.get('rsi14',50)])
    return np.array(X)

def build_label(df, horizon=3, pct=0.005):
    close = df['close'].astype(float).values
    y = []
    for i in range(len(close)):
        if i + horizon < len(close):
            future = close[i+1:i+1+horizon].max()
            y.append(1 if (future/close[i]-1) > pct else 0)
        else:
            y.append(0)
    return np.array(y)

def train_quick(df):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise RuntimeError('scikit-learn not installed: '+str(e))
    X = featurize(df)
    y = build_label(df)
    if len(X) < 50:
        raise RuntimeError('Not enough data to train')
    model = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=1)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

def predict(df):
    model = load_model()
    if model is None:
        # simple rule: if close > ema25 and rsi < 70 => buy signal (1), else 0
        last = df.iloc[-1]
        if last['close'] > last['ema25'] and last['rsi14'] < 70:
            return 1
        return 0
    X = featurize(df)
    return int(model.predict(X[-1].reshape(1,-1))[0])
