import os, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = 'models'
SUP_MODEL = os.path.join(MODEL_DIR, 'supervised.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_supervised_dataset(df, future_period=5, return_threshold=0.01):
    df = df.copy().dropna()
    df['future_ret'] = df['close'].shift(-future_period)/df['close'] - 1.0
    df['label'] = (df['future_ret'] > return_threshold).astype(int)
    features = ['rsi14','ema20','ema50','atr14','vol_ma20']
    X = df[features].fillna(0).values
    y = df['label'].fillna(0).astype(int).values
    return X,y

def train_supervised(df):
    X,y = prepare_supervised_dataset(df)
    if len(X) < 50:
        raise RuntimeError('Not enough data to train (need >=50 rows)')
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X,y)
    joblib.dump(model, SUP_MODEL)
    return model

def load_supervised():
    if os.path.exists(SUP_MODEL):
        return joblib.load(SUP_MODEL)
    return None

def notes_heavy_ai():
    return """For RL training use stable-baselines3 PPO with custom env (env.py).
For transformer/LSTM forecasting, prepare sequence dataset and use PyTorch/TF or HuggingFace models.
These require GPU and extra setup; include them only if you will run training separately."""
