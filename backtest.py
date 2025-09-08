# backtest.py - simple backtester
def run_backtest(df, strategy_func, initial_balance=1000):
    balance = initial_balance
    position = None
    trades = []
    for i in range(30, len(df)):
        sub = df.iloc[:i+1]
        sig = strategy_func(sub)
        price = float(sub['close'].iloc[-1])
        if sig == 'LONG' and position is None:
            position = {'entry': price, 'side':'LONG', 'entry_index': i}
        elif sig == 'SHORT' and position is None:
            position = {'entry': price, 'side':'SHORT', 'entry_index': i}
        # simple exit: close when reverse signal
        if position is not None and sig is None:
            pnl = (price - position['entry']) if position['side']=='LONG' else (position['entry'] - price)
            trades.append({'entry': position['entry'], 'exit': price, 'pnl': pnl})
            balance += pnl
            position = None
    return trades, balance
