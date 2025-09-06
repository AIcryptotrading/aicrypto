import pandas as pd

class Backtester:
    def __init__(self, strategy, initial_balance=1000):
        self.strategy = strategy
        self.balance = initial_balance
        self.positions = []
        self.history = []

    def run(self, df: pd.DataFrame):
        for i in range(len(df)):
            signal = self.strategy(df.iloc[:i+1])
            price = df['close'].iloc[i]

            if signal == 'BUY' and self.balance > 0:
                self.positions.append(price)
                self.balance -= price
                self.history.append(('BUY', price, self.balance))
            elif signal == 'SELL' and self.positions:
                entry = self.positions.pop(0)
                self.balance += price
                self.history.append(('SELL', price, self.balance))
        return pd.DataFrame(self.history, columns=['Action','Price','Balance'])
