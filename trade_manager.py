# trade_manager.py - simple paper order manager
import json, os, time
from datetime import datetime

ORDERS_FILE = 'data/paper_orders.json'
os.makedirs('data', exist_ok=True)

def load_orders():
    if os.path.exists(ORDERS_FILE):
        try:
            with open(ORDERS_FILE,'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_orders(orders):
    with open(ORDERS_FILE,'w') as f:
        json.dump(orders, f, indent=2, default=str)

def place_order(symbol, side, qty, price, note='paper'):
    orders = load_orders()
    order = {
        'id': int(time.time()*1000),
        'time': datetime.utcnow().isoformat(),
        'symbol': symbol,
        'side': side,
        'qty': qty,
        'price': price,
        'note': note,
        'status': 'open'
    }
    orders.append(order)
    save_orders(orders)
    return order

def close_order(order_id, exit_price):
    orders = load_orders()
    for o in orders:
        if o['id'] == order_id and o['status']=='open':
            o['status'] = 'closed'
            o['exit_price'] = exit_price
            o['close_time'] = datetime.utcnow().isoformat()
    save_orders(orders)
    return True
