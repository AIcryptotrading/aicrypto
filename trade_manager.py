import time
PAPER_ORDERS = []

def place_paper_order(symbol: str, side: str, price: float, size: float, note: str=""):
    order = {
        'id': len(PAPER_ORDERS)+1,
        'ts': time.time(),
        'symbol': symbol.upper(),
        'side': side.upper(),
        'price': float(price),
        'size': float(size),
        'note': note
    }
    PAPER_ORDERS.append(order)
    return order

def list_paper_orders():
    return PAPER_ORDERS

def clear_paper_orders():
    PAPER_ORDERS.clear()
