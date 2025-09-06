import requests

def send_telegram(bot_token: str, chat_id: str, text: str):
    if not bot_token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text})
        return resp.ok
    except Exception:
        return False
