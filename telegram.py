# telegram.py - simple sender (optional)
import requests

def send_telegram(token, chat_id, text):
    if not token or not chat_id:
        return False
    try:
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        r = requests.post(url, json={'chat_id': chat_id, 'text': text})
        return r.ok
    except Exception as e:
        print('tg error', e)
        return False
