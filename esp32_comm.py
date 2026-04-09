import requests

ESP32_IP = "http://192.168.1.45"   # <-- change this

def send_phase(ns, ew):
    """
    ns, ew = 'green', 'yellow', 'red'
    """
    try:
        url = f"{ESP32_IP}/set?ns={ns}&ew={ew}"
        requests.get(url, timeout=2)
    except Exception as e:
        print("ESP32 Error:", e)
