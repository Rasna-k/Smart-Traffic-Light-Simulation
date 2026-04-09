import requests

ESP32_IP = "http://10.149.225.219"   # <-- CHANGE THIS

def send_signal(direction, phase):
    """
    direction: North, South, East, West
    phase: green, yellow, red
    """

    try:
        url = f"{ESP32_IP}/set?dir={direction}&state={phase}"
        requests.get(url, timeout=2)
    except Exception as e:
        print("ESP32 Error:", e)
