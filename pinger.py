import time
import requests

URL = "https://hiteckgroupp-hi-tecknuminfo.hf.space"

while True:
    try:
        response = requests.get(URL, timeout=10)
        print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Ping failed: {e}")

    time.sleep(300)  # 5 minutes