import json
import requests

URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"

payload = {
    "appid": 1245620,       # Elden Ring
    "filter": 1,            # 1 = Recent
    "languages": ["english"],
    "cursor": "*",
}

r = requests.get(URL, params={"input_json": json.dumps(payload)}, timeout=15)
print("status:", r.status_code)
print("content-type:", r.headers.get("content-type"))

data = r.json()
print("top-level keys:", list(data.keys()))
body = data.get("response", data)   # Steam often wraps results in "response"
print("body keys:", list(body.keys()))

reviews = body.get("reviews", [])
print("reviews returned:", len(reviews))
if reviews:
    print("first review keys:", list(reviews[0].keys()))
    print(json.dumps(reviews[0], indent=2)[:800])