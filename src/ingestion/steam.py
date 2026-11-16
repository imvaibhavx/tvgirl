import json
import time
from pathlib import Path

import requests

URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

GAMES = {
    1086940: "Baldur's Gate 3",
    1245620: "Elden Ring",
    1091500: "Cyberpunk 2077",
    570: "Dota 2",
}


def fetch_page(appid, cursor="*"):
    payload = {
        "appid": appid,
        "filter": 1,  # 1 = Recent
        "languages": ["english"],
        "cursor": cursor,
    }
    r = requests.get(URL, params={"input_json": json.dumps(payload)}, timeout=15)
    r.raise_for_status()
    return r.json()["response"]


def clean(review):
    text = " ".join(review["review"].split())  # collapse whitespace/newlines
    return {
        "review_id": review["recommendationid"],
        "text": text,
        "voted_up": review["voted_up"],
        "playtime_at_review": review["author"].get("playtime_at_review", 0),
        "timestamp_created": review["timestamp_created"],
    }


def fetch_reviews(appid, max_reviews=500, min_chars=20):
    out, seen, cursor = [], set(), "*"
    while len(out) < max_reviews:
        body = fetch_page(appid, cursor)
        batch = body.get("reviews", [])
        if not batch:
            break
        for rv in batch:
            row = clean(rv)
            if len(row["text"]) >= min_chars and row["review_id"] not in seen:
                seen.add(row["review_id"])
                out.append(row)
        next_cursor = body.get("cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.5)  # be polite
    return out[:max_reviews]


def save(appid, rows):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"steam_{appid}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path

def ingest_all():
    for appid, name in GAMES.items():
        rows = fetch_reviews(appid)
        path = save(appid, rows)
        print(f"{name}: kept {len(rows)} reviews -> {path}")

if __name__ == "__main__":
    for appid, name in GAMES.items():
        rows = fetch_reviews(appid)
        path = save(appid, rows)
        pos = sum(r["voted_up"] for r in rows)
        print(f"{name}: kept {len(rows)} reviews ({pos} positive) -> {path}")