import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from src.ingestion.steam import GAMES

load_dotenv()

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DDL = """
CREATE TABLE IF NOT EXISTS reviews (
    source TEXT NOT NULL,
    review_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    item_name TEXT,
    text TEXT NOT NULL,
    voted_up BOOLEAN,
    playtime_at_review INT,
    created_at TIMESTAMPTZ,
    PRIMARY KEY (source, review_id)
);
"""


def connect():
    return psycopg2.connect(
        host=os.getenv("WAREHOUSE_HOST", "localhost"),
        port=int(os.getenv("WAREHOUSE_PORT", 5433)),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )


def ascii_ratio(t):
    return sum(c.isascii() for c in t) / len(t)


def transform(row, appid):
    text = row["text"]
    if ascii_ratio(text) < 0.9:
        return None  # drop mostly non-English
    return (
        "steam",
        row["review_id"],
        str(appid),
        GAMES.get(appid),
        text[:1000],
        row["voted_up"],
        row["playtime_at_review"],
        row["timestamp_created"],
    )


def load():
    rows = []
    for path in RAW_DIR.glob("steam_*.jsonl"):
        appid = int(path.stem.split("_")[1])
        with open(path, encoding="utf-8") as f:
            for line in f:
                out = transform(json.loads(line), appid)
                if out:
                    rows.append(out)

    sql = """
        INSERT INTO reviews
        (source, review_id, item_id, item_name, text, voted_up, playtime_at_review, created_at)
        VALUES %s
        ON CONFLICT (source, review_id) DO NOTHING
    """
    template = "(%s,%s,%s,%s,%s,%s,%s,to_timestamp(%s))"

    with connect() as conn, conn.cursor() as cur:
        cur.execute(DDL)
        execute_values(cur, sql, rows, template=template)
        cur.execute("SELECT item_name, COUNT(*), SUM(voted_up::int) FROM reviews GROUP BY item_name")
        for name, n, pos in cur.fetchall():
            print(f"{name}: {n} rows, {pos} positive")


if __name__ == "__main__":
    load()