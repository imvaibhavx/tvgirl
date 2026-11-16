import json
from pathlib import Path

rows = []
for p in Path("data/raw").glob("steam_*.jsonl"):
    rows += [json.loads(line) for line in open(p, encoding="utf-8")]

print("total reviews:", len(rows))

lens = sorted(len(r["text"]) for r in rows)
print("length  min/median/max:", lens[0], lens[len(lens)//2], lens[-1])

# reviews that are mostly non-English letters (Spanish, Russian, ASCII art, etc.)
def ascii_ratio(t):
    return sum(c.isascii() for c in t) / len(t)

weird = [r for r in rows if ascii_ratio(r["text"]) < 0.9]
print("low-ASCII reviews:", len(weird))

print("\n--- 5 random negative reviews ---")
import random
neg = [r for r in rows if not r["voted_up"]]
for r in random.sample(neg, 5):
    print("-", r["text"][:150])