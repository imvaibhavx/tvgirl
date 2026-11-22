import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

from src.ingestion.load import connect

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def load_reviews():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT review_id, item_name, text, voted_up FROM reviews")
        return pd.DataFrame(
            cur.fetchall(), columns=["review_id", "item_name", "text", "voted_up"]
        )


def predict(texts, model=MODEL):
    clf = pipeline(
        "sentiment-analysis", model=model, truncation=True, max_length=256, batch_size=32
    )
    out = clf(texts)
    return [r["label"] == "POSITIVE" for r in out], [r["score"] for r in out]


if __name__ == "__main__":
    df = load_reviews()
    print(f"loaded {len(df)} reviews")

    df["pred"], df["score"] = predict(df["text"].tolist())

    print("\n=== Overall ===")
    print(classification_report(df["voted_up"], df["pred"], target_names=["negative", "positive"]))
    print("confusion matrix [rows=true neg/pos, cols=pred neg/pos]:")
    print(confusion_matrix(df["voted_up"], df["pred"]))

    print("\n=== Per game ===")
    for name, g in df.groupby("item_name"):
        acc = (g["voted_up"] == g["pred"]).mean()
        neg = g[~g["voted_up"]]
        neg_recall = (~neg["pred"]).mean() if len(neg) else float("nan")
        print(f"{name}: accuracy {acc:.2f}, negatives caught {neg_recall:.2f} ({len(neg)} negatives)")

    print("\n=== Confident disagreements (model sure, label says otherwise) ===")
    wrong = df[df["voted_up"] != df["pred"]].sort_values("score", ascending=False).head(8)
    for _, r in wrong.iterrows():
        label = "up" if r["voted_up"] else "down"
        print(f"[{r['item_name']}] label={label}, score={r['score']:.2f}: {r['text'][:120]}")