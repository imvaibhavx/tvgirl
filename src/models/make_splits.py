from pathlib import Path

from sklearn.model_selection import train_test_split

from src.models.baseline import load_reviews

OUT = Path("data/processed")

if __name__ == "__main__":
    df = load_reviews().drop_duplicates("review_id")
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["voted_up"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False, encoding="utf-8")
    test.to_csv(OUT / "test.csv", index=False, encoding="utf-8")
    print(f"train: {len(train)} rows, test: {len(test)} rows")
    print("negative share (train/test):",
          f"{(~train['voted_up']).mean():.3f} / {(~test['voted_up']).mean():.3f}")