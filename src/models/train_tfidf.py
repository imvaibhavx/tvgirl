import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.evaluate import score

if __name__ == "__main__":
    mlflow.set_experiment("steam-sentiment")
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    params = {"ngram_max": 2, "min_df": 2, "C": 1.0, "class_weight": "balanced"}

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, params["ngram_max"]),
                                  min_df=params["min_df"], sublinear_tf=True)),
        ("clf", LogisticRegression(C=params["C"],
                                   class_weight=params["class_weight"], max_iter=1000)),
    ])

    with mlflow.start_run(run_name="tfidf-logreg"):
        mlflow.log_params({"model": "tfidf+logreg", "type": "trained",
                           "train_rows": len(train), "test_rows": len(test), **params})
        pipe.fit(train["text"], train["voted_up"])
        pred = pipe.predict(test["text"])
        metrics = score(test["voted_up"], pred)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("data/processed.dvc")
        print({k: round(float(v), 3) for k, v in metrics.items()})