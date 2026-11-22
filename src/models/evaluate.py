import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.models.baseline import MODEL, predict


def score(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[False, True], zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "neg_precision": p[0],
        "neg_recall": r[0],
        "neg_f1": f[0],
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


if __name__ == "__main__":
    mlflow.set_experiment("steam-sentiment")
    test = pd.read_csv("data/processed/test.csv")

    with mlflow.start_run(run_name="distilbert-sst2-baseline"):
        mlflow.log_params({
            "model": MODEL,
            "type": "pretrained",
            "max_length": 256,
            "test_rows": len(test),
        })
        pred, _ = predict(test["text"].tolist())
        metrics = score(test["voted_up"], pred)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("data/processed.dvc")  # ties this run to the data version
        print({k: round(v, 3) for k, v in metrics.items()})