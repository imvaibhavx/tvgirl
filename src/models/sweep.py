import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models.evaluate import score

if __name__ == "__main__":
    mlflow.set_experiment("steam-sentiment")
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 5],
        "clf__C": [0.3, 1, 3, 10],
    }
    neg_f1 = make_scorer(f1_score, pos_label=False)  # F1 on the negative class
    search = GridSearchCV(
        pipe, grid, scoring=neg_f1, n_jobs=-1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
    )
    search.fit(train["text"], train["voted_up"])

    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    print(res[["params", "mean_test_score", "std_test_score"]].head(5).to_string())

    best = search.best_estimator_
    pred = best.predict(test["text"])
    metrics = score(test["voted_up"], pred)  # test set used once, for the winner only

    with mlflow.start_run(run_name="tfidf-logreg-tuned"):
        mlflow.log_params({
            "model": "tfidf+logreg", "type": "trained+cv",
            "cv_folds": 5, "grid_size": len(res),
            **{k: str(v) for k, v in search.best_params_.items()},
        })
        mlflow.log_metric("cv_neg_f1", search.best_score_)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("data/processed.dvc")
        mlflow.sklearn.log_model(best, name="model",
                                 registered_model_name="steam-sentiment-model")
        print("TEST:", {k: round(float(v), 3) for k, v in metrics.items()})
        print("cv_neg_f1:", round(search.best_score_, 3))