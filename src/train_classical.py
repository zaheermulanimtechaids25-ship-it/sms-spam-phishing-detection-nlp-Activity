"""Train the classical baseline: TF-IDF (word + char n-grams) + Logistic Regression.

Additional experiments: Linear SVM and Multinomial Naive Bayes on the same features.

* Imbalance: class_weight="balanced" (ham is ~82% of the data).
* Hyper-parameters: grid search with 5-fold stratified CV on the TRAINING split only,
  scored by macro-F1. The validation split is used to report the selected model's
  development score; the test split is never touched here.

Run:  python -m src.train_classical
"""
from __future__ import annotations

import json
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import CLASSICAL_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, SEED
from src.feature_extraction import build_vectorizer


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", keep_default_na=False)


def make_pipeline(clf) -> Pipeline:
    return Pipeline([("features", build_vectorizer()), ("clf", clf)])


EXPERIMENTS = {
    "logreg": (LogisticRegression(class_weight="balanced", max_iter=3000, solver="lbfgs", random_state=SEED),
               {"clf__C": [0.3, 1, 3, 10, 30], "features__word__ngram_range": [(1, 1), (1, 2)],
                "features__char__ngram_range": [(2, 4), (2, 5)]}),
    "linear_svm": (LinearSVC(class_weight="balanced", random_state=SEED),
                   {"clf__C": [0.03, 0.1, 0.3, 1]}),
    "naive_bayes": (MultinomialNB(), {"clf__alpha": [0.01, 0.05, 0.1, 0.5]}),
}


def main():
    train, val = load_split("train"), load_split("val")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    summary = {}
    for name, (clf, grid) in EXPERIMENTS.items():
        t0 = time.perf_counter()
        search = GridSearchCV(make_pipeline(clf), grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
        search.fit(train.text, train.label)
        train_time = time.perf_counter() - t0
        pred = search.predict(val.text)
        val_f1 = f1_score(val.label, pred, average="macro")
        rep = classification_report(val.label, pred, labels=LABELS, output_dict=True, zero_division=0)
        summary[name] = {"best_params": {k: str(v) for k, v in search.best_params_.items()},
                         "cv_macro_f1": round(search.best_score_, 4), "val_macro_f1": round(val_f1, 4),
                         "val_phishing_recall": round(rep["phishing"]["recall"], 4),
                         "grid_search_time_s": round(train_time, 1)}
        print(f"{name:12s} cv={search.best_score_:.4f} val_macroF1={val_f1:.4f} "
              f"phish_recall={rep['phishing']['recall']:.3f} params={search.best_params_} ({train_time:.0f}s)")
        joblib.dump(search.best_estimator_, CLASSICAL_DIR / f"{name}.joblib", compress=3)

    # Refit the final (primary) model once more to record a clean single-fit training time
    best = joblib.load(CLASSICAL_DIR / "logreg.joblib")
    t0 = time.perf_counter()
    best.fit(train.text, train.label)
    summary["logreg"]["final_fit_time_s"] = round(time.perf_counter() - t0, 2)
    joblib.dump(best, CLASSICAL_DIR / "logreg.joblib", compress=3)
    (METRICS_DIR / "classical_training.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
