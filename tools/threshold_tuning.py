"""Security operating point: tune a phishing decision threshold on the VALIDATION split.

Rule: predict "phishing" if P(phishing) >= t, otherwise the arg-max of ham/spam.
t is chosen on validation as the largest threshold that reaches the target phishing recall
(default 0.90); the test split is only used to report the effect of that fixed threshold.

Run:  python tools/threshold_tuning.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import LABELS, METRICS_DIR, PROCESSED_DIR  # noqa: E402
from src.predict import load_predictor  # noqa: E402

TARGET = 0.90
MODELS = {"logreg": "TF-IDF + Logistic Regression", "onnx_pruned_int8": "mMiniLM-L12 ONNX int8 + vocab pruning"}


def decide(probs, t):
    pred = np.where(probs[:, 2] >= t, 2, probs[:, :2].argmax(1))
    return np.array(LABELS)[pred]


def stats(y, pred, lang=None):
    y, pred = np.asarray(y), np.asarray(pred)
    return {"macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "phishing_recall": round(recall_score(y, pred, labels=["phishing"], average="macro", zero_division=0), 4),
            "phishing_precision": round(precision_score(y, pred, labels=["phishing"], average="macro", zero_division=0), 4),
            "phishing_predicted_as_ham": int(((y == "phishing") & (pred == "ham")).sum()),
            "ham_false_alarm_rate": round(float(((y == "ham") & (pred != "ham")).sum() / max((y == "ham").sum(), 1)), 4)}


def main():
    val = pd.read_csv(PROCESSED_DIR / "val.csv", keep_default_na=False)
    test = pd.read_csv(PROCESSED_DIR / "test.csv", keep_default_na=False)
    out = {"target_val_phishing_recall": TARGET}
    for key, name in MODELS.items():
        m = load_predictor(key)
        pv, pt = m.predict_proba(val.text.tolist()), m.predict_proba(test.text.tolist())
        yv = (val.label == "phishing").values
        ts = np.linspace(0.05, 0.95, 91)
        ok = [t for t in ts if (pv[yv, 2] >= t).mean() >= TARGET]
        t = float(max(ok)) if ok else 0.05
        h = (test.language == "hinglish").values
        out[key] = {"display_name": name, "threshold": round(t, 2),
                    "val_default": stats(val.label, np.array(LABELS)[pv.argmax(1)]),
                    "val_tuned": stats(val.label, decide(pv, t)),
                    "test_default": stats(test.label, np.array(LABELS)[pt.argmax(1)]),
                    "test_tuned": stats(test.label, decide(pt, t)),
                    "test_hinglish_tuned": stats(test.label[h], decide(pt[h], t))}
        print(name, "t =", round(t, 2), "| test default", out[key]["test_default"], "| tuned", out[key]["test_tuned"])
    (METRICS_DIR / "threshold_tuning.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
