"""Evaluate all models on the SAME held-out test split and measure efficiency.

Outputs (results/):
* metrics/test_metrics.json         - all metrics for every model
* metrics/comparison_table.csv/.md  - headline comparison table
* metrics/errors_<model>.csv        - misclassified test messages (PII-masked)
* confusion_matrices/*.png          - confusion matrices (counts + row-normalised)
* graphs/pr_curves_phishing.png, graphs/model_comparison.png

Latency: single-message (batch size 1) CPU inference, 4 threads, median over N messages.
Memory: resident-set-size increase after loading the model, measured in a fresh process.

Run:  python -m src.evaluate [--models logreg naive_bayes linear_svm transformer onnx_pruned_int8]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)

from src.config import CM_DIR, GRAPHS_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, ROOT, TRANSFORMER_DIR
from src.predict import load_predictor

PRETTY = {
    "logreg": "TF-IDF + Logistic Regression", "naive_bayes": "TF-IDF + Multinomial NB",
    "linear_svm": "TF-IDF + Linear SVM", "transformer": "mMiniLM-L12 (PyTorch fp32)",
    "transformer:distilbert-base-multilingual-cased": "DistilBERT-multilingual (PyTorch fp32)",
    "onnx_fp32": "mMiniLM-L12 ONNX fp32", "onnx_int8": "mMiniLM-L12 ONNX int8",
    "onnx_pruned_int8": "mMiniLM-L12 ONNX int8 + vocab pruning",
}


def metric_block(y_true, y_pred, probs) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    y_bin = np.array([[t == lab for lab in LABELS] for t in y_true], dtype=int)
    out = {
        "n": len(y_true), "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "per_class": {lab: {k: rep[lab][k] for k in ("precision", "recall", "f1-score", "support")} for lab in LABELS},
        "confusion_matrix": cm.tolist(),
        # security view: phishing missed (false negatives) and any fraud (spam/phishing) let through as ham
        "phishing_false_negative_rate": 1 - rep["phishing"]["recall"],
        "phishing_predicted_as_ham": int(cm[2, 0]),
        "fraud_recall_binary": float(((np.array(y_true) != "ham") & (np.array(y_pred) != "ham")).sum()
                                     / max((np.array(y_true) != "ham").sum(), 1)),
        "ham_false_alarm_rate": float(((np.array(y_true) == "ham") & (np.array(y_pred) != "ham")).sum()
                                      / max((np.array(y_true) == "ham").sum(), 1)),
    }
    present = [i for i in range(3) if y_bin[:, i].any() and not y_bin[:, i].all()]
    if len(present) == 3:
        out["roc_auc_macro_ovr"] = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    out["pr_auc"] = {LABELS[i]: average_precision_score(y_bin[:, i], probs[:, i]) for i in present}
    return out


def plot_cm(cm, title, fname):
    cm = np.array(cm)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=axes[0], cbar=False)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Reds", xticklabels=LABELS, yticklabels=LABELS, ax=axes[1],
                cbar=False, vmin=0, vmax=1)
    for ax, sub in zip(axes, ("counts", "row-normalised (recall)")):
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(sub)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(CM_DIR / fname, dpi=150)
    plt.close(fig)


def latency_ms(pred, texts, n=200, warmup=10) -> dict:
    for t in texts[:warmup]:
        pred.predict_proba([t])
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        pred.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    pred.predict_proba(texts)
    batch_total = time.perf_counter() - t0
    return {"median_ms": float(np.median(times)), "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)), "throughput_msgs_per_s_batched": len(texts) / batch_total}


def memory_mb(model_name: str) -> float | None:
    """RSS increase (MB) when loading + running the model once, measured in a clean subprocess."""
    code = (
        "import psutil,os,gc;p=psutil.Process(os.getpid());import numpy, sklearn, joblib;"
        "import src.predict as P;b=p.memory_info().rss;"
        f"m=P.load_predictor('{model_name}');m.predict(['test message for memory']);gc.collect();"
        "print((p.memory_info().rss-b)/1024**2)"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
        return round(float(r.stdout.strip().splitlines()[-1]), 1)
    except Exception:
        return None


def evaluate(models: list[str]) -> dict:
    import torch
    torch.set_num_threads(4)
    test = pd.read_csv(PROCESSED_DIR / "test.csv", keep_default_na=False)
    texts, y = test.text.tolist(), test.label.tolist()
    results_path = METRICS_DIR / "test_metrics.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in models:
        print(f"== {name}")
        pred = load_predictor(name)
        t0 = time.perf_counter()
        probs = pred.predict_proba(texts)
        y_pred = [LABELS[i] for i in probs.argmax(1)]
        res = {"display_name": PRETTY.get(name, name), "overall": metric_block(y, y_pred, probs)}
        for lang in ("english", "hinglish"):
            m = (test.language == lang).values
            res[lang] = metric_block([y[i] for i in np.where(m)[0]], [y_pred[i] for i in np.where(m)[0]], probs[m])
        res["efficiency"] = {"model_size_mb": round(pred.size_mb(), 2), **latency_ms(pred, texts),
                             "rss_increase_mb": memory_mb(name), "full_test_inference_s": time.perf_counter() - t0}
        tag = name.replace(":", "_")
        plot_cm(res["overall"]["confusion_matrix"], f"{res['display_name']} - test set (n={len(y)})", f"cm_{tag}.png")
        plot_cm(res["hinglish"]["confusion_matrix"], f"{res['display_name']} - Hinglish test subset", f"cm_{tag}_hinglish.png")
        err = test.assign(pred=y_pred, conf=probs.max(1))
        err[err.label != err.pred][["text", "label", "pred", "conf", "language", "source"]].to_csv(
            METRICS_DIR / f"errors_{tag}.csv", index=False)
        np.save(METRICS_DIR / f"probs_{tag}.npy", probs)
        results[name] = res
        o = res["overall"]
        print(f"   acc={o['accuracy']:.4f} macroF1={o['macro_f1']:.4f} phishR={o['per_class']['phishing']['recall']:.3f} "
              f"hinglish macroF1={res['hinglish']['macro_f1']:.3f} size={res['efficiency']['model_size_mb']:.1f}MB "
              f"lat={res['efficiency']['median_ms']:.2f}ms mem={res['efficiency']['rss_increase_mb']}MB")
        results_path.write_text(json.dumps(results, indent=2, default=float))
    write_tables(results)
    plot_pr_curves(results, test)
    plot_comparison(results)
    return results


def write_tables(results: dict):
    rows = []
    for name, r in results.items():
        o, e = r["overall"], r["efficiency"]
        rows.append({
            "Model": r["display_name"], "Accuracy": o["accuracy"], "Macro precision": o["macro_precision"],
            "Macro recall": o["macro_recall"], "Macro F1": o["macro_f1"],
            "Phishing precision": o["per_class"]["phishing"]["precision"],
            "Phishing recall": o["per_class"]["phishing"]["recall"],
            "Phishing FN rate": o["phishing_false_negative_rate"], "Spam recall": o["per_class"]["spam"]["recall"],
            "Fraud recall (spam+phish vs ham)": o["fraud_recall_binary"], "Ham false-alarm rate": o["ham_false_alarm_rate"],
            "ROC-AUC (macro OvR)": o.get("roc_auc_macro_ovr"), "PR-AUC phishing": o["pr_auc"].get("phishing"),
            "Hinglish macro F1": r["hinglish"]["macro_f1"], "Hinglish phishing recall":
                r["hinglish"]["per_class"]["phishing"]["recall"], "English macro F1": r["english"]["macro_f1"],
            "Model size (MB)": e["model_size_mb"], "Latency/SMS median (ms)": e["median_ms"],
            "Latency p95 (ms)": e["p95_ms"], "Throughput (msg/s, batched)": e["throughput_msgs_per_s_batched"],
            "RAM increase (MB)": e["rss_increase_mb"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "comparison_table.csv", index=False)
    t = df.set_index("Model").T
    fmt = t.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else
                (f"{v:.4f}" if isinstance(v, float) and abs(v) <= 1 else (f"{v:.2f}" if isinstance(v, float) else str(v))))
    (METRICS_DIR / "comparison_table.md").write_text(fmt.to_markdown())


def plot_pr_curves(results, test):
    y = (test.label == "phishing").values.astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        f = METRICS_DIR / f"probs_{name.replace(':', '_')}.npy"
        if not f.exists():
            continue
        p = np.load(f)[:, 2]
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{r['display_name']} (AP={average_precision_score(y, p):.3f})")
    ax.set_xlabel("Recall (phishing)")
    ax.set_ylabel("Precision (phishing)")
    ax.set_title("Precision-recall curve, phishing class (test set)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "pr_curves_phishing.png", dpi=150)
    plt.close(fig)


def plot_comparison(results):
    names = list(results)
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("phish_recall", "Phishing recall"),
               ("hinglish_f1", "Hinglish macro F1")]
    vals = {m: [] for m, _ in metrics}
    for n in names:
        o = results[n]["overall"]
        vals["accuracy"].append(o["accuracy"])
        vals["macro_f1"].append(o["macro_f1"])
        vals["phish_recall"].append(o["per_class"]["phishing"]["recall"])
        vals["hinglish_f1"].append(results[n]["hinglish"]["macro_f1"])
    x = np.arange(len(names))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    for i, (m, lab) in enumerate(metrics):
        axes[0].bar(x + (i - 1.5) * w, vals[m], w, label=lab)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([results[n]["display_name"].replace(" + ", "+\n").replace(" (", "\n(") for n in names],
                            fontsize=7)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(fontsize=8, ncol=4, loc="lower center")
    axes[0].set_title("Classification quality (test set)")
    sizes = [results[n]["efficiency"]["model_size_mb"] for n in names]
    lats = [results[n]["efficiency"]["median_ms"] for n in names]
    axes[1].scatter(sizes, lats, s=60)
    for n, s, l in zip(names, sizes, lats):
        axes[1].annotate(results[n]["display_name"].split(" (")[0].replace("mMiniLM-L12 ", ""), (s, l), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Model size on disk (MB, log scale)")
    axes[1].set_ylabel("Median latency per SMS (ms, CPU)")
    axes[1].set_title("Efficiency")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logreg", "naive_bayes", "linear_svm", "transformer"])
    ap.add_argument("--reset", action="store_true", help="discard previous results file")
    a = ap.parse_args()
    if a.reset and (METRICS_DIR / "test_metrics.json").exists():
        (METRICS_DIR / "test_metrics.json").unlink()
    evaluate(a.models)
