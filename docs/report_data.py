"""Shared loader for report/presentation builders: every number comes from results/metrics."""
from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "metrics"
G = ROOT / "results" / "graphs"
CM = ROOT / "results" / "confusion_matrices"
DIAG = ROOT / "docs" / "diagrams"
SHOTS = ROOT / "docs" / "screenshots"

TEAM = [("Niraj Bankar", "EN25552002"), ("Zaheer Mulani", "EN25552018")]
TITLE = ("Detecting Spam and Phishing SMS Messages, Including Mixed Hindi-English (Hinglish) Text, "
         "Using Classical Machine Learning and Lightweight Transformers")
SHORT_TITLE = "SMS Spam & Phishing Detection (English + Hinglish)"
INSTITUTE = ["Department of Computer Engineering (AI & DS)", "Vidya Pratishthan's",
             "Kamalnayan Bajaj Institute of Engineering and Technology", "Baramati-413133, Dist-Pune (M.S.) India"]
GUIDE = "Prof. ______________________"   # fill in the course lecturer's name
YEAR = "Academic Year 2025-26"
MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
DISTIL = "distilbert-base-multilingual-cased"


def j(name, default=None):
    p = M / name
    return json.loads(p.read_text()) if p.exists() else default


class R:
    """Convenience accessors over all result files."""

    def __init__(self):
        self.stats = j("dataset_stats.json", {})
        self.eda = j("eda_summary.json", {})
        self.classical = j("classical_training.json", {})
        self.survey = j("model_selection_survey.json", {})
        self.tr = j(f"transformer_training_{MINILM}.json", {})
        self.tr_distil = j(f"transformer_training_{DISTIL}.json", {})
        self.test = j("test_metrics.json", {})
        self.onnx = j("onnx_export_report.json", {})
        self.demo = j("demo_predictions.json", {})
        self.thr = j("threshold_tuning.json", {})
        self.abl = j("pruning_ablation.json", [])

    def m(self, model):
        return self.test.get(model)

    def o(self, model, key):
        return self.test[model]["overall"][key]

    def pc(self, model, cls, key="recall", subset="overall"):
        return self.test[model][subset]["per_class"][cls][key]

    def eff(self, model, key):
        return self.test[model]["efficiency"][key]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f4(x):
    return f"{x:.4f}"


def cmp_word(a, b, higher="higher", lower="lower", same="the same as"):
    if abs(a - b) < 5e-4:
        return same
    return higher if a > b else lower


def env_versions():
    pk = {}
    for name in ["torch", "transformers", "scikit-learn", "onnxruntime", "onnxruntime-extensions", "onnx", "pandas",
                 "numpy", "gradio", "tokenizers"]:
        try:
            pk[name] = metadata.version(name)
        except Exception:
            pass
    return {"python": sys.version.split()[0], "platform": f"{platform.system()} {platform.release()} "
            f"({platform.machine()})", "packages": pk}
