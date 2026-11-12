"""Ablation: how many vocabulary rows can be pruned before accuracy suffers?

For each K (number of most-frequent SentencePiece pieces kept, in addition to all pieces seen in
the training split) the pruned model is exported, INT8-quantized and evaluated on the test split.
Writes results/metrics/pruning_ablation.json and results/graphs/pruning_ablation.png.
The deployed model (models/optimized) is not modified.

Run:  python tools/pruning_ablation.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.predict as P  # noqa: E402
from deployment.export_onnx import LogitsWrapper, export, oov_rate, prune_vocab, quantize  # noqa: E402
from src.config import GRAPHS_DIR, LABELS, METRICS_DIR, OPTIMIZED_DIR, PROCESSED_DIR, TRANSFORMER_DIR  # noqa: E402

KS = [0, 5000, 20000, 50000, 100000]


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    src_dir = TRANSFORMER_DIR / P.DEFAULT_TRANSFORMER
    tok = AutoTokenizer.from_pretrained(src_dir)
    test = pd.read_csv(PROCESSED_DIR / "test.csv", keep_default_na=False)
    texts, y = test.text.tolist(), test.label.values
    hmask = (test.language == "hinglish").values
    ref = P.TransformerPredictor(src_dir).predict_proba(texts).argmax(1)
    rows = []
    tmp = Path(tempfile.mkdtemp())
    for name in ("sentencepiece.bpe.model", "labels.json"):
        shutil.copy(OPTIMIZED_DIR / name, tmp / name)
    orig_dir = P.OPTIMIZED_DIR
    try:
        P.OPTIMIZED_DIR = tmp
        for k in KS:
            model = AutoModelForSequenceClassification.from_pretrained(src_dir).eval()
            model, remap, info = prune_vocab(model, tok, k)
            export(LogitsWrapper(model, remap).eval(), tmp / "abl_fp32.onnx", tok, 128)
            quantize(tmp / "abl_fp32.onnx", tmp / "abl_int8.onnx")
            probs = P.OnnxPredictor("abl_int8").predict_proba(texts)
            pred = np.array(LABELS)[probs.argmax(1)]
            unk = int(remap[tok.unk_token_id])
            rows.append({
                "keep_top": k, "kept_vocab": info["kept_vocab"],
                "size_mb": round((tmp / "abl_int8.onnx").stat().st_size / 1024 ** 2, 2),
                "test_oov_pct": round(100 * oov_rate(tok, remap, unk, test.text), 2),
                "hinglish_oov_pct": round(100 * oov_rate(tok, remap, unk, test[hmask].text), 2),
                "agreement_with_pytorch_pct": round(100 * float((probs.argmax(1) == ref).mean()), 2),
                "macro_f1": round(f1_score(y, pred, average="macro"), 4),
                "hinglish_macro_f1": round(f1_score(y[hmask], pred[hmask], average="macro"), 4),
            })
            print(rows[-1])
    finally:
        P.OPTIMIZED_DIR = orig_dir
        shutil.rmtree(tmp, ignore_errors=True)
    (METRICS_DIR / "pruning_ablation.json").write_text(json.dumps(rows, indent=2))
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d.size_mb, d.macro_f1, "o-", label="test macro-F1")
    ax.plot(d.size_mb, d.hinglish_macro_f1, "s--", label="Hinglish macro-F1")
    for _, rr in d.iterrows():
        ax.annotate(f"K={int(rr.keep_top):,}", (rr.size_mb, rr.macro_f1), fontsize=7, xytext=(3, 5),
                    textcoords="offset points")
    ax.set_xlabel("INT8 model size (MB)")
    ax.set_ylabel("macro-F1")
    ax.set_title("Vocabulary pruning ablation (INT8 ONNX)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "pruning_ablation.png", dpi=150)


if __name__ == "__main__":
    main()
