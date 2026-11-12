"""Stand-alone, mobile-style inference: ONLY numpy + onnxruntime + sentencepiece.

No PyTorch, no scikit-learn, no project imports. This file mirrors exactly what an
Android/iOS app has to do, so it doubles as the reference implementation for porting:

  1. normalise + mask the SMS (same rules as training: URLs, e-mails, phones, digit runs)
  2. tokenize with sentencepiece.bpe.model (XLM-R ids: <s>=0 <pad>=1 </s>=2 <unk>=3, piece+1)
  3. run model_pruned_int8.onnx  (inputs: input_ids, attention_mask; output: logits)
  4. softmax -> {ham, spam, phishing}

Run:
  python deployment/mobile_inference.py "Aapka SBI account band ho jayega. KYC update karo"
  python deployment/mobile_inference.py --benchmark
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "optimized"

# ---- preprocessing (kept identical to src/preprocessing.py::transformer_text; checked by tests)
_MOJIBAKE = {"ï¿½": "", "�": "", "Â£": "£", "Â": "", "â‚¬": "€", "â€™": "'", "â€˜": "'", "â€œ": '"',
             "â€\x9d": '"', "â€“": "-", "â€”": "-", "â€¦": "...", "&lt;": "<", "&gt;": ">", "&amp;": "&", "&#039;": "'"}
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL = re.compile(
    r"(?:(?:https?://|www\.)\S*)|(?:\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|in|co|uk|io|ly|me|info|biz|"
    r"mobi|xyz|top|online|site|click|link|example|test|do|gl|cc|tk|ml|ga|cf|gq|app|live|club|win)(?:/\S*)?)", re.I)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s-]{7,}\d)(?!\w)")
_GLUED = re.compile(r"\d{7,}")
_NUM = re.compile(r"(?<![\w<])\d{4,}(?![\w>])")
_REPEAT = re.compile(r"(.)\1{2,}")
_PUNCT = re.compile(r"([!?.])\1+")
_SPACE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text).strip()
    text = _EMAIL.sub(" <email> ", text)
    text = _URL.sub(" <url> ", text)
    text = _PHONE.sub(" <phone> ", text)
    text = _GLUED.sub(" <phone> ", text)
    text = _NUM.sub(" <num> ", text)
    text = _SPACE.sub(" ", text).strip()
    text = _REPEAT.sub(r"\1\1", text)
    text = _PUNCT.sub(r"\1", text)
    return _SPACE.sub(" ", text).strip()


class MobileSMSClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR, variant: str = "model_pruned_int8", threads: int = 2):
        cfg = json.loads((model_dir / "labels.json").read_text())
        self.labels, self.max_len, self.pad_id = cfg["labels"], cfg["max_len"], cfg["pad_token_id"]
        self.sp = spm.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.bpe.model"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads           # phones: 2-4 big cores
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_dir / f"{variant}.onnx"), so, providers=["CPUExecutionProvider"])

    def classify(self, sms: str) -> dict:
        pieces = [i + 1 if i != 0 else 3 for i in self.sp.encode(preprocess(sms))]
        ids = [0] + pieces[: self.max_len - 2] + [2]
        input_ids = np.array([ids], dtype=np.int64)
        mask = np.ones_like(input_ids)
        logits = self.sess.run(["logits"], {"input_ids": input_ids, "attention_mask": mask})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(p.argmax())
        return {"label": self.labels[i], "confidence": round(float(p[i]), 4),
                "scores": {lab: round(float(v), 4) for lab, v in zip(self.labels, p)}}


def benchmark(clf: MobileSMSClassifier, n: int = 300) -> dict:
    samples = ["Your appointment is confirmed for tomorrow at 10 AM.",
               "Congratulations! You have won a free gift. Claim now.",
               "Your bank account will be blocked today. Update KYC immediately using this link.",
               "Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.",
               "Kal college lecture 10 baje hai, time pe aa jana."]
    for s in samples:
        clf.classify(s)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        clf.classify(samples[i % len(samples)])
        times.append((time.perf_counter() - t0) * 1000)
    return {"runs": n, "median_ms": round(float(np.median(times)), 2), "p95_ms": round(float(np.percentile(times, 95)), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--variant", default="model_pruned_int8")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    clf = MobileSMSClassifier(variant=a.variant, threads=a.threads)
    for t in a.text:
        print(json.dumps({"sms": t, **clf.classify(t)}, ensure_ascii=False))
    if a.benchmark:
        print(json.dumps({"variant": a.variant, "threads": a.threads, **benchmark(clf)}))
