"""Unified prediction interface for all model types.

    from src.predict import load_predictor
    p = load_predictor("logreg")            # or "transformer", "onnx_int8", "onnx_pruned_int8"
    p.predict(["Aapka SBI account band ho jayega ..."])

CLI:  python -m src.predict --model transformer "Your text here"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.config import CLASSICAL_DIR, LABELS, OPTIMIZED_DIR, TRANSFORMER_DIR
from src.preprocessing import transformer_text

DEFAULT_TRANSFORMER = "paraphrase-multilingual-MiniLM-L12-v2"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class BasePredictor:
    name = "base"
    files: list[Path] = []

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, texts: list[str] | str) -> list[dict]:
        if isinstance(texts, str):
            texts = [texts]
        probs = self.predict_proba(texts)
        out = []
        for p in probs:
            i = int(p.argmax())
            out.append({"label": LABELS[i], "confidence": float(p[i]),
                        "scores": {lab: float(v) for lab, v in zip(LABELS, p)}})
        return out

    def size_mb(self) -> float:
        total = 0
        for f in self.files:
            f = Path(f)
            total += sum(x.stat().st_size for x in f.rglob("*") if x.is_file()) if f.is_dir() else f.stat().st_size
        return total / 1024 ** 2


class ClassicalPredictor(BasePredictor):
    def __init__(self, name: str = "logreg"):
        self.name = name
        path = CLASSICAL_DIR / f"{name}.joblib"
        self.pipe = joblib.load(path)
        self.files = [path]
        self.order = [list(self.pipe.classes_).index(lab) for lab in LABELS]

    def predict_proba(self, texts):
        clf = self.pipe.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            p = self.pipe.predict_proba(texts)
        else:  # LinearSVC: softmax over decision scores (ranking only, not calibrated)
            p = _softmax(self.pipe.decision_function(texts))
        return p[:, self.order]


class TransformerPredictor(BasePredictor):
    def __init__(self, model_dir: Path | str | None = None, device: str = "cpu", max_len: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        model_dir = Path(model_dir or TRANSFORMER_DIR / DEFAULT_TRANSFORMER)
        self.name = f"transformer:{model_dir.name}"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        self.device, self.max_len = device, max_len
        self.files = [p for p in model_dir.iterdir() if p.suffix in {".safetensors", ".bin", ".json", ".model"}
                      or p.name in {"tokenizer.json", "sentencepiece.bpe.model", "vocab.txt"}]

    def predict_proba(self, texts, batch_size: int = 64):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                chunk = [transformer_text(t) for t in texts[i:i + batch_size]]
                enc = self.tok(chunk, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
                enc.pop("token_type_ids", None)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out.append(self.torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
        return np.vstack(out)


class OnnxPredictor(BasePredictor):
    """ONNX Runtime inference. Needs only onnxruntime + sentencepiece (no PyTorch, no transformers).

    Tokenisation uses the 5 MB SentencePiece model with the XLM-R id convention
    (<s>=0, <pad>=1, </s>=2, <unk>=3, piece ids shifted by +1), which reproduces the
    training tokenizer exactly (verified on all train+test messages) while using far less
    memory than loading the 250k-entry tokenizer.json.
    """

    def __init__(self, variant: str = "model_pruned_int8", max_len: int = 128, threads: int = 4):
        import onnxruntime as ort
        import sentencepiece as spm
        path = OPTIMIZED_DIR / f"{variant}.onnx"
        self.name = f"onnx:{variant}"
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.sp = spm.SentencePieceProcessor(model_file=str(OPTIMIZED_DIR / "sentencepiece.bpe.model"))
        self.max_len = max_len
        cfg = json.loads((OPTIMIZED_DIR / "labels.json").read_text())
        self.order = [cfg["labels"].index(lab) for lab in LABELS]
        self.files = [path, OPTIMIZED_DIR / "sentencepiece.bpe.model", OPTIMIZED_DIR / "labels.json"]

    def encode(self, text: str) -> list[int]:
        pieces = [i + 1 if i != 0 else 3 for i in self.sp.encode(transformer_text(text))]
        return [0] + pieces[: self.max_len - 2] + [2]

    def predict_proba(self, texts, batch_size: int = 1):
        # Default batch_size=1 (no padding) mirrors on-device use. With dynamic INT8 quantization the
        # activation scale is computed over the whole input tensor, so padding a message inside a batch
        # changes its quantization range slightly; unpadded single-message inference avoids this.
        out = []
        for i in range(0, len(texts), batch_size):
            encs = [self.encode(t) for t in texts[i:i + batch_size]]
            L = max(len(e) for e in encs)
            ids = np.ones((len(encs), L), dtype=np.int64)          # <pad> = 1
            mask = np.zeros((len(encs), L), dtype=np.int64)
            for j, e in enumerate(encs):
                ids[j, :len(e)] = e
                mask[j, :len(e)] = 1
            logits = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
            out.append(_softmax(logits))
        return np.vstack(out)[:, self.order]


def load_predictor(name: str = "logreg", **kw) -> BasePredictor:
    if name in {"logreg", "linear_svm", "naive_bayes"}:
        return ClassicalPredictor(name)
    if name == "transformer":
        return TransformerPredictor(**kw)
    if name.startswith("transformer:"):
        return TransformerPredictor(TRANSFORMER_DIR / name.split(":", 1)[1], **kw)
    if name.startswith("onnx"):
        variant = {"onnx_fp32": "model_fp32", "onnx_int8": "model_int8",
                   "onnx_pruned_int8": "model_pruned_int8", "onnx": "model_pruned_int8"}[name]
        return OnnxPredictor(variant, **kw)
    raise ValueError(f"Unknown model {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg")
    ap.add_argument("text", nargs="+")
    a = ap.parse_args()
    pred = load_predictor(a.model)
    for t, r in zip(a.text, pred.predict(a.text)):
        print(f"[{r['label']:8s} {r['confidence']:.3f}] {t}")
