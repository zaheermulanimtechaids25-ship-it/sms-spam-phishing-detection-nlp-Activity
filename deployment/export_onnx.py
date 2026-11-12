"""Export the fine-tuned transformer to ONNX and optimise it for mobile CPUs.

Variants written to models/optimized/:
  model_fp32.onnx         - plain export
  model_int8.onnx         - dynamic INT8 quantization (weights int8, activations quantized at run time)
  model_pruned_int8.onnx  - vocabulary pruning + dynamic INT8 quantization

Vocabulary pruning
------------------
The multilingual MiniLM keeps ~82% of its parameters in a 250k-token embedding table
covering ~100 languages. SMS text uses a tiny fraction of it. We keep:
  * the K most frequent SentencePiece pieces (ids are ordered by unigram frequency), and
  * every piece that occurs in the TRAINING split (never val/test), and
  * all special tokens.
A lookup table (original id -> new id, unknown -> <unk>) is stored inside the ONNX graph,
so the standard SentencePiece tokenizer can be used unchanged on the device. The effect on accuracy
is verified on the test set (agreement with the PyTorch model) and reported.

Run:  python -m deployment.export_onnx [--keep_top 20000]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import LABELS, METRICS_DIR, OPTIMIZED_DIR, PROCESSED_DIR, TRANSFORMER_DIR  # noqa: E402
from src.predict import DEFAULT_TRANSFORMER, OnnxPredictor, TransformerPredictor  # noqa: E402
from src.preprocessing import transformer_text  # noqa: E402


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model, remap: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.register_buffer("remap", remap if remap is not None else torch.zeros(1, dtype=torch.long),
                             persistent=False)
        self.use_remap = remap is not None

    def forward(self, input_ids, attention_mask):
        if self.use_remap:
            input_ids = torch.index_select(self.remap, 0, input_ids.reshape(-1)).reshape(input_ids.shape)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(module: torch.nn.Module, path: Path, tok, max_len: int):
    enc = tok(["Aapka account band ho jayega, abhi KYC update karo"], return_tensors="pt", padding="max_length",
              truncation=True, max_length=32)
    torch.onnx.export(
        module, (enc["input_ids"], enc["attention_mask"]), str(path), input_names=["input_ids", "attention_mask"],
        output_names=["logits"], opset_version=17, do_constant_folding=True, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
    )


def quantize(src: Path, dst: Path):
    """Pre-process + dynamic INT8 quantization. Runs inside a temporary working directory because the
    ONNX shape-inference tools write large temporary files (*.data) into the current directory."""
    import os
    import tempfile
    src, dst = Path(src).resolve(), Path(dst).resolve()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            _quantize(src, dst)
        finally:
            os.chdir(cwd)


def _quantize(src: Path, dst: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = src.with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=False)
    except Exception as e:  # pre-processing is an optimisation, not a requirement
        print("quant_pre_process skipped:", e)
        shutil.copy(src, pre)
    quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm", "Gather"])
    pre.unlink(missing_ok=True)


def prune_vocab(model, tok, keep_top: int):
    train = pd.read_csv(PROCESSED_DIR / "train.csv", keep_default_na=False)
    used = set()
    for t in train.text:
        used.update(tok(transformer_text(t))["input_ids"])
    vocab = model.get_input_embeddings().weight.shape[0]
    keep = sorted(set(range(min(keep_top, vocab))) | used | set(tok.all_special_ids))
    remap = torch.full((vocab,), keep.index(tok.unk_token_id), dtype=torch.long)
    remap[torch.tensor(keep)] = torch.arange(len(keep))
    old = model.get_input_embeddings()
    new = torch.nn.Embedding(len(keep), old.weight.shape[1], padding_idx=None)
    new.weight.data = old.weight.data[torch.tensor(keep)].clone()
    model.set_input_embeddings(new)
    model.config.vocab_size = len(keep)
    return model, remap, {"original_vocab": vocab, "kept_vocab": len(keep), "train_pieces": len(used),
                          "keep_top": keep_top}


def oov_rate(tok, remap, unk_new, texts) -> float:
    total = unk = 0
    for t in texts:
        ids = tok(transformer_text(t))["input_ids"]
        new = remap[torch.tensor(ids)]
        total += len(ids)
        unk += int((new == unk_new).sum()) - ids.count(tok.unk_token_id)
    return unk / max(total, 1)


def main(keep_top: int, max_len: int = 128):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    src_dir = TRANSFORMER_DIR / DEFAULT_TRANSFORMER
    tok = AutoTokenizer.from_pretrained(src_dir)
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    spm_src = src_dir / "sentencepiece.bpe.model"
    if not spm_src.exists():
        spm_src = Path(hf_hub_download("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                       "sentencepiece.bpe.model"))
    shutil.copy(spm_src, OPTIMIZED_DIR / "sentencepiece.bpe.model")
    (OPTIMIZED_DIR / "tokenizer.json").unlink(missing_ok=True)
    (OPTIMIZED_DIR / "labels.json").write_text(json.dumps({"labels": LABELS, "max_len": max_len,
                                                           "pad_token_id": tok.pad_token_id, "tokenizer": "sentencepiece.bpe.model",
                                                           "id_convention": "xlm-r: <s>=0 <pad>=1 </s>=2 <unk>=3, piece_id+1",
                                                           "source_model": DEFAULT_TRANSFORMER}, indent=2))
    report = {"source_model_dir": str(src_dir.relative_to(ROOT))}

    model = AutoModelForSequenceClassification.from_pretrained(src_dir).eval()
    pt_bytes = sum(f.stat().st_size for f in src_dir.glob("*.safetensors"))
    report["pytorch_fp32_weights_mb"] = round(pt_bytes / 1024 ** 2, 1)

    print("Exporting fp32 ...")
    export(LogitsWrapper(model).eval(), OPTIMIZED_DIR / "model_fp32.onnx", tok, max_len)
    print("Quantizing int8 ...")
    quantize(OPTIMIZED_DIR / "model_fp32.onnx", OPTIMIZED_DIR / "model_int8.onnx")

    print("Pruning vocabulary ...")
    pmodel = AutoModelForSequenceClassification.from_pretrained(src_dir).eval()
    pmodel, remap, info = prune_vocab(pmodel, tok, keep_top)
    test = pd.read_csv(PROCESSED_DIR / "test.csv", keep_default_na=False)
    unk_new = int(remap[tok.unk_token_id])
    info["test_token_oov_rate_pct"] = round(100 * oov_rate(tok, remap, unk_new, test.text), 3)
    info["test_hinglish_token_oov_rate_pct"] = round(
        100 * oov_rate(tok, remap, unk_new, test[test.language == "hinglish"].text), 3)
    report["vocab_pruning"] = info
    export(LogitsWrapper(pmodel, remap).eval(), OPTIMIZED_DIR / "model_pruned_fp32.onnx", tok, max_len)
    quantize(OPTIMIZED_DIR / "model_pruned_fp32.onnx", OPTIMIZED_DIR / "model_pruned_int8.onnx")

    for f in OPTIMIZED_DIR.glob("*.onnx"):
        report.setdefault("onnx_sizes_mb", {})[f.stem] = round(f.stat().st_size / 1024 ** 2, 2)

    # ------------------------------------------------------------------ verification
    print("Verifying ONNX outputs against PyTorch on the full test set ...")
    texts = test.text.tolist()
    ref = TransformerPredictor(src_dir).predict_proba(texts)
    ref_lab = ref.argmax(1)
    report["verification"] = {}
    for variant in ("model_fp32", "model_int8", "model_pruned_fp32", "model_pruned_int8"):
        p = OnnxPredictor(variant).predict_proba(texts)
        report["verification"][variant] = {
            "label_agreement_with_pytorch_pct": round(100 * float((p.argmax(1) == ref_lab).mean()), 2),
            "max_abs_prob_diff": round(float(np.abs(p - ref).max()), 5),
            "mean_abs_prob_diff": round(float(np.abs(p - ref).mean()), 6),
            "test_accuracy": round(float((np.array(LABELS)[p.argmax(1)] == test.label.values).mean()), 4),
        }
    report["pytorch_test_accuracy"] = round(float((np.array(LABELS)[ref_lab] == test.label.values).mean()), 4)
    (METRICS_DIR / "onnx_export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep_top", type=int, default=20000)
    ap.add_argument("--max_len", type=int, default=128)
    a = ap.parse_args()
    main(a.keep_top, a.max_len)
