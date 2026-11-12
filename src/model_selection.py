"""Candidate transformer survey for model selection.

For each candidate we measure, WITHOUT fine-tuning:
* parameter count and fp32 size (embeddings vs encoder),
* vocabulary size,
* tokenisation of our Hinglish/English training texts (tokens per message, words split
  into many pieces, [UNK] rate),
* single-message CPU latency (batch 1, seq len = chosen max_len).

Latency does not depend on the weight values, so models are instantiated from their
config (random weights) to avoid downloading large checkpoints just for timing.
The two most practical candidates are then fine-tuned (src.train_transformer) and the
final choice is made on validation macro-F1 / phishing recall + size + latency.

Run:  python -m src.model_selection
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.config import METRICS_DIR, PROCESSED_DIR

CANDIDATES = {
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "distilbert-base-multilingual-cased": "distilbert-base-multilingual-cased",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "hing-bert (L3Cube, Hinglish)": "l3cube-pune/hing-bert",
}


def survey(max_len: int = 128, n_runs: int = 30) -> dict:
    torch.set_num_threads(4)  # comparable to a mid-range phone/laptop CPU budget
    train = pd.read_csv(PROCESSED_DIR / "train.csv", keep_default_na=False)
    hin = train[train.language == "hinglish"].text.tolist()
    eng = train[train.language == "english"].text.tolist()
    sample = "Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo."
    results = {}
    for short, repo in CANDIDATES.items():
        print(f"-- {short}")
        tok = AutoTokenizer.from_pretrained(repo)
        cfg = AutoConfig.from_pretrained(repo, num_labels=3)
        model = AutoModelForSequenceClassification.from_config(cfg).eval()
        n_params = sum(p.numel() for p in model.parameters())
        emb = model.get_input_embeddings().weight.numel()

        def tok_stats(texts):
            lens = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
            words = [w for t in texts for w in t.split()]
            pieces = [len(tok.tokenize(w)) for w in words[:5000]]
            unk = sum(tok.convert_tokens_to_ids(p) == tok.unk_token_id for t in texts[:500] for p in tok.tokenize(t))
            total = sum(len(tok.tokenize(t)) for t in texts[:500])
            return {"mean_tokens": round(float(np.mean(lens)), 1), "p95_tokens": int(np.percentile(lens, 95)),
                    "p99_tokens": int(np.percentile(lens, 99)), "max_tokens": int(max(lens)),
                    "pieces_per_word": round(float(np.mean(pieces)), 2),
                    "unk_rate_pct": round(100 * unk / max(total, 1), 3)}

        enc = tok([sample], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        enc.pop("token_type_ids", None)
        with torch.inference_mode():
            for _ in range(5):
                model(**enc)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(**enc)
                times.append((time.perf_counter() - t0) * 1000)
        results[short] = {
            "hf_id": repo, "layers": getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", None)),
            "hidden": getattr(cfg, "hidden_size", getattr(cfg, "dim", None)),
            "vocab_size": cfg.vocab_size, "params_millions": round(n_params / 1e6, 1),
            "embedding_params_millions": round(emb / 1e6, 1),
            "embedding_share_pct": round(100 * emb / n_params, 1),
            "fp32_size_mb": round(n_params * 4 / 1024 ** 2, 1),
            "cpu_latency_ms_batch1": round(float(np.median(times)), 1),
            "tokenisation_hinglish": tok_stats(hin), "tokenisation_english": tok_stats(eng),
            "sample_tokens": tok.tokenize(sample),
        }
        print(json.dumps({k: v for k, v in results[short].items() if k != "sample_tokens"}, indent=1))
    (METRICS_DIR / "model_selection_survey.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    survey()
