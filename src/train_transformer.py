"""Fine-tune a lightweight multilingual transformer for 3-class SMS classification.

* Input: PII-masked text with original casing (src.preprocessing.transformer_text).
* Loss: class-weighted cross-entropy (weights = inverse class frequency, normalised),
  so the minority phishing/spam classes are not ignored.
* Model selection: best validation macro-F1 checkpoint; early stopping (patience 2).
* Device: CUDA or Apple MPS if available, otherwise CPU.

Run:  python -m src.train_transformer --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

from src.config import ID2LABEL, LABEL2ID, LABELS, METRICS_DIR, PROCESSED_DIR, SEED, TRANSFORMER_DIR


class SMSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.enc = tokenizer(df.text.tolist(), truncation=True, max_length=max_len)
        self.labels = df.label.map(LABEL2ID).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items() if k != "token_type_ids"}
        item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = logits.argmax(-1)
    rep = classification_report(labels, pred, labels=list(range(3)), output_dict=True, zero_division=0)
    return {"macro_f1": f1_score(labels, pred, average="macro"), "accuracy": rep["accuracy"],
            "phishing_recall": rep["2"]["recall"], "spam_recall": rep["1"]["recall"]}


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def train(model_id: str, epochs: int = 8, lr: float = 5e-5, batch: int = 32, max_len: int = 128,
          warmup: float = 0.1, weight_decay: float = 0.01, out_name: str | None = None) -> dict:
    set_seed(SEED)
    name = out_name or short_name(model_id)
    out_dir = TRANSFORMER_DIR / name
    ckpt_dir = TRANSFORMER_DIR / f"_checkpoints_{name}"
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", keep_default_na=False)
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv", keep_default_na=False)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    counts = train_df.label.value_counts().reindex(LABELS).values.astype(float)
    weights = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float)

    args = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch, per_device_eval_batch_size=64, warmup_ratio=warmup,
        weight_decay=weight_decay, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        logging_strategy="epoch", report_to=[], seed=SEED, dataloader_num_workers=0,
        disable_tqdm=True,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=SMSDataset(train_df, tok, max_len),
        eval_dataset=SMSDataset(val_df, tok, max_len), processing_class=tok,
        compute_metrics=compute_metrics, class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    device = str(trainer.args.device)
    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0
    best = trainer.evaluate()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    history = [h for h in trainer.state.log_history if "eval_macro_f1" in h or "loss" in h]
    info = {
        "model_id": model_id, "saved_to": str(out_dir.relative_to(TRANSFORMER_DIR.parent.parent)),
        "device": device, "epochs_run": int(trainer.state.epoch), "max_epochs": epochs,
        "best_checkpoint": trainer.state.best_model_checkpoint, "lr": lr, "batch_size": batch,
        "max_len": max_len, "warmup_ratio": warmup, "weight_decay": weight_decay,
        "class_weights": dict(zip(LABELS, [round(float(w), 3) for w in weights])),
        "train_time_s": round(train_time, 1), "val_metrics_best": {k: round(v, 4) for k, v in best.items()
                                                                    if isinstance(v, float)},
        "history": history,
    }
    (out_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    (METRICS_DIR / f"transformer_training_{name}.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=2))
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--out_name", default=None)
    a = p.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train(a.model, a.epochs, a.lr, a.batch, a.max_len, out_name=a.out_name)
