"""Dataset acquisition, label mapping, PII masking, de-duplication and splitting.

Run:  python -m src.data_loader

Label mapping strategy (three classes: ham / spam / phishing)
--------------------------------------------------------------
* UCI SMS Spam Collection only has ``ham`` / ``spam``. We do NOT relabel its spam as
  phishing ourselves.
* The Mishra & Soni dataset (CC BY 4.0) contains ``ham`` / ``spam`` / ``smishing``
  labels assigned by its authors, and it re-uses most UCI messages. Where a message is
  present in that dataset, its author-assigned label is used (``smishing`` -> ``phishing``).
* Messages that exist only in UCI keep their UCI label (ham or spam).
* Messages whose two sources disagree on ham vs non-ham are dropped as label conflicts.
* Original labels from both sources are preserved in ``label_uci`` and ``label_ms``.

Limitation: the "phishing" class in the English data follows Mishra & Soni's definition of
smishing, which includes prize/lottery fraud asking victims to call premium numbers. It is
broader than credential phishing, and the spam/smishing boundary in that dataset is noisy.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.config import DATASETS, HINGLISH_DIR, METRICS_DIR, PROCESSED_DIR, RAW_DIR, SEED
from src.hinglish_generator import generate as generate_hinglish
from src.preprocessing import is_hinglish, transformer_text

UCI_FILE = RAW_DIR / "SMSSpamCollection"
MS_FILE = RAW_DIR / "Dataset_5971.csv"
HINGLISH_FILE = HINGLISH_DIR / "hinglish_sms_synthetic.csv"
HEADERS = {"User-Agent": "sms-phishing-nlp-project/1.0 (academic use)"}


# ------------------------------------------------------------------ download
def _download(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def download_uci(force: bool = False) -> None:
    if UCI_FILE.exists() and not force:
        return
    print("Downloading UCI SMS Spam Collection ...")
    with zipfile.ZipFile(io.BytesIO(_download(DATASETS["uci"]["url"]))) as zf:
        UCI_FILE.write_bytes(zf.read("SMSSpamCollection"))
        (RAW_DIR / "uci_readme.txt").write_bytes(zf.read("readme"))


def download_mishra_soni(force: bool = False) -> None:
    if MS_FILE.exists() and not force:
        return
    print("Downloading Mishra & Soni SMS Phishing Dataset (Mendeley Data) ...")
    blob = _download(DATASETS["mishra_soni"]["url"])
    digest = hashlib.sha256(blob).hexdigest()
    if digest != DATASETS["mishra_soni"]["sha256"]:
        raise ValueError(f"Checksum mismatch for Mishra & Soni dataset: {digest}")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        MS_FILE.write_bytes(zf.read("Dataset_5971.csv"))


def download_all(force: bool = False) -> None:
    download_uci(force)
    download_mishra_soni(force)


# ------------------------------------------------------------------ load
def load_uci() -> pd.DataFrame:
    df = pd.read_csv(UCI_FILE, sep="\t", header=None, names=["label", "text"], quoting=3, encoding="utf-8")
    df["label"] = df["label"].str.strip().str.lower()
    return df


def load_mishra_soni() -> pd.DataFrame:
    df = pd.read_csv(MS_FILE, encoding="latin-1")
    df = df.rename(columns={"LABEL": "label", "TEXT": "text"})[["label", "text"]]
    df["label"] = df["label"].str.strip().str.lower().replace({"smishing": "phishing"})
    return df


def dedup_key(text: str) -> str:
    """Key used for de-duplication: PII-masked, lower-cased, punctuation/space-insensitive."""
    t = transformer_text(text).lower()
    return re.sub(r"[^a-z0-9<>]+", " ", t).strip()


def build_english_corpus() -> tuple[pd.DataFrame, dict]:
    uci, ms = load_uci(), load_mishra_soni()
    stats = {"uci_raw": len(uci), "uci_raw_counts": uci.label.value_counts().to_dict(),
             "ms_raw": len(ms), "ms_raw_counts": ms.label.value_counts().to_dict(),
             "uci_missing_text": int(uci.text.isna().sum()), "ms_missing_text": int(ms.text.isna().sum())}
    uci, ms = uci.dropna(subset=["text"]), ms.dropna(subset=["text"])
    uci["key"], ms["key"] = uci.text.map(dedup_key), ms.text.map(dedup_key)
    uci, ms = uci[uci.key != ""], ms[ms.key != ""]

    def agg(df):
        g = df.groupby("key").agg(text=("text", "first"), labels=("label", lambda s: sorted(set(s))))
        return g

    gu, gm = agg(uci), agg(ms)
    keys = gu.index.union(gm.index)
    rows, conflicts = [], {"within_ms": 0, "within_uci": 0, "cross_source": 0}
    for k in keys:
        lu = gu.labels.get(k)
        lm = gm.labels.get(k)
        if lu is not None and len(lu) > 1:
            conflicts["within_uci"] += 1
            continue
        if lm is not None and len(lm) > 1:
            conflicts["within_ms"] += 1
            continue
        lu = lu[0] if lu else None
        lm = lm[0] if lm else None
        if lu and lm and (lu == "ham") != (lm == "ham"):
            conflicts["cross_source"] += 1
            continue
        label = lm or lu
        text = gu.text.get(k) if lu else gm.text.get(k)
        source = "uci+mishra_soni" if (lu and lm) else ("uci" if lu else "mishra_soni")
        rows.append({"text": transformer_text(text), "label": label, "label_uci": lu, "label_ms": lm,
                     "source": source, "language": "english", "category": "", "template_id": ""})
    df = pd.DataFrame(rows)
    stats.update({"english_after_dedup": len(df), "label_conflicts_dropped": conflicts,
                  "english_counts": df.label.value_counts().to_dict(),
                  "english_source_counts": df.source.value_counts().to_dict()})
    return df, stats


def load_hinglish() -> pd.DataFrame:
    if not HINGLISH_FILE.exists():
        generate_hinglish().to_csv(HINGLISH_FILE, index=False)
    df = pd.read_csv(HINGLISH_FILE)
    df["text"] = df.text.map(transformer_text)
    df["label_uci"], df["label_ms"] = None, None
    return df


# ------------------------------------------------------------------ split
def split_english(df: pd.DataFrame):
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)
    return train, val, test


def split_by_template(df: pd.DataFrame):
    """Group split: every template's variants go to exactly one split (no template leakage)."""
    import random
    rng = random.Random(SEED)
    parts = {"train": [], "val": [], "test": []}
    for label, g in df.groupby("label"):
        tids = sorted(g.template_id.unique())
        rng.shuffle(tids)
        n = len(tids)
        n_test, n_val = max(1, round(n * 0.15)), max(1, round(n * 0.15))
        assign = {t: "test" for t in tids[:n_test]}
        assign.update({t: "val" for t in tids[n_test:n_test + n_val]})
        assign.update({t: "train" for t in tids[n_test + n_val:]})
        for split in parts:
            parts[split].append(g[g.template_id.map(assign) == split])
    return tuple(pd.concat(parts[s]) for s in ("train", "val", "test"))


def build_dataset() -> dict:
    download_all()
    eng, stats = build_english_corpus()
    hin = load_hinglish()
    # guard: synthetic texts must not collide with public texts
    hin["key"] = hin.text.map(dedup_key)
    eng["key"] = eng.text.map(dedup_key)
    hin = hin[~hin.key.isin(set(eng.key))].drop_duplicates("key")

    e_tr, e_va, e_te = split_english(eng)
    h_tr, h_va, h_te = split_by_template(hin)
    cols = ["text", "label", "language", "source", "category", "template_id", "label_uci", "label_ms"]
    splits = {}
    for name, e, h in (("train", e_tr, h_tr), ("val", e_va, h_va), ("test", e_te, h_te)):
        d = pd.concat([e, h])[cols].sample(frac=1, random_state=SEED).reset_index(drop=True)
        d["hinglish_detected"] = d.text.map(is_hinglish)
        d.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        splits[name] = d

    # leakage checks
    k = {n: set(splits[n].text.map(dedup_key)) for n in splits}
    t = {n: set(splits[n].template_id.dropna().astype(str)) - {""} for n in splits}
    stats.update({
        "hinglish_synthetic": len(hin), "hinglish_counts": hin.label.value_counts().to_dict(),
        "hinglish_templates": int(hin.template_id.nunique()),
        "split_sizes": {n: len(d) for n, d in splits.items()},
        "split_label_counts": {n: d.label.value_counts().to_dict() for n, d in splits.items()},
        "split_language_counts": {n: d.language.value_counts().to_dict() for n, d in splits.items()},
        "leakage_text_overlap": {"train_val": len(k["train"] & k["val"]), "train_test": len(k["train"] & k["test"]),
                                 "val_test": len(k["val"] & k["test"])},
        "leakage_template_overlap": {"train_test": len(t["train"] & t["test"]), "train_val": len(t["train"] & t["val"])},
        "public_detected_hinglish": int(sum(splits[n][splits[n].language == "english"].hinglish_detected.sum()
                                            for n in splits)),
    })
    (METRICS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats


if __name__ == "__main__":
    print(json.dumps(build_dataset(), indent=2, default=str))
