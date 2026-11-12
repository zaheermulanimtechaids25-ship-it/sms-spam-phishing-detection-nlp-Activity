"""Exploratory data analysis. Saves all figures to results/graphs/.

Run:  python -m src.eda
"""
from __future__ import annotations

import json
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.config import GRAPHS_DIR, LABELS, METRICS_DIR, PROCESSED_DIR
from src.preprocessing import PLACEHOLDERS, clean_text

PALETTE = {"ham": "#2a9d8f", "spam": "#e9c46a", "phishing": "#e63946"}
# Very common function words are hidden only in the frequency *plots* (not in model input).
PLOT_STOP = set("""a an the to you your i me my is are am was be it of in on for and or at this that with
have has do not no so but if we our can will just get go got its it's u ur 2 4 from by as all up out
what when how call me now are there they he she him her his them then than been about""".split())


def load_all() -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        d = pd.read_csv(PROCESSED_DIR / f"{split}.csv", keep_default_na=False)
        d["split"] = split
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["n_chars"] = df.text.str.len()
    df["n_words"] = df.text.str.split().str.len()
    return df


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / name, dpi=150)
    plt.close(fig)


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df.label.value_counts().reindex(LABELS)
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[l] for l in counts.index])
    ax.bar_label(bars)
    ax.set_title("Class distribution (all splits, after de-duplication)")
    ax.set_ylabel("Messages")
    _save(fig, "01_class_distribution.png")


def plot_class_by_split_and_language(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ct = pd.crosstab(df.split, df.label).reindex(index=["train", "val", "test"], columns=LABELS)
    ct.plot(kind="bar", stacked=True, ax=axes[0], color=[PALETTE[l] for l in LABELS], rot=0)
    axes[0].set_title("Ham / Spam / Phishing per split")
    axes[0].set_ylabel("Messages")
    ct2 = pd.crosstab(df.language, df.label).reindex(columns=LABELS)
    ct2.plot(kind="bar", ax=axes[1], color=[PALETTE[l] for l in LABELS], rot=0)
    axes[1].set_title("Ham / Spam / Phishing by language")
    for c in axes[1].containers:
        axes[1].bar_label(c, fontsize=8)
    _save(fig, "02_class_by_split_and_language.png")


def plot_original_labels(df):
    eng = df[df.language == "english"]
    ct = pd.crosstab(eng.label_ms.replace("", "(not in M&S)"), eng.label_uci.replace("", "(not in UCI)"))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Original UCI label")
    ax.set_ylabel("Mishra & Soni label (used)")
    ax.set_title("Label mapping: original labels of English messages")
    _save(fig, "03_label_mapping.png")


def plot_lengths(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label in LABELS:
        sub = df[df.label == label]
        sns.kdeplot(sub.n_chars.clip(upper=400), ax=axes[0], label=label, color=PALETTE[label], fill=True, alpha=0.25)
    axes[0].set_title("Message length (characters, clipped at 400)")
    axes[0].legend()
    sns.boxplot(data=df, x="label", y="n_words", order=LABELS, hue="label", palette=PALETTE, ax=axes[1], legend=False)
    axes[1].set_title("Words per message")
    axes[1].set_ylim(0, 80)
    _save(fig, "04_message_length.png")


def top_words(texts, n=15):
    c = Counter()
    for t in texts:
        for w in re.findall(r"<\w+>|[a-z][a-z']+", clean_text(t)):
            if w not in PLOT_STOP:
                c[w] += 1
    return c.most_common(n)


def plot_top_words(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    out = {}
    for ax, label in zip(axes, LABELS):
        words = top_words(df[df.label == label].text)
        out[label] = words
        ws, cs = zip(*words)
        ax.barh(ws[::-1], cs[::-1], color=PALETTE[label])
        ax.set_title(f"Most frequent words: {label}")
    _save(fig, "05_top_words.png")
    return out


def plot_wordclouds(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, label in zip(axes, LABELS):
        text = " ".join(clean_text(t) for t in df[df.label == label].text)
        text = re.sub(r"<(\w+)>", r"\1_token", text)
        wc = WordCloud(width=600, height=400, background_color="white", stopwords=PLOT_STOP,
                       colormap="viridis", random_state=42, collocations=False).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label)
    _save(fig, "06_wordclouds.png")


def plot_language(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lang = df.language.value_counts()
    axes[0].pie(lang.values, labels=[f"{k}\n({v})" for k, v in lang.items()], autopct="%1.1f%%",
                colors=["#457b9d", "#f4a261"])
    axes[0].set_title("English vs Hinglish (synthetic) messages")
    src = df.source.value_counts()
    axes[1].barh(src.index, src.values, color="#457b9d")
    axes[1].bar_label(axes[1].containers[0])
    axes[1].set_title("Messages by source")
    _save(fig, "07_language_and_source.png")


def plot_placeholders(df):
    rows = []
    for label in LABELS:
        sub = df[df.label == label]
        for tok in ("<url>", "<phone>", "<num>", "<email>"):
            rows.append({"label": label, "token": tok, "share": sub.text.str.contains(tok, regex=False).mean() * 100})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=d, x="token", y="share", hue="label", palette=PALETTE, ax=ax, hue_order=LABELS)
    ax.set_ylabel("% of messages containing token")
    ax.set_title("Masked URLs / phone numbers / digit runs by class")
    _save(fig, "08_placeholder_tokens.png")
    return d.pivot(index="token", columns="label", values="share").round(1).to_dict()


def main():
    sns.set_theme(style="whitegrid")
    df = load_all()
    plot_class_distribution(df)
    plot_class_by_split_and_language(df)
    plot_original_labels(df)
    plot_lengths(df)
    words = plot_top_words(df)
    plot_wordclouds(df)
    plot_language(df)
    tokens = plot_placeholders(df)
    summary = {
        "total": len(df),
        "length_chars_by_label": df.groupby("label").n_chars.describe().round(1).to_dict(orient="index"),
        "length_words_by_label": df.groupby("label").n_words.describe().round(1).to_dict(orient="index"),
        "top_words": {k: v for k, v in words.items()},
        "placeholder_share_pct": tokens,
    }
    (METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved EDA figures to {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
