"""Draw the report/presentation diagrams with matplotlib (docs/diagrams/*.png).

architecture.png, preprocessing_flow.png, training_workflow.png, use_case.png
Run:  python docs/build_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)
C = {"data": "#dbeafe", "proc": "#e0f2f1", "model": "#fde68a", "deploy": "#fbcfe8", "out": "#e5e7eb",
     "edge": "#334155"}


def box(ax, x, y, w, h, text, color, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color,
                                ec=C["edge"], lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True,
            fontweight="bold" if bold else "normal")
    return (x, y, w, h)


def arrow(ax, a, b, side_a="r", side_b="l", text=None):
    def pt(bx, s):
        x, y, w, h = bx
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[s]
    p, q = pt(a, side_a), pt(b, side_b)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=C["edge"]))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, ha="center", fontsize=7, color="#475569")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def architecture():
    fig, ax = canvas(14, 7)
    ax.text(7, 6.7, "System architecture: SMS spam & phishing detection (English + Hinglish)", ha="center",
            fontsize=13, fontweight="bold")
    d1 = box(ax, 0.3, 5.0, 2.6, 0.9, "UCI SMS Spam Collection\n(ham/spam, CC BY 4.0)", C["data"])
    d2 = box(ax, 0.3, 3.8, 2.6, 0.9, "Mishra & Soni SMS Phishing\n(ham/spam/smishing, CC BY 4.0)", C["data"])
    d3 = box(ax, 0.3, 2.6, 2.6, 0.9, "Synthetic Hinglish SMS\n(90 templates, labelled synthetic)", C["data"])
    p1 = box(ax, 3.5, 3.7, 2.6, 1.3, "Data pipeline\nPII masking, label mapping,\nde-dup, template-grouped split",
             C["proc"])
    s = box(ax, 3.5, 1.9, 2.6, 1.1, "Train / Val / Test\n4419 / 957 / 945", C["proc"])
    for d in (d1, d2, d3):
        arrow(ax, d, p1)
    arrow(ax, p1, s, "b", "t")
    m1 = box(ax, 6.8, 4.6, 3.0, 1.2, "Baseline\nword+char TF-IDF\n+ Logistic Regression", C["model"])
    m2 = box(ax, 6.8, 2.6, 3.0, 1.4, "Lightweight transformer\nmultilingual MiniLM-L12\n(fine-tuned, 3 classes)",
             C["model"])
    arrow(ax, s, m1, "r", "l")
    arrow(ax, s, m2, "r", "l")
    ev = box(ax, 6.8, 0.6, 3.0, 1.2, "Evaluation\nmacro-F1, phishing recall,\nHinglish subset, latency, size",
             C["out"])
    arrow(ax, m2, ev, "b", "t")
    ax.text(8.3, 2.25, "both models", ha="left", fontsize=7, color="#475569")
    o = box(ax, 10.5, 2.6, 3.2, 1.4, "Mobile optimisation\nONNX -> INT8 -> vocab pruning\n+ embedded tokenizer",
            C["deploy"])
    arrow(ax, m2, o)
    a1 = box(ax, 10.5, 4.6, 3.2, 1.2, "Gradio demo / FastAPI\n(compare both models,\nindicators, no link opening)",
             C["deploy"])
    a2 = box(ax, 10.5, 0.6, 3.2, 1.4, "Android package\nsms_classifier_e2e.onnx +\nORT Mobile (Kotlin sample)",
             C["deploy"])
    arrow(ax, m1, a1)
    arrow(ax, o, a1, "t", "b")
    arrow(ax, o, a2, "b", "t")
    fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def preprocessing():
    fig, ax = canvas(14, 4.2)
    ax.text(7, 3.95, "Data preprocessing pipeline", ha="center", fontsize=13, fontweight="bold")
    steps = ["Raw SMS\n(3 sources)", "Fix encoding\n(mojibake, NFKC)",
             "Mask PII\n<email> <url>\n<phone> <num>", "Label mapping\nsmishing -> phishing\nkeep original labels",
             "De-duplicate\n(masked key)\ndrop conflicts", "Split\nstratified (public)\nby template (synthetic)"]
    xs = [0.2 + i * 2.3 for i in range(len(steps))]
    bxs = [box(ax, x, 2.1, 2.0, 1.3, s, C["proc"] if i else C["data"], fs=8.5) for i, (x, s) in
           enumerate(zip(xs, steps))]
    for a, b in zip(bxs, bxs[1:]):
        arrow(ax, a, b)
    t1 = box(ax, 2.5, 0.2, 4.2, 1.3, "Classical branch (clean_text)\nlower-case, squash 'sooo' -> 'soo',\n"
             "Hinglish/SMS spelling normalisation\n(nhi->nahi, kro->karo, u->you)", C["model"], fs=8)
    t2 = box(ax, 7.4, 0.2, 4.2, 1.3, "Transformer branch (transformer_text)\nkeep casing ('URGENT'),\n"
             "squash repeats; SentencePiece\ntokenizer, max 128 tokens", C["model"], fs=8)
    arrow(ax, bxs[-1], t1, "b", "r")
    arrow(ax, bxs[-1], t2, "b", "r")
    ax.text(12.6, 0.8, "No stemming,\nno stop-word removal\n(keeps 'your', 'abhi',\n'karo' cues)", fontsize=8,
            ha="center", color="#b42318")
    fig.savefig(OUT / "preprocessing_flow.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def training():
    fig, ax = canvas(14, 5)
    ax.text(7, 4.7, "Model training and evaluation workflow", ha="center", fontsize=13, fontweight="bold")
    tr = box(ax, 0.2, 2.0, 2.0, 1.1, "Train split\n(4419)", C["data"])
    a1 = box(ax, 2.8, 3.2, 3.0, 1.1, "TF-IDF word(1-2)+char(2-5)\nGridSearch C, 5-fold CV\nclass_weight=balanced",
             C["model"], fs=8)
    a2 = box(ax, 2.8, 0.8, 3.0, 1.3, "Model survey (4 candidates)\n-> fine-tune MiniLM & DistilBERT\n"
             "weighted CE, AdamW, warm-up", C["model"], fs=8)
    arrow(ax, tr, a1)
    arrow(ax, tr, a2)
    va = box(ax, 6.4, 2.0, 2.2, 1.1, "Val split (957)\nmodel / epoch\nselection, early stop", C["data"], fs=8)
    arrow(ax, a1, va, "r", "t")
    arrow(ax, a2, va, "r", "b")
    te = box(ax, 9.2, 2.0, 2.2, 1.1, "Test split (945)\nused once for\nfinal comparison", C["data"], fs=8)
    arrow(ax, va, te)
    out = box(ax, 11.9, 2.6, 1.9, 1.6, "Metrics\nper-class P/R/F1\nconfusion matrix\nPR/ROC-AUC", C["out"], fs=8)
    out2 = box(ax, 11.9, 0.6, 1.9, 1.6, "Efficiency\nsize, latency,\nRAM, ONNX\nverification", C["out"], fs=8)
    arrow(ax, te, out, "r", "l")
    arrow(ax, te, out2, "r", "l")
    fig.savefig(OUT / "training_workflow.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def use_case():
    fig, ax = canvas(12, 6.5)
    ax.text(6, 6.2, "Use-case diagram", ha="center", fontsize=13, fontweight="bold")
    ax.add_patch(FancyBboxPatch((2.6, 0.4), 6.8, 5.4, boxstyle="round,pad=0.02", fc="#f8fafc", ec=C["edge"]))
    ax.text(6, 5.5, "SMS Phishing Detection System", ha="center", fontsize=10, fontweight="bold")

    def actor(x, y, name):
        ax.add_patch(plt.Circle((x, y + 0.75), 0.18, fill=False, lw=1.5))
        ax.plot([x, x], [y + 0.57, y + 0.05], "k", lw=1.5)
        ax.plot([x - 0.3, x + 0.3], [y + 0.4, y + 0.4], "k", lw=1.5)
        ax.plot([x, x - 0.25], [y + 0.05, y - 0.3], "k", lw=1.5)
        ax.plot([x, x + 0.25], [y + 0.05, y - 0.3], "k", lw=1.5)
        ax.text(x, y - 0.6, name, ha="center", fontsize=9)
        return (x, y + 0.3)

    uses = {"Enter / paste SMS": (4.8, 4.8), "Analyze SMS (classify)": (6.3, 3.95),
            "View confidence scores": (4.6, 3.05), "View suspicious indicators": (4.6, 2.2),
            "Compare both models": (4.6, 1.35), "Train / evaluate models": (7.6, 2.4),
            "Export mobile model": (7.6, 1.2)}
    for t, (x, y) in uses.items():
        ax.add_patch(Ellipse((x, y), 2.8, 0.62, fc="#fff7ed", ec=C["edge"]))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5)
    u = actor(1.2, 3.6, "SMS user")
    d = actor(10.8, 1.6, "Developer /\nresearcher")
    for t in ["Enter / paste SMS", "Analyze SMS (classify)", "View confidence scores",
              "View suspicious indicators", "Compare both models"]:
        x, y = uses[t]
        ax.plot([u[0] + 0.3, x - 1.4], [u[1], y], color="#64748b", lw=1)
    for t in ["Train / evaluate models", "Export mobile model", "Analyze SMS (classify)"]:
        x, y = uses[t]
        ax.plot([d[0] - 0.3, x + 1.4], [d[1], y], color="#64748b", lw=1)
    fig.savefig(OUT / "use_case.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    preprocessing()
    training()
    use_case()
    print("diagrams written to", OUT)
