"""Interactive demo: SMS spam / phishing detection for English and Hinglish.

Run locally:   python app/app.py            (opens http://127.0.0.1:7860)
In Colab:      python app/app.py --share    (prints a public gradio.live link)

Safety: the app never opens, fetches or renders links from the SMS. Links shown as
evidence are "defanged" (example.com -> example[.]com) so they are not clickable.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import OPTIMIZED_DIR, TRANSFORMER_DIR  # noqa: E402
from src.explainability import logreg_top_features, suspicious_indicators  # noqa: E402
from src.predict import DEFAULT_TRANSFORMER, load_predictor  # noqa: E402
from src.preprocessing import clean_text, is_hinglish  # noqa: E402

MODELS: dict = {}


def get_models():
    if not MODELS:
        MODELS["TF-IDF + Logistic Regression"] = load_predictor("logreg")
        if (OPTIMIZED_DIR / "model_pruned_int8.onnx").exists():
            MODELS["Transformer (mobile ONNX int8)"] = load_predictor("onnx_pruned_int8")
        elif (TRANSFORMER_DIR / DEFAULT_TRANSFORMER).exists():
            MODELS["Transformer (PyTorch)"] = load_predictor("transformer")
    return MODELS


COLORS = {"ham": ("#1b7f5a", "#e8f6ef", "Looks legitimate (Ham)"),
          "spam": ("#9a6700", "#fff6dd", "Spam: unwanted promotional / bulk message"),
          "phishing": ("#b42318", "#fde8e7", "PHISHING / FRAUD suspected")}


def defang(s: str) -> str:
    s = re.sub(r"https?://", "hxxp://", s, flags=re.I)
    return s.replace(".", "[.]")


def banner(label: str, conf: float, model_name: str) -> str:
    fg, bg, text = COLORS[label]
    warn = ""
    if label == "phishing":
        warn = ("<div style='margin-top:6px'>Do <b>not</b> click links, call numbers, or share OTP / PIN / "
                "card / Aadhaar details. Contact your bank only through its official app or number.</div>")
    elif label == "spam":
        warn = "<div style='margin-top:6px'>Probably harmless advertising, but be careful with links and offers.</div>"
    return (f"<div style='border-left:6px solid {fg};background:{bg};color:#111;padding:10px 14px;border-radius:6px;"
            f"margin-bottom:8px'><div style='font-size:1.15em;font-weight:700;color:{fg}'>{text}</div>"
            f"<div>{html.escape(model_name)} confidence: <b>{conf:.1%}</b></div>{warn}</div>")


def analyze(sms: str, language: str, model_choice: str):
    sms = (sms or "").strip()
    if not sms:
        return "<i>Please enter an SMS message.</i>", {}, {}, "", ""
    models = get_models()
    detected = "Hinglish (code-mixed)" if is_hinglish(sms) else "English"
    lang_note = f"**Language:** {detected}" + (" (auto-detected)" if language == "Auto-detect" else
                                               f" (auto-detected; you selected {language})")
    results = {name: m.predict([sms])[0] for name, m in models.items()}
    names = list(results)
    primary = names[-1] if model_choice == "Compare both" else model_choice
    if primary not in results:
        primary = names[-1]
    show = names if model_choice == "Compare both" else [primary]
    banners = "".join(banner(results[n]["label"], results[n]["confidence"], n) for n in show)
    if model_choice == "Compare both" and len({results[n]["label"] for n in names}) > 1:
        banners += ("<div style='color:#555'>The two models disagree. Treat the message as suspicious and "
                    "do not act on it.</div>")
    lr_scores = results[names[0]]["scores"] if names[0] in show else {}
    tr_scores = results[names[-1]]["scores"] if len(names) > 1 and names[-1] in show else {}

    inds = suspicious_indicators(sms)
    lines = [lang_note, "", "**Suspicious indicators found:**" if inds else "**No rule-based warning signs found.**"]
    for f in inds:
        ev = ", ".join(f"`{defang(e)}`" for e in f["evidence"])
        lines.append(f"- **{f['indicator']}**: {f['why']} Evidence: {ev}")
    lines.append("\n_Links are detected only; this app never opens them._")
    lr = models["TF-IDF + Logistic Regression"]
    lr_label = results["TF-IDF + Logistic Regression"]["label"]
    feats = logreg_top_features(lr.pipe, sms, lr_label)
    feat_md = ""
    if feats:
        feat_md = (f"**Baseline model: n-grams pushing towards '{lr_label}'**\n\n" +
                   "\n".join(f"- `{f}` (+{w})" for f, w in feats))
    feat_md += f"\n\n<sub>Normalised text seen by the baseline (links/numbers masked): `{clean_text(sms)}`</sub>"
    return banners, lr_scores, tr_scores, "\n".join(lines), feat_md


EXAMPLES = [
    ["Your appointment is confirmed for tomorrow at 10 AM.", "Auto-detect", "Compare both"],
    ["Congratulations! You have won a free gift. Claim now.", "Auto-detect", "Compare both"],
    ["Your bank account will be blocked today. Update KYC immediately using this link.", "Auto-detect", "Compare both"],
    ["Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.", "Auto-detect", "Compare both"],
    ["Kal college lecture 10 baje hai, time pe aa jana.", "Auto-detect", "Compare both"],
    ["Aapke Paytm me Rs 2500 ka refund pending hai. Claim karne ke liye UPI PIN aur OTP share karein",
     "Auto-detect", "Compare both"],
    ["Flipkart Big Billion Days: 70% off on mobiles, sirf aaj! Shop now", "Auto-detect", "Compare both"],
]


def build_ui() -> gr.Blocks:
    models = get_models()
    choices = ["Compare both"] + list(models)
    with gr.Blocks(title="SMS Spam & Phishing Detector") as demo:
        gr.Markdown("## SMS Spam & Phishing Detector (English + Hinglish)\n"
                    "MTech NLP mini-project: Niraj Bankar (EN25552002) & Zaheer Mulani (EN25552018). "
                    "Compares a TF-IDF + Logistic Regression baseline with a fine-tuned multilingual MiniLM "
                    "transformer (ONNX, int8, mobile-sized).")
        with gr.Row():
            with gr.Column(scale=3):
                sms = gr.Textbox(label="SMS message", lines=4, placeholder="Paste or type an SMS here ...")
                with gr.Row():
                    lang = gr.Radio(["Auto-detect", "English", "Hinglish"], value="Auto-detect", label="Language")
                    model = gr.Radio(choices, value="Compare both", label="Model")
                btn = gr.Button("Analyze SMS", variant="primary")
            with gr.Column(scale=3):
                verdict = gr.HTML(label="Prediction", elem_id="verdict")
                with gr.Row():
                    lr_out = gr.Label(label="TF-IDF + Logistic Regression: confidence", num_top_classes=3)
                    tr_out = gr.Label(label="Transformer: confidence", num_top_classes=3)
        with gr.Row():
            ind = gr.Markdown()
            feats = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[sms, lang, model], label="Synthetic demo messages (not real SMS)")
        gr.Markdown("<sub>Research prototype. Predictions can be wrong; the phishing class follows the "
                    "Mishra & Soni (2022) dataset definition of smishing, which includes prize/lottery fraud. "
                    "Hinglish training data is synthetic.</sub>")
        btn.click(analyze, [sms, lang, model], [verdict, lr_out, tr_out, ind, feats])
        sms.submit(analyze, [sms, lang, model], [verdict, lr_out, tr_out, ind, feats])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link (Colab)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_ui().launch(share=a.share, server_port=a.port)
