"""Build the editable 16-slide presentation from the actual results.

Output: docs/SMS_Spam_Phishing_Detection_NLP_Presentation.pptx
Run:    python docs/build_presentation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_data import CM, DIAG, G, INSTITUTE, ROOT, SHOTS, TEAM, YEAR, R, f4  # noqa: E402

OUT = ROOT / "docs" / "SMS_Spam_Phishing_Detection_NLP_Presentation.pptx"
r = R()
NAVY, TEAL, RED, GREY, LIGHT = (RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x2A, 0x9D, 0x8F), RGBColor(0xC0, 0x39, 0x2B),
                                RGBColor(0x55, 0x5F, 0x6D), RGBColor(0xF1, 0xF5, 0xF9))
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height
LR, TR, ONX = "logreg", "transformer", "onnx_pruned_int8"
ox = r.onnx
_n = [0]


def text(slide, x, y, w, h, body, size=18, color=NAVY, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = body if isinstance(body, list) else [body]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(line, tuple):
            r1 = p.add_run()
            r1.text = line[0]
            r1.font.bold = True
            r1.font.size = Pt(size)
            r1.font.color.rgb = color
            r2 = p.add_run()
            r2.text = line[1]
            r2.font.size = Pt(size)
            r2.font.color.rgb = GREY
        else:
            run = p.add_run()
            run.text = line
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
        p.alignment = align
        p.space_after = Pt(8)
    return tb


def bullets(slide, x, y, w, h, items, size=17):
    return text(slide, x, y, w, h, [("• " + a, b) if isinstance(it := i, tuple) and (a := it[0]) and (b := it[1])
                                    else "• " + i for i in items], size=size, color=NAVY)


def new(title, subtitle=None):
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    text(s, Inches(0.5), Inches(0.18), Inches(11.5), Inches(0.7), title, 28, RGBColor(255, 255, 255), True)
    if subtitle:
        text(s, Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.5), subtitle, 15, TEAL, True)
    text(s, Inches(12.3), Inches(7.0), Inches(0.9), Inches(0.4), str(_n[0]), 11, GREY, align=PP_ALIGN.RIGHT)
    text(s, Inches(0.5), Inches(7.0), Inches(9), Inches(0.4),
         "SMS Spam & Phishing Detection (English + Hinglish) | Bankar & Mulani", 10, GREY)
    return s


def pic(slide, path, x, y, w=None, h=None):
    path = Path(path)
    if path.exists():
        return slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    text(slide, x, y, Inches(4), Inches(0.5), f"[missing {path.name}]", 12, RED)


def table(slide, x, y, w, rows, col_w=None, size=12, header_fill=NAVY):
    nr, nc = len(rows), len(rows[0])
    t = slide.shapes.add_table(nr, nc, x, y, w, Inches(0.38 * nr)).table
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v)
            para = c.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            if i == 0:
                para.font.bold = True
                para.font.color.rgb = RGBColor(255, 255, 255)
                c.fill.solid()
                c.fill.fore_color.rgb = header_fill
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    return t


def stat(slide, x, y, value, label, color=TEAL):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(2.9), Inches(1.45))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = color
    text(slide, x, y + Inches(0.12), Inches(2.9), Inches(0.7), value, 30, color, True, PP_ALIGN.CENTER)
    text(slide, x + Inches(0.1), y + Inches(0.82), Inches(2.7), Inches(0.6), label, 12, GREY, align=PP_ALIGN.CENTER)


# 1 ---------------------------------------------------------------- title
_n[0] += 1
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
text(s, Inches(0.8), Inches(1.2), Inches(11.7), Inches(2.0),
     "Detecting Spam and Phishing SMS Messages, Including Hinglish Text, Using Classical ML and Lightweight "
     "Transformers", 34, RGBColor(255, 255, 255), True)
text(s, Inches(0.8), Inches(3.3), Inches(11), Inches(0.6), "MTech Mini-Project | Natural Language Processing",
     20, RGBColor(0x9F, 0xE0, 0xD6))
text(s, Inches(0.8), Inches(4.3), Inches(11), Inches(1.2),
     [f"{n}  ({roll})" for n, roll in TEAM], 20, RGBColor(255, 255, 255))
text(s, Inches(0.8), Inches(5.7), Inches(11.5), Inches(1.0), [INSTITUTE[2] + ", Baramati", YEAR], 14,
     RGBColor(0xCB, 0xD5, 0xE1))

# 2 ---------------------------------------------------------------- intro
s = new("Introduction", "SMS is still the channel banks, couriers and OTPs rely on, and so do fraudsters")
bullets(s, Inches(0.6), Inches(1.8), Inches(7.2), Inches(5), [
    ("Smishing: ", "SMS phishing that pushes victims to click a link, call a number or share OTP / PIN / KYC data"),
    ("Indian patterns: ", "fake KYC updates, UPI refund & collect fraud, 'account band ho jayega', fake delivery fees, "
                          "lottery and job scams"),
    ("Hinglish: ", "Romanized Hindi mixed with English; spellings vary (nahi / nhi / nahin)"),
    ("Constraint: ", "the detector should run on the phone itself: small, fast, private")])
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(2.0), Inches(4.5), Inches(3.4))
box.fill.solid()
box.fill.fore_color.rgb = RGBColor(0xFD, 0xE8, 0xE7)
box.line.color.rgb = RED
text(s, Inches(8.55), Inches(2.2), Inches(4.0), Inches(3.0),
     ["Synthetic example", "\"Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.\"",
      "-> phishing"], 17, RED)

# 3 ---------------------------------------------------------------- problem
s = new("Problem Statement")
text(s, Inches(0.6), Inches(1.4), Inches(12), Inches(1.2),
     "Classify a single SMS (English or Hinglish) as Ham, Spam or Phishing, accurately enough to protect users "
     "and small enough to run on a mobile CPU.", 22, NAVY, True)
table(s, Inches(0.6), Inches(3.0), Inches(12), [
    ["Class", "Meaning", "Why it matters"],
    ["Ham", "legitimate message", "false alarms erode trust"],
    ["Spam", "unwanted promotional / bulk", "annoying, usually low risk"],
    ["Phishing", "steals credentials, OTPs, bank data or money", "missed phishing = financial loss -> recall first"],
], col_w=[2.0, 5.0, 5.0], size=15)

# 4 ---------------------------------------------------------------- objectives
s = new("Project Objectives")
bullets(s, Inches(0.6), Inches(1.4), Inches(12), Inches(5.5), [
    "Build a licensed, privacy-safe 3-class dataset with a documented label-mapping strategy",
    "Add a transparent synthetic Hinglish supplement (Indian scam patterns + everyday messages)",
    "Classical baseline: word + character TF-IDF with Logistic Regression (plus SVM and NB)",
    "Lightweight transformer: select, fine-tune and evaluate a multilingual model",
    "Compare on the same held-out test set, focusing on phishing recall and Hinglish",
    "Compress for mobile (ONNX, INT8, vocabulary pruning) and verify the predictions still match",
    "Working demo with confidence scores and human-readable warning signs"], size=19)

# 5 ---------------------------------------------------------------- literature
s = new("Literature Review")
table(s, Inches(0.5), Inches(1.3), Inches(12.3), [
    ["Work", "Contribution", "Used here for"],
    ["Almeida et al. (2011, 2013)", "SMS Spam Collection; classical ML baselines", "English data, baseline design"],
    ["Mishra & Soni (2020, 2022)", "Smishing detector; ham/spam/smishing dataset", "Phishing labels"],
    ["Devlin et al. (BERT, 2019)", "Pre-trained transformer encoders", "Fine-tuning paradigm"],
    ["Sanh et al. 2019; Wang et al. 2020", "DistilBERT; MiniLM distillation", "Lightweight candidates"],
    ["Conneau et al. 2020; Reimers & Gurevych 2020", "XLM-R vocabulary; multilingual distillation",
     "Selected model + tokenizer"],
    ["Nayak & Joshi 2022 (HingBERT)", "Hinglish-pre-trained BERT", "Hinglish motivation, candidate"],
    ["Jacob et al. 2018; ONNX Runtime", "INT8 quantization; portable inference", "Mobile deployment"],
], col_w=[3.8, 4.7, 3.8], size=13)
text(s, Inches(0.5), Inches(5.9), Inches(12.3), Inches(0.8),
     "Gap: public SMS data is mostly English and binary; Hinglish scam SMS data with clear licences is scarce.",
     15, RED, True)

# 6 ---------------------------------------------------------------- dataset
st_ = r.stats
s = new("Dataset Description", "UCI (CC BY 4.0) + Mishra & Soni smishing (CC BY 4.0) + synthetic Hinglish (labelled)")
table(s, Inches(0.5), Inches(1.7), Inches(6.6), [
    ["Split", "Total", "Ham", "Spam", "Phish", "Hinglish"],
    *[[n, st_["split_sizes"][n], st_["split_label_counts"][n]["ham"], st_["split_label_counts"][n]["spam"],
       st_["split_label_counts"][n]["phishing"], st_["split_language_counts"][n].get("hinglish", 0)]
      for n in ("train", "val", "test")]], size=14)
bullets(s, Inches(0.5), Inches(3.5), Inches(6.6), Inches(3.4), [
    "UCI spam NOT relabelled by us: phishing = Mishra & Soni's smishing label",
    f"{st_['hinglish_synthetic']} synthetic Hinglish SMS from {st_['hinglish_templates']} templates; split by template",
    "0 shared texts / templates between train and test",
    "Limitation: smishing includes prize fraud; spam/smishing boundary is noisy"], size=14)
pic(s, G / "02_class_by_split_and_language.png", Inches(7.3), Inches(1.8), w=Inches(5.8))

# 7 ---------------------------------------------------------------- architecture
s = new("Proposed System Architecture")
pic(s, DIAG / "architecture.png", Inches(0.7), Inches(1.2), w=Inches(11.9))

# 8 ---------------------------------------------------------------- preprocessing
s = new("Data Preprocessing", "Privacy first: every URL, e-mail, phone number and 4+ digit run is masked before saving")
pic(s, DIAG / "preprocessing_flow.png", Inches(0.5), Inches(1.7), w=Inches(12.3))
table(s, Inches(0.6), Inches(5.3), Inches(12.1), [
    ["Raw (synthetic)", "Processed"],
    ["URGENT!!! Your SBI a/c will be BLOCKED. Update KYC at http://sbi-kyc.example/verify",
     "urgent! your sbi account will be blocked. update kyc at <url>"],
], col_w=[6.4, 5.7], size=12)

# 9 ---------------------------------------------------------------- classical
c = r.classical.get("logreg", {})
s = new("Classical ML Baseline: TF-IDF + Logistic Regression")
bullets(s, Inches(0.6), Inches(1.4), Inches(6.4), Inches(5.3), [
    ("Word n-grams (1-2): ", "phrases such as 'claim your prize', 'kyc update karo'"),
    ("Char n-grams (2-5): ", "robust to Hinglish spelling variation (nahi / nhi)"),
    ("Logistic Regression: ", "linear, fast, interpretable per-n-gram weights"),
    ("Imbalance: ", "class_weight = balanced (ham is about 82%)"),
    ("Tuning: ", f"5-fold CV grid search on train only; best {', '.join(f'{k.split('__')[-1]}={v}' for k, v in c.get('best_params', {}).items())}"),
    ("Size: ", f"{r.eff(LR, 'model_size_mb'):.1f} MB, {r.eff(LR, 'median_ms'):.2f} ms per SMS")], size=15)
pic(s, CM / "cm_logreg.png", Inches(7.1), Inches(1.6), w=Inches(5.9))

# 10 --------------------------------------------------------------- transformer
tr = r.tr
s = new("Lightweight Transformer: multilingual MiniLM-L12")
rows = [["Candidate", "Params", "fp32 MB", "CPU ms", "Hinglish pieces/word"]]
for k, v in r.survey.items():
    rows.append([k.replace("paraphrase-multilingual-", "").replace("-base-multilingual-cased", "-multi"),
                 f"{v['params_millions']}M", v["fp32_size_mb"], v["cpu_latency_ms_batch1"],
                 v["tokenisation_hinglish"]["pieces_per_word"]])
table(s, Inches(0.5), Inches(1.3), Inches(7.4), rows, col_w=[3.0, 1.0, 1.1, 1.0, 1.3], size=12)
bullets(s, Inches(0.5), Inches(3.6), Inches(7.4), Inches(3.3), [
    "12 layers, hidden 384, XLM-R SentencePiece vocab (no [UNK] on Hinglish)",
    "82% of parameters in the 250k embedding table, so it is ideal for vocab pruning",
    f"Fine-tuned: weighted CE, lr {tr.get('lr')}, batch {tr.get('batch_size')}, max {tr.get('max_len')} tokens, "
    f"early stopping ({tr.get('epochs_run')} epochs, {tr.get('train_time_s')} s on Apple MPS)",
    "Not pre-trained on Romanized Hindi; Hinglish ability is measured, not assumed"], size=14)
pic(s, G / "transformer_training_curves.png", Inches(8.1), Inches(1.5), w=Inches(5.0))
pic(s, CM / "cm_transformer.png", Inches(8.1), Inches(3.5), w=Inches(5.0))

# 11 --------------------------------------------------------------- results
s = new("Experimental Results", "Same 945-message held-out test set for every model")
rows = [["Model", "Accuracy", "Macro F1", "Phishing recall", "Phishing precision", "Hinglish macro F1",
         "Ham false alarms"]]
for k in (LR, "naive_bayes", "linear_svm", TR, "transformer:distilbert-base-multilingual-cased", ONX):
    if k in r.test:
        o = r.test[k]["overall"]
        rows.append([r.test[k]["display_name"], f4(o["accuracy"]), f4(o["macro_f1"]),
                     f4(o["per_class"]["phishing"]["recall"]), f4(o["per_class"]["phishing"]["precision"]),
                     f4(r.test[k]["hinglish"]["macro_f1"]), f"{100 * o['ham_false_alarm_rate']:.2f}%"])
table(s, Inches(0.4), Inches(1.6), Inches(12.5), rows, col_w=[3.9, 1.3, 1.3, 1.5, 1.6, 1.6, 1.3], size=12)
text(s, Inches(0.4), Inches(5.0), Inches(12.5), Inches(1.8), [
    ("Baseline: ", "best macro-F1 and phishing recall on the (mostly English) test set"),
    ("Transformer: ", "fewer false alarms, higher phishing precision, better on unseen Hinglish templates"),
    ("Both: ", f"only {r.o(LR, 'phishing_predicted_as_ham')} and {r.o(TR, 'phishing_predicted_as_ham')} of 83 "
              f"phishing SMS passed as ham; most errors are spam <-> phishing (noisy source labels)")], 15)

# 12 --------------------------------------------------------------- comparison
s = new("Model Comparison: TF-IDF + LR vs Lightweight Transformer")
o1, o2, o3 = r.test[LR]["overall"], r.test[TR]["overall"], r.test[ONX]["overall"]
table(s, Inches(0.5), Inches(1.3), Inches(7.0), [
    ["Metric", "TF-IDF + LR", "MiniLM (PyTorch)", "MiniLM mobile"],
    ["Accuracy", f4(o1["accuracy"]), f4(o2["accuracy"]), f4(o3["accuracy"])],
    ["Macro precision", f4(o1["macro_precision"]), f4(o2["macro_precision"]), f4(o3["macro_precision"])],
    ["Macro recall", f4(o1["macro_recall"]), f4(o2["macro_recall"]), f4(o3["macro_recall"])],
    ["Macro F1", f4(o1["macro_f1"]), f4(o2["macro_f1"]), f4(o3["macro_f1"])],
    ["Phishing recall", f4(o1["per_class"]["phishing"]["recall"]), f4(o2["per_class"]["phishing"]["recall"]),
     f4(o3["per_class"]["phishing"]["recall"])],
    ["Hinglish macro F1", f4(r.test[LR]["hinglish"]["macro_f1"]), f4(r.test[TR]["hinglish"]["macro_f1"]),
     f4(r.test[ONX]["hinglish"]["macro_f1"])],
    ["Model size", f"{r.eff(LR, 'model_size_mb'):.1f} MB", f"{r.eff(TR, 'model_size_mb'):.0f} MB",
     f"{r.eff(ONX, 'model_size_mb'):.1f} MB"],
    ["Latency / SMS (CPU)", f"{r.eff(LR, 'median_ms'):.2f} ms", f"{r.eff(TR, 'median_ms'):.1f} ms",
     f"{r.eff(ONX, 'median_ms'):.1f} ms"],
    ["RAM increase", f"{r.eff(LR, 'rss_increase_mb')} MB", f"{r.eff(TR, 'rss_increase_mb')} MB",
     f"{r.eff(ONX, 'rss_increase_mb')} MB"],
], col_w=[2.2, 1.6, 1.6, 1.6], size=13)
pic(s, G / "pr_curves_phishing.png", Inches(7.8), Inches(1.3), w=Inches(5.2))

# 13 --------------------------------------------------------------- mobile
e2e = ox.get("end_to_end_model", {})
s = new("Mobile Deployment", "ONNX -> INT8 -> vocabulary pruning -> tokenizer embedded in one file")
pt_mb, pr_mb = ox.get("pytorch_fp32_weights_mb"), ox.get("onnx_sizes_mb", {}).get("model_pruned_int8")
stat(s, Inches(0.6), Inches(1.8), f"{pt_mb:.0f} MB", "PyTorch fp32", GREY)
stat(s, Inches(3.7), Inches(1.8), f"{ox['onnx_sizes_mb']['model_int8']:.0f} MB", "ONNX INT8", NAVY)
stat(s, Inches(6.8), Inches(1.8), f"{pr_mb:.1f} MB", "INT8 + vocab pruning", TEAL)
stat(s, Inches(9.9), Inches(1.8), f"{e2e.get('size_mb', 0):.1f} MB", "single Android file\n(tokenizer inside)", RED)
v = ox["verification"]["model_pruned_int8"]
bullets(s, Inches(0.6), Inches(3.6), Inches(12.2), Inches(3.2), [
    f"Verified on all 945 test SMS: {v['label_agreement_with_pytorch_pct']}% same labels as PyTorch; "
    f"end-to-end file {e2e.get('label_agreement_with_python_onnx_pct')}% identical to the Python ONNX pipeline",
    f"Vocabulary: {ox['vocab_pruning']['kept_vocab']:,} of {ox['vocab_pruning']['original_vocab']:,} rows kept "
    f"(top 20k + all training pieces)",
    f"Latency {e2e.get('latency_cpu', {}).get('2_threads', {}).get('median_ms', '?')} ms per SMS (2 CPU threads), "
    f"about {e2e.get('session_rss_increase_mb')} MB RAM for the session",
    "Android: ONNX Runtime Mobile + onnxruntime-extensions; Kotlin class + masking rules provided",
    "Honest status: verified in ONNX Runtime on desktop; not yet run on a physical phone"], size=15)

# 14 --------------------------------------------------------------- demo
s = new("Working Demo (Gradio): real screenshots")
pic(s, SHOTS / "crop_phishing_hinglish.png", Inches(0.35), Inches(1.25), w=Inches(6.25))
pic(s, SHOTS / "crop_spam_english.png", Inches(6.75), Inches(1.25), w=Inches(6.25))
text(s, Inches(0.35), Inches(6.45), Inches(12.6), Inches(0.5),
     "Left: Hinglish KYC phishing, flagged by both models. Right: the brief's prize message; the models disagree, "
     "and the indicator panel still flags 'Prize / reward claim'. Links are never opened.", 12, GREY)

# 15 --------------------------------------------------------------- conclusion
s = new("Conclusion and Future Scope")
bullets(s, Inches(0.6), Inches(1.4), Inches(6.2), Inches(5.4), [
    f"Tuned TF-IDF + LR is a strong, tiny baseline (macro F1 {f4(o1['macro_f1'])}, 1 MB)",
    f"MiniLM: macro F1 {f4(o2['macro_f1'])}; more precise, better on unseen Hinglish templates",
    f"Compressed {pt_mb / pr_mb:.0f}x for mobile with {v['label_agreement_with_pytorch_pct']}% prediction agreement",
    "Both models rarely let phishing through as ham",
    "Best practical design: run both and flag if either says fraud"], size=16)
text(s, Inches(7.1), Inches(1.4), Inches(5.8), Inches(0.5), "Limitations / future work", 18, RED, True)
bullets(s, Inches(7.1), Inches(2.0), Inches(5.8), Inches(4.8), [
    "Hinglish data is synthetic; test subset only 50 SMS",
    "Broad, noisy smishing labels in public data",
    "Collect consented real Hinglish SMS",
    "URL reputation and sender-ID features",
    "On-device testing on mid-range Android phones"], size=15)

# 16 --------------------------------------------------------------- references
s = new("References")
text(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.7), [
    "[1] Almeida, Hidalgo, Yamakami, 'Contributions to the study of SMS spam filtering', ACM DocEng 2011.",
    "[2] Mishra, Soni, 'Smishing Detector...', Future Generation Computer Systems 108, 2020.",
    "[3] Mishra, Soni, 'SMS Phishing Dataset for ML and Pattern Recognition', Mendeley Data V1, 2022 (CC BY 4.0).",
    "[4] Devlin et al., 'BERT', NAACL 2019.   [5] Vaswani et al., 'Attention is all you need', NeurIPS 2017.",
    "[6] Wang et al., 'MiniLM', NeurIPS 2020.   [7] Sanh et al., 'DistilBERT', arXiv:1910.01108, 2019.",
    "[8] Conneau et al., 'XLM-R', ACL 2020.   [9] Reimers & Gurevych, multilingual distillation, EMNLP 2020.",
    "[10] Kudo & Richardson, 'SentencePiece', EMNLP Demo 2018.",
    "[11] Nayak & Joshi, 'L3Cube-HingCorpus and HingBERT', WILDRE @ LREC 2022.",
    "[12] Jacob et al., 'Quantization ... integer-arithmetic-only inference', CVPR 2018.",
    "[13] ONNX Runtime, https://onnxruntime.ai   [14] Pedregosa et al., 'Scikit-learn', JMLR 2011.",
    "[15] Wolf et al., 'Transformers', EMNLP Demo 2020.   [16] Salton & Buckley, TF-IDF weighting, IP&M 1988.",
    "Full IEEE-style list in the project report."], 14, NAVY)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
