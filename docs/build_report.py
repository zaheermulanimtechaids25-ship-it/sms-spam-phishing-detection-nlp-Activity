"""Build the MTech project report (Word) from the actual results.

Output: docs/SMS_Spam_Phishing_Detection_NLP_Project_Report.docx
Run:    python docs/build_report.py      (after the pipeline has produced results/metrics/*)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_data import (CM, DIAG, DISTIL, G, GUIDE, INSTITUTE, MINILM, ROOT, SHOTS, TEAM, TITLE, YEAR, R,  # noqa: E402
                         cmp_word, env_versions, f4, pct)

OUT = ROOT / "docs" / "SMS_Spam_Phishing_Detection_NLP_Project_Report.docx"
r = R()
doc = Document()
_fig = [0]
_tab = [0]

# ------------------------------------------------------------------ styles
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.font.size = Pt(12)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for lvl, size in ((1, 16), (2, 13), (3, 12)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Times New Roman"
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
for s in doc.sections:
    s.top_margin = s.bottom_margin = Cm(2.3)
    s.left_margin = Cm(2.8)
    s.right_margin = Cm(2.2)


def page_number_footer():
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        run = p.add_run()
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def toc():
    p = doc.add_paragraph()
    run = p.add_run()
    for kind, text in (("begin", None), (None, 'TOC \\o "1-2" \\h \\z \\u'), ("separate", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    t = p.add_run("Right-click here and choose 'Update Field' (or press F9) to build the table of contents.")
    t.italic = True
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    p.add_run()._r.append(end)


def H(text, level=1):
    return doc.add_heading(text, level)


def P(text, bold_prefix=None, italic=False, align=None, size=None):
    p = doc.add_paragraph()
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    run = p.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def B(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)


def caption(text, kind="Figure"):
    ctr = _fig if kind == "Figure" else _tab
    ctr[0] += 1
    p = doc.add_paragraph()
    run = p.add_run(f"{kind} {ctr[0]}: {text}")
    run.italic = True
    run.font.size = Pt(10)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return f"{kind} {ctr[0]}"


def FIG(path, cap, width=15.5):
    path = Path(path)
    if not path.exists():
        P(f"[missing figure: {path.name}]", italic=True)
        return ""
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return caption(cap)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), color)
    tcPr.append(sh)


def TABLE(header, rows, cap, widths=None, size=9.5):
    ref = caption(cap, "Table")
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(size)
        shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run("" if v is None else str(v))
            run.font.size = Pt(size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return ref


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ------------------------------------------------------------------ derived numbers
LR, TR, NB, SVM = "logreg", "transformer", "naive_bayes", "linear_svm"
DB = f"transformer:{DISTIL}"
ONX = "onnx_pruned_int8"
has = lambda k: k in r.test  # noqa: E731
st_ = r.stats
lr_f1, tr_f1 = r.o(LR, "macro_f1"), r.o(TR, "macro_f1")
lr_pr, tr_pr = r.pc(LR, "phishing"), r.pc(TR, "phishing")
lr_h, tr_h = r.test[LR]["hinglish"]["macro_f1"], r.test[TR]["hinglish"]["macro_f1"]
onx_f1 = r.o(ONX, "macro_f1") if has(ONX) else None
ox = r.onnx
pt_mb = ox.get("pytorch_fp32_weights_mb")
sizes = ox.get("onnx_sizes_mb", {})
pr_mb = sizes.get("model_pruned_int8")
e2e = ox.get("end_to_end_model", {})
vp = ox.get("vocab_pruning", {})
best_overall = max((k for k in r.test), key=lambda k: r.o(k, "macro_f1"))


def fmt_model_row(k):
    o, e = r.test[k]["overall"], r.test[k]["efficiency"]
    return [r.test[k]["display_name"], f4(o["accuracy"]), f4(o["macro_precision"]), f4(o["macro_recall"]),
            f4(o["macro_f1"]), f4(o["per_class"]["phishing"]["recall"]), f4(r.test[k]["hinglish"]["macro_f1"]),
            f"{e['model_size_mb']:.1f}", f"{e['median_ms']:.2f}"]


def training_curve_figure():
    out = G / "transformer_training_curves.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for info, name in ((r.tr, "multilingual MiniLM-L12"), (r.tr_distil, "DistilBERT-multilingual")):
        if not info:
            continue
        h = info["history"]
        ev = [x for x in h if "eval_macro_f1" in x]
        trl = [x for x in h if "loss" in x and "eval_loss" not in x]
        ev = [x for x in ev if x is not ev[-1] or len(ev) == 1 or x["epoch"] != ev[-2]["epoch"]]
        ax[0].plot([x["epoch"] for x in trl], [x["loss"] for x in trl], "o-", label=f"{name} train")
        ax[0].plot([x["epoch"] for x in ev], [x["eval_loss"] for x in ev], "s--", label=f"{name} val")
        ax[1].plot([x["epoch"] for x in ev], [x["eval_macro_f1"] for x in ev], "o-", label=f"{name} macro-F1")
        ax[1].plot([x["epoch"] for x in ev], [x["eval_phishing_recall"] for x in ev], "x:",
                   label=f"{name} phishing recall")
    ax[0].set_title("Weighted cross-entropy loss")
    ax[1].set_title("Validation metrics per epoch")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ================================================================== COVER
page_number_footer()
for _ in range(2):
    doc.add_paragraph()
P("A MINI PROJECT REPORT ON", align=WD_ALIGN_PARAGRAPH.CENTER, size=13)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(TITLE)
run.bold = True
run.font.size = Pt(17)
P("Subject: Natural Language Processing (NLP)", align=WD_ALIGN_PARAGRAPH.CENTER)
P("Domain: Natural Language Processing, Text Classification, Cybersecurity, Machine Learning",
  align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
doc.add_paragraph()
P("Submitted by", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
for name, roll in TEAM:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"{name}").bold = True
    p.add_run(f"   (Roll No. {roll})")
P("Master of Technology (M.Tech.), Computer Engineering", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
P("Under the guidance of", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
P(GUIDE, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
for line in INSTITUTE:
    P(line, align=WD_ALIGN_PARAGRAPH.CENTER)
P(YEAR, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ================================================================== ABSTRACT
H("Abstract")
P(f"SMS remains a primary channel for fraud in India, where scam messages often mix Romanized Hindi and "
  f"English (Hinglish). This project builds and compares two classifiers that label an SMS as ham, spam or "
  f"phishing. The data combine the UCI SMS Spam Collection with the Mishra & Soni SMS Phishing Dataset (both "
  f"CC BY 4.0), whose author-assigned smishing labels provide the phishing class, plus a clearly labelled "
  f"synthetic Hinglish supplement of {st_.get('hinglish_synthetic')} messages generated from "
  f"{st_.get('hinglish_templates')} scam and everyday templates. Personal data such as phone numbers, URLs, "
  f"e-mails and long digit sequences is masked, and splits are leakage-controlled (synthetic data is split by "
  f"template). The classical baseline, word and character TF-IDF with class-balanced Logistic Regression, "
  f"reaches a test macro-F1 of {f4(lr_f1)} and phishing recall of {f4(lr_pr)}. A fine-tuned multilingual "
  f"MiniLM-L12 transformer reaches a macro-F1 of {f4(tr_f1)} and phishing recall of {f4(tr_pr)}; on the unseen "
  f"Hinglish templates the macro-F1 values are {f4(lr_h)} (baseline) and {f4(tr_h)} (transformer). "
  + (f"For mobile deployment the transformer was exported to ONNX, quantized to INT8 and vocabulary-pruned, "
     f"shrinking it from {pt_mb:.0f} MB to {pr_mb:.1f} MB while keeping a test macro-F1 of {f4(onx_f1)}. "
     if pr_mb and onx_f1 else "")
  + (f"A single {e2e.get('size_mb', 0):.1f} MB ONNX file with the tokenizer embedded was verified in ONNX Runtime "
     f"with the extensions library used on Android. " if e2e else "")
  + "A Gradio demo compares both models and explains suspicious indicators without opening any link.")
P("Keywords: SMS phishing, smishing, spam detection, Hinglish, code-mixing, TF-IDF, logistic regression, "
  "multilingual transformer, MiniLM, ONNX, quantization, mobile inference.", italic=True)
P("Use of AI tools: an AI coding assistant (Claude) was used to help draft source code and documentation. "
  "All experiments were executed by the team, and every reported number is produced by the code in the "
  "project repository.", italic=True, size=10)
page_break()
H("Table of Contents")
toc()
page_break()

# ================================================================== 1 INTRODUCTION
H("1. Introduction")
P("Short Message Service (SMS) is still the default channel for banks, delivery companies, government services "
  "and one-time passwords (OTPs). That trust makes it attractive to fraudsters. 'Smishing' (SMS phishing) "
  "messages imitate banks or service providers and try to make the victim click a link, call a number, or "
  "reveal an OTP, PIN or card details. In India, common patterns include fake KYC/PAN/Aadhaar updates, UPI "
  "collect-request fraud, threats of account or SIM blocking, fake delivery fees, lottery prizes and fake "
  "job offers.")
P("Indian users often write, and scammers often send, messages in Hinglish: Hindi written in the Roman script and "
  "mixed with English words ('Aapka SBI account band ho jayega, KYC update karo'). Most SMS spam research and "
  "datasets are English-only, and classical keyword filters break when spellings vary ('nahi', 'nhi', 'nahin'). "
  "Multilingual transformers are pre-trained on many languages, but Romanized Hindi is not a standard "
  "pre-training language, so their behaviour on Hinglish must be measured rather than assumed.")
P("This mini-project develops an end-to-end, reproducible system: legally usable datasets, privacy-preserving "
  "preprocessing, a classical machine-learning baseline, a lightweight transformer, a rigorous comparison on the "
  "same held-out test set, an optimised model that is small enough for a phone, and a working demo.")

H("2. Problem Statement")
P("Given the text of a single SMS in English or Hinglish, classify it as ham (legitimate), spam (unwanted "
  "promotional/bulk message) or phishing (fraud seeking credentials, OTPs, banking details, money or other "
  "sensitive information). The classifier must (i) favour high phishing recall, because a missed phishing message "
  "can cause financial loss; (ii) keep false alarms on legitimate messages low; (iii) handle spelling variation and "
  "code-mixing; and (iv) be small and fast enough to run on a mobile CPU.")

H("3. Research Objectives")
B([("O1. ", "Build a three-class dataset from publicly available, licensed sources, with a documented label-mapping "
             "strategy, PII masking and leakage-safe splits."),
   ("O2. ", "Create a transparent synthetic Hinglish supplement that covers common Indian scam patterns and "
             "legitimate Hinglish messages."),
   ("O3. ", "Implement a classical baseline (TF-IDF + Logistic Regression) and additional classical experiments."),
   ("O4. ", "Select and fine-tune a lightweight multilingual transformer, based on size, multilingual tokenisation, "
             "CPU latency and validation performance."),
   ("O5. ", "Compare the models with accuracy, precision, recall, F1, macro-F1, per-class metrics, confusion "
             "matrices and PR/ROC-AUC, with emphasis on phishing recall and the Hinglish subset."),
   ("O6. ", "Optimise the transformer for mobile (ONNX, INT8 quantization, vocabulary pruning) and verify "
             "that the optimised model's predictions match the original."),
   ("O7. ", "Deliver a working demo that shows predictions, confidence scores and human-readable indicators.")])
P("Research questions: (RQ1) Does a fine-tuned lightweight multilingual transformer outperform a well-tuned "
  "TF-IDF baseline on English and Hinglish SMS? (RQ2) How much can the transformer be compressed for mobile "
  "without losing accuracy? (RQ3) How well do both models generalise to Hinglish scam templates never seen in "
  "training?")

# ================================================================== 4 LITERATURE
H("4. Literature Review")
P("SMS spam filtering was popularised by Almeida et al. [1], who released the SMS Spam Collection and showed that "
  "classical classifiers such as Naive Bayes and linear SVMs on bag-of-words features are strong baselines; "
  "a follow-up study [2] extended these results. TF-IDF weighting [3] with linear models remains a standard, "
  "interpretable baseline for short-text classification. For phishing specifically, Mishra and Soni proposed "
  "a smishing detector that combines SMS content analysis with URL behaviour analysis [4], and published a "
  "labelled ham/spam/smishing dataset [5], which provides our phishing labels.")
P("Transformers [6] and BERT [7] made pre-trained contextual encoders the state of the art for text "
  "classification. Because they are large, compressed variants were proposed: DistilBERT [8] reduces BERT "
  "through knowledge distillation, and MiniLM [9] distils the self-attention module into a small student. "
  "Cross-lingual models such as XLM-RoBERTa [10] share a SentencePiece [11] vocabulary across about 100 languages. "
  "Reimers and Gurevych [12] distilled multilingual sentence encoders, including the paraphrase-multilingual "
  "MiniLM-L12 model used here, from an English teacher.")
P("Code-mixed Hindi-English text is common on Indian social media [13]. The L3Cube HingBERT models [14] were "
  "pre-trained on Romanized Hinglish text, which shows that Hinglish needs dedicated attention. For deployment, "
  "integer quantization [15] and ONNX Runtime [16] are widely used to run neural networks efficiently on "
  "CPUs and mobile devices.")
TABLE(["Work", "Focus", "Method", "Relevance to this project"], [
    ["Almeida et al. [1], [2]", "SMS spam", "Classical ML on bag-of-words", "UCI dataset; classical baseline design"],
    ["Mishra & Soni [4], [5]", "Smishing", "Content + URL features; labelled dataset", "Phishing labels (smishing)"],
    ["Salton & Buckley [3]", "Term weighting", "TF-IDF", "Baseline features"],
    ["Devlin et al. [7]", "Pre-trained encoders", "Masked-LM transformer", "Fine-tuning paradigm"],
    ["Sanh et al. [8]; Wang et al. [9]", "Compression", "Distillation (DistilBERT, MiniLM)", "Lightweight candidates"],
    ["Conneau et al. [10]; Reimers & Gurevych [12]", "Multilingual", "XLM-R vocab; multilingual distillation",
     "Selected model's tokenizer and weights"],
    ["Nayak & Joshi [14]; Bali et al. [13]", "Hinglish", "Hinglish BERT; code-mixing analysis",
     "Hinglish motivation; candidate HingBERT"],
    ["Jacob et al. [15]; ONNX Runtime [16]", "Efficient inference", "INT8 quantization; ONNX",
     "Mobile optimisation"],
], "Summary of related work", widths=[4.2, 2.6, 4.2, 4.6])
P("Research gap: public SMS datasets are mostly English and binary (ham/spam), Hinglish scam SMS datasets with "
  "clear licences are scarce, and few student-scale studies verify that a compressed transformer still makes "
  "the same predictions after mobile optimisation. This project addresses these gaps on a small, transparent scale.")

# ================================================================== 5 DATASET
H("5. Dataset Description")
TABLE(["Source", "Labels", "Raw size", "Licence", "Use"], [
    ["UCI SMS Spam Collection [1]", "ham / spam", f"{st_['uci_raw']:,}", "CC BY 4.0", "primary English data"],
    ["Mishra & Soni SMS Phishing [5]", "ham / spam / smishing", f"{st_['ms_raw']:,}", "CC BY 4.0",
     "phishing labels, extra English messages"],
    ["Synthetic Hinglish (this project)", "ham / spam / phishing",
     f"{st_['hinglish_synthetic']} (from {st_['hinglish_templates']} templates)", "MIT",
     "Hinglish supplement (synthetic, labelled as such)"],
], "Datasets used", widths=[4.3, 3.0, 2.6, 2.0, 4.0])
H("5.1 Label mapping strategy", 2)
P(f"UCI has only ham/spam. Its spam class was not relabelled as phishing by the team. Instead, the Mishra & "
  f"Soni dataset, which re-uses most UCI messages, provides author-assigned smishing labels, mapped to phishing. "
  f"Messages found only in UCI keep their UCI label. Original labels are kept in the columns label_uci and "
  f"label_ms. {st_['label_conflicts_dropped']['within_ms']} messages with conflicting labels inside the Mishra & "
  f"Soni file were dropped; no ham-vs-non-ham conflicts were found between the two sources. After PII masking and "
  f"de-duplication, {st_['english_after_dedup']:,} unique English messages remain "
  f"({st_['english_counts']['ham']:,} ham, {st_['english_counts']['spam']} spam, "
  f"{st_['english_counts']['phishing']} phishing).")
P("Limitation: Mishra & Soni define smishing broadly, including prize and lottery fraud that asks victims to call "
  "premium-rate numbers. So the English 'phishing' class is broader than credential theft, and the spam/smishing "
  "boundary in that dataset is noisy.", italic=True)
H("5.2 Synthetic Hinglish supplement", 2)
P("No public, clearly licensed Hinglish scam-SMS dataset was found, so the team wrote 90 templates (32 phishing, "
  "26 spam, 32 ham) that imitate publicly reported Indian scam patterns: fake KYC/PAN/Aadhaar updates, UPI "
  "refund/collect fraud, account/SIM/electricity blocking, OTP requests, fake delivery and customs fees, "
  "KBC/lottery prizes, fake jobs, tax refunds, loan fees, and family or police impersonation. Ham templates "
  "include college, family, work and legitimate transaction alerts (including OTP and debit alerts that resemble "
  "scams). Slot filling and spelling perturbations (nahi/nhi, karo/kro, case changes) create variants. URLs use "
  "the reserved .example domain and phone numbers use the unallocated +91 55555 range. These messages are "
  "synthetic and are reported as such throughout.")
s = st_["split_label_counts"]
sl = st_["split_language_counts"]
TABLE(["Split", "Total", "Ham", "Spam", "Phishing", "English", "Hinglish (synthetic)"],
      [[n, st_["split_sizes"][n], s[n]["ham"], s[n]["spam"], s[n]["phishing"], sl[n].get("english", 0),
        sl[n].get("hinglish", 0)] for n in ("train", "val", "test")],
      "Final processed dataset (after masking and de-duplication)")
P(f"Leakage controls: public data is split with a stratified 70/15/15 split; synthetic data is split by template, so "
  f"all variants of a template stay in one split. Checks found {st_['leakage_text_overlap']['train_test']} identical "
  f"texts and {st_['leakage_template_overlap']['train_test']} shared templates between train and test.")
FIG(G / "02_class_by_split_and_language.png", "Class distribution per split and per language")
FIG(G / "04_message_length.png", "Message length distribution by class")
FIG(G / "05_top_words.png", "Most frequent words per class (function words hidden only in the plot)")
FIG(G / "06_wordclouds.png", "Word clouds per class (placeholder tokens shown as url_token, phone_token)")
FIG(G / "08_placeholder_tokens.png", "Share of messages containing masked URLs / phone numbers / digit runs")

# ================================================================== 6 ARCHITECTURE
H("6. System Architecture")
P("The system has five layers: data acquisition and preparation; two model branches; evaluation; "
  "mobile optimisation; and user-facing applications (Gradio demo, FastAPI endpoint, and an Android package).")
FIG(DIAG / "architecture.png", "System architecture")
FIG(DIAG / "use_case.png", "Use-case diagram", width=13)

# ================================================================== 7 METHODOLOGY
H("7. Proposed Methodology")
B(["Acquire the public datasets automatically (with SHA-256 verification) and generate the synthetic Hinglish data.",
   "Mask PII, map labels to three classes, de-duplicate, and split without leakage.",
   "Train the classical baseline with cross-validated hyper-parameter search on the training split.",
   "Survey four transformer candidates; fine-tune the two most practical ones; select on validation metrics.",
   "Evaluate every model once on the same held-out test split, overall and per language.",
   "Export the selected transformer to ONNX, quantize to INT8, prune its vocabulary, embed the tokenizer, and "
   "verify the optimised models against the PyTorch reference on the full test set.",
   "Serve both models in a demo with rule-based indicators and per-n-gram explanations."], style="List Number")
FIG(DIAG / "training_workflow.png", "Model training and evaluation workflow")

# ================================================================== 8 PREPROCESSING
H("8. Data Preprocessing")
FIG(DIAG / "preprocessing_flow.png", "Data preprocessing pipeline")
B([("Encoding repair: ", "fixes mojibake such as 'Â£' -> '£' from the public files; Unicode NFKC normalisation."),
   ("PII masking: ", "e-mails -> <email>, URLs (including bare domains) -> <url>, phone numbers including digits "
                     "glued to words -> <phone>, remaining 4+ digit runs (OTP, PIN, account fragments) -> <num>. "
                     "A unit test checks that no 9+ digit number or URL remains in the processed data."),
   ("Normalisation: ", "repeated characters squashed ('sooo' -> 'soo'), repeated punctuation reduced, whitespace "
                       "normalised; lower-casing for the classical branch only (the transformer keeps case, e.g. "
                       "'URGENT')."),
   ("Hinglish handling: ", "Romanized Hindi is preserved, not translated. Frequent spelling variants are unified "
                           "(nhi/nahin -> nahi, kro -> karo, bhjo -> bhejo, apka -> aapka). Ambiguous short forms "
                           "('h', 'he') are only normalised in messages detected as Hinglish."),
   ("No stemming / stop-word removal: ", "words like 'your', 'now', 'abhi', 'karo' are phishing cues."),
   ("Language heuristic: ", "a message is tagged Hinglish when it contains at least two Romanized-Hindi function "
                            "words, including one strong marker (e.g. 'hai', 'karo', 'nahi'). This is used for "
                            "EDA and the demo hint only."),
   ("Label encoding: ", "ham = 0, spam = 1, phishing = 2.")])
TABLE(["Input", "After masking + normalisation (classical branch)"], [
    ["URGENT!!! Your SBI a/c will be BLOCKED. Update KYC at http://sbi-kyc.example/verify or call +91 5555501234",
     "urgent! your sbi account will be blocked. update kyc at <url> or call <phone>"],
    ["Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.",
     "aapka sbi account band ho jayega. kyc update karne ke liye abhi link open karo."],
    ["Sooooo happy!!! mail me at test.user@example.com", "soo happy! mail me at <email>"],
], "Preprocessing examples", widths=[8.0, 7.5])

# ================================================================== 9 CLASSICAL
H("9. Classical ML Implementation")
P("TF-IDF represents a message by weighted n-gram counts: terms frequent in a message but rare in the corpus "
  "get high weights. Word n-grams (1-2) capture phrases such as 'claim your prize' or 'kyc update karo'. "
  "Character n-grams (char_wb, 2-5) are robust to SMS and Hinglish spelling variation, because 'nahi', 'nhi' "
  "and 'nahin' share character n-grams. The two feature blocks are concatenated (FeatureUnion) with "
  "sub-linear TF scaling. Logistic Regression is a linear model that turns a weighted sum of features into class "
  "probabilities with softmax. It is fast, well calibrated for its size, interpretable (each n-gram has a "
  "weight per class), and a standard strong baseline for short-text classification. Class imbalance (ham is about "
  "82% of the data) is handled with class_weight='balanced'.")
rows = []
for k, name in ((LR, "TF-IDF + Logistic Regression"), (SVM, "TF-IDF + Linear SVM"), (NB, "TF-IDF + Multinomial NB")):
    c = r.classical.get(k, {})
    rows.append([name, "; ".join(f"{a.split('__')[-1]}={b}" for a, b in c.get("best_params", {}).items()),
                 c.get("cv_macro_f1"), c.get("val_macro_f1"), c.get("val_phishing_recall"),
                 c.get("grid_search_time_s")])
TABLE(["Model", "Selected hyper-parameters", "5-fold CV macro-F1", "Val macro-F1", "Val phishing recall",
       "Grid search time (s)"], rows, "Classical models: grid search (training split only) and validation scores",
      widths=[3.5, 4.6, 2.0, 1.8, 2.0, 1.8], size=8.5)
P(f"The final Logistic Regression refit takes {r.classical.get('logreg', {}).get('final_fit_time_s', 'n/a')} s on "
  f"the CPU, and the saved pipeline (vectoriser + model) is {r.eff(LR, 'model_size_mb'):.1f} MB.")

# ================================================================== 10 TRANSFORMER
H("10. Lightweight Transformer Implementation")
H("10.1 Model selection", 2)
rows = []
for k, v in r.survey.items():
    rows.append([k, v["layers"], v["hidden"], f"{v['vocab_size']:,}", v["params_millions"], v["embedding_share_pct"],
                 v["fp32_size_mb"], v["cpu_latency_ms_batch1"], v["tokenisation_hinglish"]["pieces_per_word"],
                 v["tokenisation_hinglish"]["p99_tokens"]])
TABLE(["Candidate", "Layers", "Hidden", "Vocab", "Params (M)", "Embedding share %", "fp32 MB",
       "CPU ms (batch 1, 128 tokens)", "Hinglish pieces/word", "Hinglish p99 tokens"], rows,
      "Transformer candidate survey (measured on this machine, 4 CPU threads)", size=8)
mini = r.survey.get(MINILM, {})
P(f"multilingual MiniLM-L12 (paraphrase-multilingual-MiniLM-L12-v2 [12]) was selected. It was the fastest candidate "
  f"({mini.get('cpu_latency_ms_batch1')} ms per message on CPU), produced the fewest SentencePiece pieces per "
  f"Hinglish word among the multilingual models, and has no unknown tokens on our data. {mini.get('embedding_share_pct')}% "
  f"of its {mini.get('params_millions')} M parameters sit in a 250k-token multilingual embedding table, which "
  f"makes it ideal for vocabulary pruning: the encoder itself is small. XLM-R base is about twice as slow, and "
  f"HingBERT, although pre-trained on Hinglish, is a 12-layer, 768-wide BERT-base with similar latency. "
  f"DistilBERT-multilingual was fine-tuned as a second candidate for comparison.")
H("10.2 Architecture", 2)
P("MiniLM-L12-H384 is a 12-layer transformer encoder with hidden size 384 and 12 attention heads. Each layer has "
  "multi-head self-attention, where every token attends to every other token with softmax(QK^T/sqrt(d))V, "
  "followed by a feed-forward network, with residual connections and layer normalisation. The input is tokenized "
  "with the XLM-R SentencePiece unigram tokenizer, which splits unseen Hinglish words into sub-word pieces "
  "('jayega' -> 'ja', 'y', 'ega') instead of mapping them to an unknown token. A classification head (dense "
  "layer on the pooled first-token representation) outputs three logits.")
P("Why it is suitable for Hinglish: its sub-word vocabulary covers Latin-script text from many languages, so "
  "Romanized Hindi is represented without unknown tokens, and fine-tuning on Hinglish examples adapts it to the "
  "task. It was not pre-trained on Romanized Hindi, so we do not claim it 'understands' Hinglish; its Hinglish "
  "performance is measured on unseen templates in Section 12.", italic=True)
H("10.3 Fine-tuning", 2)
tr = r.tr
TABLE(["Setting", "Value"], [
    ["Pre-trained checkpoint", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
    ["Max sequence length", f"{tr.get('max_len')} tokens (English p99 = 69, Hinglish p99 = 46 tokens)"],
    ["Optimizer / LR / schedule", f"AdamW, lr = {tr.get('lr')}, linear decay, warm-up ratio {tr.get('warmup_ratio')}"],
    ["Batch size / weight decay", f"{tr.get('batch_size')} / {tr.get('weight_decay')}"],
    ["Loss", "cross-entropy with class weights " + ", ".join(f"{k}={v}" for k, v in tr.get('class_weights', {}).items())],
    ["Epochs", f"max {tr.get('max_epochs')}, early stopping (patience 2) on validation macro-F1; ran "
               f"{tr.get('epochs_run')} epochs, best checkpoint restored"],
    ["Hardware / time", f"{tr.get('device')} (Apple M5 GPU via PyTorch MPS), {tr.get('train_time_s')} s"],
    ["Best validation", f"macro-F1 {tr.get('val_metrics_best', {}).get('eval_macro_f1')}, phishing recall "
                        f"{tr.get('val_metrics_best', {}).get('eval_phishing_recall')}"],
], "Transformer fine-tuning configuration (multilingual MiniLM-L12)", widths=[4.5, 11.0])
if r.tr_distil:
    d = r.tr_distil
    P(f"DistilBERT-multilingual (lr {d.get('lr')}) reached a best validation macro-F1 of "
      f"{d.get('val_metrics_best', {}).get('eval_macro_f1')} in {d.get('train_time_s')} s "
      f"({d.get('epochs_run')} epochs). Its test results are included in Section 12.")
FIG(training_curve_figure(), "Training loss and validation metrics per epoch")

# ================================================================== 11 SETUP
H("11. Experimental Setup")
ev = env_versions()
TABLE(["Item", "Value"], [
    ["Hardware", "Apple M5 laptop, 10 CPU cores, 16 GB RAM; training on the Apple GPU (MPS); all latency measured on CPU"],
    ["OS / Python", f"{ev['platform']} / Python {ev['python']}"],
    ["Libraries", ", ".join(f"{k} {v}" for k, v in ev["packages"].items())],
    ["Random seed", "42 (data split, models, training)"],
    ["Test protocol", "one held-out test split (945 messages) used once for all models; Hinglish subset = 50 "
                      "messages from 14 templates never seen in training"],
    ["Latency protocol", "batch size 1, 4 CPU threads, median over 200 test messages after warm-up"],
    ["Memory protocol", "RSS increase after loading the model and predicting once, in a fresh process"],
], "Experimental setup", widths=[3.5, 12.0], size=9)

# ================================================================== 12 RESULTS
H("12. Results and Comparative Analysis")
order = [k for k in (LR, NB, SVM, TR, DB, "onnx_int8", ONX) if has(k)]
TABLE(["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Phishing recall", "Hinglish macro F1", "Size (MB)",
       "Latency (ms/SMS)"], [fmt_model_row(k) for k in order],
      "Test-set comparison (same 945 held-out messages)", size=8)
lt, tt = r.eff(LR, "median_ms"), r.eff(TR, "median_ms")
P(f"Headline comparison: the transformer's test macro-F1 ({f4(tr_f1)}) is {cmp_word(tr_f1, lr_f1)} than the "
  f"baseline's ({f4(lr_f1)}), and its phishing recall ({f4(tr_pr)}) is {cmp_word(tr_pr, lr_pr)} than the "
  f"baseline's ({f4(lr_pr)}). On the unseen Hinglish templates, the transformer's macro-F1 ({f4(tr_h)}) is "
  f"{cmp_word(tr_h, lr_h)} than the baseline's ({f4(lr_h)}). The baseline is far cheaper: "
  f"{r.eff(LR, 'model_size_mb'):.1f} MB and {lt:.2f} ms per SMS, against {r.eff(TR, 'model_size_mb'):.0f} MB and "
  f"{tt:.1f} ms for the PyTorch transformer. The best macro-F1 overall is obtained by "
  f"{r.test[best_overall]['display_name']} ({f4(r.o(best_overall, 'macro_f1'))}).")
rows = []
for k in (LR, TR):
    for c in ("ham", "spam", "phishing"):
        pcx = r.test[k]["overall"]["per_class"][c]
        rows.append([r.test[k]["display_name"], c, f4(pcx["precision"]), f4(pcx["recall"]), f4(pcx["f1-score"]),
                     int(pcx["support"])])
TABLE(["Model", "Class", "Precision", "Recall", "F1", "Support"], rows, "Per-class results on the test set")
rows = []
for k in order:
    o = r.test[k]["overall"]
    rows.append([r.test[k]["display_name"], f4(o["phishing_false_negative_rate"]), o["phishing_predicted_as_ham"],
                 f4(o["fraud_recall_binary"]), f4(o["ham_false_alarm_rate"]),
                 f4(o.get("roc_auc_macro_ovr", 0)), f4(o["pr_auc"].get("phishing", 0))])
TABLE(["Model", "Phishing FN rate", "Phishing -> ham", "Fraud recall (spam+phish)", "Ham false-alarm rate",
       "ROC-AUC (macro)", "PR-AUC phishing"], rows, "Security-oriented metrics", size=8.5)
P("Interpretation: a phishing message predicted as spam is still flagged as unwanted, while a phishing message "
  "predicted as ham reaches the user unflagged. That is why the table reports 'phishing -> ham' separately "
  "and the binary fraud recall (spam or phishing detected as non-ham).")
if r.thr:
    rows = []
    for k in ("logreg", "onnx_pruned_int8"):
        t = r.thr.get(k)
        if not t:
            continue
        for split, lab in (("test_default", "arg-max (default)"), ("test_tuned", f"P(phishing) >= {t['threshold']}")):
            v = t[split]
            rows.append([t["display_name"], lab, f4(v["macro_f1"]), f4(v["phishing_recall"]),
                         f4(v["phishing_precision"]), v["phishing_predicted_as_ham"], f4(v["ham_false_alarm_rate"])])
    P(f"Security operating point: a lower phishing threshold was chosen on the validation split as the largest "
      f"threshold that reaches {pct(r.thr['target_val_phishing_recall'], 0)} validation phishing recall (searched "
      f"down to 0.05), then applied unchanged to the test split.")
    TABLE(["Model", "Decision rule", "Macro F1", "Phishing recall", "Phishing precision", "Phishing -> ham",
           "Ham false-alarm"], rows, "Effect of a phishing-favouring threshold (threshold tuned on validation only)",
          size=8.5)
    P("Lowering the threshold gives only small recall gains: the transformer is very confident, and most missed "
      "phishing messages are confidently predicted as spam (which is still flagged), not as ham. Only "
      f"{r.o(LR, 'phishing_predicted_as_ham')} (baseline) and {r.o(TR, 'phishing_predicted_as_ham')} (transformer) "
      "of the 83 test phishing messages reach the user labelled as legitimate.")
rows = []
for k in order:
    h, e = r.test[k]["hinglish"], r.test[k]["english"]
    rows.append([r.test[k]["display_name"], f4(e["macro_f1"]), f4(e["per_class"]["phishing"]["recall"]),
                 f4(h["accuracy"]), f4(h["macro_f1"]), f4(h["per_class"]["phishing"]["recall"]),
                 f4(h["per_class"]["spam"]["recall"])])
TABLE(["Model", "English macro F1", "English phishing R", "Hinglish accuracy", "Hinglish macro F1",
       "Hinglish phishing R", "Hinglish spam R"], rows,
      "Separate English and Hinglish evaluation (Hinglish n = 50, unseen templates)", size=8.5)
P("Caution: the Hinglish test subset is small and synthetic, so a single message changes a per-class recall by "
  "several points. These numbers show generalisation to new templates, not real-world Hinglish performance.",
  italic=True)
FIG(CM / "cm_logreg.png", "Confusion matrix: TF-IDF + Logistic Regression (test)")
FIG(CM / "cm_transformer.png", "Confusion matrix: multilingual MiniLM-L12 (test)")
FIG(CM / "cm_transformer_hinglish.png", "Confusion matrix: multilingual MiniLM-L12 (Hinglish test subset)")
FIG(G / "pr_curves_phishing.png", "Precision-recall curves for the phishing class", width=11)
FIG(G / "model_comparison.png", "Quality and efficiency comparison of all evaluated models")

H("12.1 Error analysis", 2)
import pandas as pd  # noqa: E402
err_lr = pd.read_csv(ROOT / "results/metrics/errors_logreg.csv", keep_default_na=False)
err_tr = pd.read_csv(ROOT / "results/metrics/errors_transformer.csv", keep_default_na=False)
both = set(err_lr.text) & set(err_tr.text)
P(f"The baseline misclassifies {len(err_lr)} and the transformer {len(err_tr)} of the 945 test messages; "
  f"{len(both)} errors are shared by both models. Most shared errors are spam/phishing confusions in the "
  f"English data, which reflects the noisy spam/smishing boundary of the source labels.")
ex = err_tr.sort_values("conf", ascending=False).head(6)
TABLE(["Message (masked, truncated)", "True", "Transformer predicted", "Confidence"],
      [[t[:110] + ("..." if len(t) > 110 else ""), a, b, f"{c:.2f}"] for t, a, b, c in
       ex[["text", "label", "pred", "conf"]].itertuples(index=False)],
      "Highest-confidence transformer errors", widths=[10.0, 1.7, 2.3, 1.7], size=8.5)

H("12.2 Demo messages from the project brief", 2)
if r.demo:
    TABLE(["Synthetic demo message", "Brief's expected label", "TF-IDF + LR", "MiniLM (PyTorch)", "MiniLM ONNX int8"],
          [[d["text"], d["expected_by_brief"], f"{d['logreg']['label']} ({d['logreg']['confidence']:.2f})",
            f"{d['transformer']['label']} ({d['transformer']['confidence']:.2f})",
            f"{d['onnx_pruned_int8']['label']} ({d['onnx_pruned_int8']['confidence']:.2f})"] for d in r.demo],
          "Predictions on the five demo messages", widths=[6.0, 2.0, 2.6, 2.6, 2.6], size=8.5)
    gift = next((d for d in r.demo if d["id"] == "spam_english"), None)
    if gift:
        P(f"Four of the five brief examples are classified as expected by both models. The prize message "
          f"('You have won a free gift. Claim now.') is predicted '{gift['logreg']['label']}' by the baseline and "
          f"'{gift['transformer']['label']}' by the transformer. The baseline's choice follows the training labels "
          f"(Mishra & Soni label prize-claim messages as smishing). The transformer's choice is a genuine miss: the "
          f"message is short and contains no link, number or reward amount, which the transformer associates with "
          f"legitimate messages. This illustrates why the demo shows rule-based indicators next to the model "
          f"scores ('Prize / lottery / reward claim' is flagged for this message).")

# ================================================================== 13 MOBILE
H("13. Mobile Deployment")
P("The selected transformer was exported to ONNX (opset 17, dynamic batch and sequence length) and optimised in "
  "three steps.")
B([("Dynamic INT8 quantization: ", "MatMul/Gemm weights and the embedding table stored as 8-bit integers; "
                                   "activations are quantized at run time. No calibration data needed."),
   ("Vocabulary pruning: ", f"keep the {vp.get('keep_top', 20000):,} most frequent SentencePiece pieces plus all "
                            f"{vp.get('train_pieces', 0):,} pieces seen in the training split (never val/test): "
                            f"{vp.get('kept_vocab', 0):,} of {vp.get('original_vocab', 0):,} rows. A lookup table "
                            f"inside the ONNX graph maps original token IDs to the pruned table, so the standard "
                            f"tokenizer is unchanged. Test-set token OOV rate after pruning: "
                            f"{vp.get('test_token_oov_rate_pct')}% (Hinglish: {vp.get('test_hinglish_token_oov_rate_pct')}%)."),
   ("Embedded tokenizer: ", "the SentencePiece tokenizer is converted into ONNX operators (onnxruntime-extensions) "
                            "and merged with the classifier into one file that takes a string and returns logits.")])
rows = [["PyTorch fp32 (safetensors)", f"{pt_mb:.1f}" if pt_mb else "", "", f"{ox.get('pytorch_test_accuracy', '')}"]]
for k, label in (("model_fp32", "ONNX fp32"), ("model_int8", "ONNX INT8"), ("model_pruned_fp32", "ONNX pruned fp32"),
                 ("model_pruned_int8", "ONNX pruned INT8 (deployed)")):
    v = ox.get("verification", {}).get(k, {})
    rows.append([label, f"{sizes.get(k, 0):.1f}", f"{v.get('label_agreement_with_pytorch_pct', '')}%",
                 f"{v.get('test_accuracy', '')}"])
if e2e:
    rows.append(["End-to-end (tokenizer inside) sms_classifier_e2e.onnx", f"{e2e.get('size_mb', 0):.1f}",
                 f"{e2e.get('label_agreement_with_python_onnx_pct')}% vs pruned INT8", f"{e2e.get('test_accuracy')}"])
TABLE(["Variant", "Size (MB)", "Label agreement with PyTorch (test)", "Test accuracy"], rows,
      "Model size before and after optimisation, with verification on all 945 test messages", widths=[6.5, 2.0, 4.5, 2.5])
if pr_mb and pt_mb:
    P(f"The deployed model is {pt_mb / pr_mb:.1f}x smaller than the PyTorch checkpoint. Its test macro-F1 is "
      f"{f4(onx_f1)}, against {f4(tr_f1)} for the original, and its median CPU latency is "
      f"{r.eff(ONX, 'median_ms'):.1f} ms per SMS with {r.eff(ONX, 'rss_increase_mb')} MB of additional RAM.")
if r.abl:
    TABLE(["Top-K pieces kept", "Vocabulary rows", "INT8 size (MB)", "Test token OOV %", "Hinglish OOV %",
           "Agreement with PyTorch", "Test macro F1", "Hinglish macro F1"],
          [[f"{a['keep_top']:,}", f"{a['kept_vocab']:,}", a["size_mb"], a["test_oov_pct"], a["hinglish_oov_pct"],
            f"{a['agreement_with_pytorch_pct']}%", f4(a["macro_f1"]), f4(a["hinglish_macro_f1"])] for a in r.abl],
          "Vocabulary pruning ablation (training-split pieces always kept)", size=8.5)
    FIG(G / "pruning_ablation.png", "Model size vs. macro-F1 for different vocabulary sizes", width=11)
    P("Overall macro-F1 is almost flat across a 2.3x range of model sizes, so K = 20,000 was kept as a balance "
      "between size and out-of-vocabulary rate. Hinglish macro-F1 moves by one or two messages between settings "
      "(n = 50) and should not be over-interpreted.")
if e2e.get("latency_cpu"):
    lc = e2e["latency_cpu"]
    P(f"The end-to-end model (tokenizer inside) needs about {e2e.get('session_rss_increase_mb')} MB of additional RAM "
      f"for its ONNX Runtime session and takes {lc['1_threads']['median_ms']} ms (1 thread), "
      f"{lc['2_threads']['median_ms']} ms (2 threads) and {lc['4_threads']['median_ms']} ms (4 threads) per SMS "
      f"(median over 200 test messages, desktop CPU). Phone CPUs are slower, so a several-fold higher latency is "
      f"expected on a mid-range device, still well within interactive limits; this was not measured on a phone.")
P("Implementation note: dynamic INT8 quantization computes activation scales over the whole input tensor, so "
  "padding a message inside a batch slightly changes its result. All ONNX numbers above are therefore measured "
  "one message at a time without padding, exactly as on a phone. The Python ONNX path uses the 5 MB SentencePiece "
  "model (XLM-R id convention), which reproduces the training tokenizer exactly on all 5,364 train and test "
  "messages and avoids the ~350 MB RAM cost of loading the 250k-entry tokenizer.json.")
P("Android integration: deployment/android/SmsPhishingClassifier.kt loads sms_classifier_e2e.onnx with ONNX Runtime "
  "Mobile (com.microsoft.onnxruntime:onnxruntime-android) and registers the extensions custom-op library "
  "(onnxruntime-extensions-android) for the tokenizer. It contains a Kotlin port of the masking rules, which a "
  "unit test checks against the Python implementation. Verified: the ONNX files run in ONNX Runtime (desktop, CPU) "
  "with the same extensions operators and produce the predictions reported above. Not verified: the Kotlin code "
  "has not been compiled or run on a physical Android device in this project.", italic=False)

# ================================================================== 14 DEMO
H("14. Working Demo")
P("app/app.py (Gradio) lets the user enter an SMS, choose or auto-detect the language, choose a model or compare "
  "both, and click 'Analyze SMS'. It shows the predicted class with a coloured warning, confidence scores for "
  "both models, rule-based suspicious indicators (links, OTP/PIN/KYC requests, urgency, threats, prizes, fees, "
  "impersonation) and the n-grams that pushed the baseline towards its decision. Links are never opened: "
  "evidence is shown 'defanged' (example[.]com). app/api.py provides the same prediction as a REST endpoint "
  "(FastAPI). The screenshots below are real captures of the running app.")
for key, cap in (("phishing_hinglish", "Demo: Hinglish KYC phishing message"),
                 ("ham_hinglish", "Demo: legitimate Hinglish message"),
                 ("phishing_english", "Demo: English account-blocking message")):
    FIG(SHOTS / f"demo_{key}.png", cap, width=15.5)

# ================================================================== 15-16
H("15. Conclusion")
P(f"The project delivers a complete, reproducible pipeline for three-class SMS classification in English and "
  f"Hinglish. A well-tuned TF-IDF + Logistic Regression baseline is a very strong, tiny and fast model (macro-F1 "
  f"{f4(lr_f1)}, {r.eff(LR, 'model_size_mb'):.1f} MB). The fine-tuned multilingual MiniLM-L12 reaches a macro-F1 "
  f"of {f4(tr_f1)} and phishing recall of {f4(tr_pr)}"
  + (f"; after ONNX export, INT8 quantization and vocabulary pruning it fits in {pr_mb:.1f} MB with test "
     f"macro-F1 {f4(onx_f1)}" if pr_mb else "")
  + ". Answering the research questions: (RQ1) "
  + ("the transformer improves on the baseline in macro-F1" if tr_f1 > lr_f1 else
     "the transformer does not beat the tuned baseline in macro-F1")
  + (" and in phishing recall" if tr_pr > lr_pr else (" but not in phishing recall" if tr_f1 > lr_f1 else
                                                       " nor in phishing recall"))
  + f"; (RQ2) the transformer can be compressed about {pt_mb / pr_mb:.0f}x with "
    f"{ox.get('verification', {}).get('model_pruned_int8', {}).get('label_agreement_with_pytorch_pct', '')}% "
    f"prediction agreement; (RQ3) on unseen Hinglish templates, macro-F1 is {f4(lr_h)} for the baseline and "
    f"{f4(tr_h)} for the transformer, measured on a small synthetic subset." if pr_mb and pt_mb else ".")
P(f"The two models fail differently. The baseline catches more phishing (recall {f4(lr_pr)}) but raises more false "
  f"alarms on legitimate messages ({pct(r.o(LR, 'ham_false_alarm_rate'), 2)} of ham), while the transformer is more "
  f"precise (phishing precision {f4(r.pc(TR, 'phishing', 'precision'))} vs {f4(r.pc(LR, 'phishing', 'precision'))}, "
  f"ham false alarms {pct(r.o(TR, 'ham_false_alarm_rate'), 2)}) and generalises better to the unseen Hinglish "
  f"templates. Most remaining errors of both models are spam/phishing confusions that follow the noisy source labels; "
  f"very few phishing messages are passed as ham. A practical system could therefore combine them: run the 1 MB "
  f"baseline everywhere, and use the 32 MB transformer as a second opinion, flagging a message when either model "
  f"considers it fraudulent. The demo already shows both predictions side by side for this reason.")
H("16. Limitations and Future Scope")
B(["The Hinglish data is synthetic (template-based). Real, consented Hinglish SMS are needed to estimate "
   "real-world performance; the Hinglish test subset has only 50 messages.",
   "The English phishing class follows Mishra & Soni's broad smishing definition, and the spam/smishing "
   "boundary is noisy, which caps achievable precision and recall between those two classes.",
   "The UCI data is old (2011-2012); modern scams (UPI, KYC, crypto, deep links) are under-represented in "
   "public English data.",
   "Personal names in public messages are not masked (no reliable name recogniser for SMS slang).",
   "The Android Kotlin integration was not compiled or tested on a device; only the ONNX models were verified.",
   "Future work: collect consented real Hinglish SMS; add URL reputation and sender-ID features; re-annotate the "
   "spam/smishing boundary with a written guideline; evaluate a model ensemble; try Hinglish-pre-trained compact models; distil the transformer into "
   "a smaller student; test on mid-range Android phones; add Devanagari and other Indian languages."])

# ================================================================== REFERENCES
H("17. References")
REFS = [
    'T. A. Almeida, J. M. G. Hidalgo and A. Yamakami, "Contributions to the study of SMS spam filtering: New collection and results," in Proc. 11th ACM Symp. Document Engineering (DocEng), 2011, pp. 259-262.',
    'T. A. Almeida, J. M. G. Hidalgo and T. P. Silva, "Towards SMS spam filtering: Results under a new dataset," Int. J. Information Security Science, vol. 2, no. 1, pp. 1-18, 2013.',
    'G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," Information Processing & Management, vol. 24, no. 5, pp. 513-523, 1988.',
    'S. Mishra and D. Soni, "Smishing Detector: A security model to detect smishing through SMS content analysis and URL behavior analysis," Future Generation Computer Systems, vol. 108, pp. 803-815, 2020.',
    'S. Mishra and D. Soni, "SMS Phishing Dataset for Machine Learning and Pattern Recognition," Mendeley Data, V1, 2022, doi: 10.17632/f45bkkt8pr.1 (CC BY 4.0).',
    'A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems (NeurIPS), 2017.',
    'J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. NAACL-HLT, 2019, pp. 4171-4186.',
    'V. Sanh, L. Debut, J. Chaumond and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.',
    'W. Wang, F. Wei, L. Dong, H. Bao, N. Yang and M. Zhou, "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," in Advances in Neural Information Processing Systems (NeurIPS), 2020.',
    'A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," in Proc. 58th Annual Meeting of the ACL, 2020, pp. 8440-8451.',
    'T. Kudo and J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing," in Proc. EMNLP: System Demonstrations, 2018, pp. 66-71.',
    'N. Reimers and I. Gurevych, "Making monolingual sentence embeddings multilingual using knowledge distillation," in Proc. EMNLP, 2020, pp. 4512-4525.',
    'K. Bali, J. Sharma, M. Choudhury and Y. Vyas, "\'I am borrowing ya mixing?\' An analysis of English-Hindi code mixing in Facebook," in Proc. First Workshop on Computational Approaches to Code Switching, 2014, pp. 116-126.',
    'R. Nayak and R. Joshi, "L3Cube-HingCorpus and HingBERT: A code mixed Hindi-English dataset and BERT language models," in Proc. WILDRE-6 Workshop @ LREC, 2022, pp. 7-12.',
    'B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in Proc. IEEE/CVF CVPR, 2018, pp. 2704-2713.',
    'ONNX Runtime developers, "ONNX Runtime," 2021. [Online]. Available: https://onnxruntime.ai',
    'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Machine Learning Research, vol. 12, pp. 2825-2830, 2011.',
    'T. Wolf et al., "Transformers: State-of-the-art natural language processing," in Proc. EMNLP: System Demonstrations, 2020, pp. 38-45.',
    'A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems (NeurIPS), 2019.',
    'D. Eastlake and A. Panitz, "Reserved Top Level DNS Names," RFC 2606, IETF, 1999.',
    'Hugging Face model card: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0). [Online]. Available: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'UCI Machine Learning Repository, "SMS Spam Collection" (CC BY 4.0). [Online]. Available: https://archive.ics.uci.edu/dataset/228/sms+spam+collection',
]
for i, ref in enumerate(REFS, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.9)
    p.add_run(f"[{i}]  ").bold = True
    p.add_run(ref).font.size = Pt(10.5)

# ================================================================== APPENDICES
page_break()
H("Appendix A. Team Contributions")
TABLE(["Module", "Niraj Bankar (EN25552002)", "Zaheer Mulani (EN25552018)"], [
    ["Dataset acquisition, licences, label mapping", "Lead", "Support"],
    ["Synthetic Hinglish dataset design", "Lead", "Support"],
    ["Preprocessing, PII masking, EDA", "Lead", "Support"],
    ["TF-IDF baselines and grid search", "Lead", "Support"],
    ["Evaluation framework", "Lead", "Support"],
    ["Transformer survey and fine-tuning", "Support", "Lead"],
    ["ONNX export, quantization, vocabulary pruning", "Support", "Lead"],
    ["Android package, mobile inference", "Support", "Lead"],
    ["Gradio demo and API", "Support", "Lead"],
    ["Explainability, tests, Colab notebook", "Joint", "Joint"],
    ["Report and presentation", "Joint", "Joint"],
], "Contribution matrix (see docs/CONTRIBUTIONS.md)", widths=[7.0, 4.2, 4.2])
H("Appendix B. How to Reproduce")
B(["pip install -r requirements.txt   (Python 3.10-3.12)",
   "bash run_all.sh   (downloads data, trains, evaluates, exports, builds this report)",
   "python app/app.py   (demo at http://127.0.0.1:7860)",
   "Google Colab: notebooks/SMS_Phishing_Colab.ipynb (T4 GPU) runs everything and prints a public demo link."])

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("saved", OUT)
