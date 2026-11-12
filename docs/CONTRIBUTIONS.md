# Team contributions

**Project:** Detecting Spam and Phishing SMS Messages, Including Mixed Hindi-English (Hinglish) Text,
Using Classical Machine Learning and Lightweight Transformers

| Member | Roll No. | Primary ownership |
|---|---|---|
| Niraj Bankar | EN25552002 | Data, preprocessing, classical baseline, evaluation framework |
| Zaheer Mulani | EN25552018 | Transformer, optimisation / ONNX, mobile deployment, demo app |

This is the *planned* division of work. Please adjust it to match who actually does each part,
because the lecturer's rule 8 requires each member to explain their own contribution. Your Git
history should show the same split: each person commits the modules they own.

## Contribution matrix

L = lead (writes and owns the code), S = support (reviews, tests, can explain it), J = joint

| Module / task | Files | Niraj | Zaheer |
|---|---|---|---|
| Dataset acquisition (UCI + Mishra & Soni, checksum, licences) | `src/data_loader.py`, `data/README.md` | L | S |
| Synthetic Hinglish dataset design (90 templates, privacy rules) | `src/hinglish_generator.py` | L | S |
| Label-mapping strategy and leakage-safe splitting | `src/data_loader.py` | L | S |
| Preprocessing and PII masking, Hinglish normalisation, language heuristic | `src/preprocessing.py` | L | S |
| EDA and visualisations | `src/eda.py`, `notebooks/01_eda.ipynb` | L | S |
| TF-IDF features and LR / SVM / NB baselines, grid search | `src/feature_extraction.py`, `src/train_classical.py`, `notebooks/02_*` | L | S |
| Evaluation framework (metrics, confusion matrices, PR curves, latency, memory) | `src/evaluate.py` | L | S |
| Transformer candidate survey and model selection | `src/model_selection.py` | S | L |
| Transformer fine-tuning (weighted loss, early stopping) | `src/train_transformer.py`, `notebooks/03_*` | S | L |
| ONNX export, INT8 quantization, vocabulary pruning, verification | `deployment/export_onnx.py` | S | L |
| End-to-end Android model (embedded tokenizer) and Kotlin integration | `deployment/build_e2e_model.py`, `deployment/android/` | S | L |
| Stand-alone mobile inference script | `deployment/mobile_inference.py` | S | L |
| Explainability (indicators, n-gram contributions) | `src/explainability.py` | J | J |
| Gradio demo and FastAPI | `app/app.py`, `app/api.py` | S | L |
| Unit tests | `tests/` | J | J |
| Colab notebook | `notebooks/SMS_Phishing_Colab.ipynb` | J | J |
| Report | `docs/SMS_Spam_Phishing_Detection_NLP_Project_Report.docx` | J (ch. 6-11) | J (ch. 12-15) |
| Presentation and demo | `docs/*.pptx` | J (slides 1-9) | J (slides 10-16) |

## Shared understanding (both members)

Both members must be able to explain, without notes:

1. The three-class label scheme, and why UCI spam was **not** relabelled as phishing by us.
2. Why the split is stratified for public data and grouped by template for synthetic data (leakage).
3. How TF-IDF works and why character n-grams help with Hinglish spelling variation.
4. How Logistic Regression produces class probabilities, and what `class_weight="balanced"` does.
5. Transformer basics: tokenization (SentencePiece), self-attention, [CLS]-style pooling, fine-tuning.
6. Why multilingual MiniLM was chosen (survey table), and what vocabulary pruning and INT8 do.
7. The headline numbers in the comparison table, and why phishing recall matters most.
8. The limitations: synthetic Hinglish, the broad smishing definition, the small Hinglish test set,
   and the lack of on-device Android testing.

## How each member can present their part (2-3 minutes each)

**Niraj (data and baseline):**
"I built the data pipeline. We combine the UCI collection with the Mishra & Soni smishing dataset,
both CC BY 4.0. Since UCI has no phishing label, we use the smishing labels that Mishra & Soni
assigned and keep the original labels in separate columns. Before saving, every URL, e-mail,
phone number and long digit run is masked; a unit test checks this. I designed the 90 synthetic
Hinglish templates and split them by template so no template leaks into the test set. For the
baseline I combined word and character TF-IDF n-grams with balanced Logistic Regression, tuned by
5-fold CV on the training data only. Then I built the evaluation script, which scores every model
on the same test split, including phishing recall, the Hinglish subset, latency and model size."

**Zaheer (transformer and deployment):**
"I compared four multilingual transformers on size, Hinglish tokenisation and CPU latency, and
chose multilingual MiniLM-L12: the fastest one, with the fewest word pieces per Hinglish word.
I fine-tuned it with class-weighted cross-entropy and early stopping on validation macro-F1. For
mobile, I exported it to ONNX, applied dynamic INT8 quantization, and pruned the 250k-token
vocabulary down to the pieces we need, which shrank the model a lot. I verified that the
optimised model agrees with the PyTorch model on the whole test set. Finally, I embedded the
SentencePiece tokenizer into a single ONNX file for Android, wrote the Kotlin integration, and
built the Gradio demo that compares both models and shows warning indicators without ever
opening links."
