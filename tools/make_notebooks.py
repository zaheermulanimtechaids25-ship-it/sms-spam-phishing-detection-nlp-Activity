"""Generate the project notebooks (run from the repository root: python tools/make_notebooks.py).

notebooks/01_eda.ipynb, 02_classical_ml.ipynb, 03_transformer_training.ipynb are executed
locally (jupyter nbconvert --execute) so that every output shown in them is real.
notebooks/SMS_Phishing_Colab.ipynb is the one-click Google Colab pipeline.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = """import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from IPython.display import Image, display, Markdown
pd.set_option('display.max_colwidth', 120)"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


eda = nb([
    ("md", "# 01 Exploratory Data Analysis\nSMS Spam & Phishing Detection (English + Hinglish). "
           "Niraj Bankar (EN25552002), Zaheer Mulani (EN25552018).\n\nThis notebook builds the dataset "
           "(download, PII masking, label mapping, de-duplication, leakage-safe splits) and reproduces the EDA figures."),
    ("code", SETUP),
    ("md", "## 1. Build the dataset\nDownloads the UCI SMS Spam Collection and the Mishra & Soni SMS Phishing "
           "Dataset (both CC BY 4.0), generates the synthetic Hinglish supplement, masks personal data and splits."),
    ("code", "from src.data_loader import build_dataset\nstats = build_dataset()\n"
             "print(json.dumps({k: stats[k] for k in ['uci_raw','uci_raw_counts','ms_raw','ms_raw_counts',"
             "'english_after_dedup','label_conflicts_dropped','english_counts','hinglish_synthetic','hinglish_counts',"
             "'split_sizes','leakage_text_overlap','leakage_template_overlap']}, indent=1))"),
    ("md", "## 2. Inspect processed data\nPhone numbers, URLs, e-mails and long digit runs are already replaced by "
           "`<phone>`, `<url>`, `<email>`, `<num>`."),
    ("code", "train = pd.read_csv('data/processed/train.csv', keep_default_na=False)\n"
             "print(train.shape); print(train.isna().sum().sum(), 'missing values')\n"
             "train.groupby(['language','label']).size().unstack()"),
    ("code", "train.sample(8, random_state=3)[['text','label','language','source']]"),
    ("md", "## 3. Preprocessing examples"),
    ("code", "from src.preprocessing import mask_pii, clean_text, is_hinglish\n"
             "for s in ['URGENT!!! Your SBI a/c will be BLOCKED. Update KYC at http://sbi-kyc.example/verify or call +91 5555501234',\n"
             "          'Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.',\n"
             "          'Kal college lecture 10 baje hai, time pe aa jana.']:\n"
             "    print('hinglish' if is_hinglish(s) else 'english', '|', clean_text(s))"),
    ("md", "## 4. Figures"),
    ("code", "from src import eda\neda.main()\nfor f in sorted(Path('results/graphs').glob('0*.png')):\n"
             "    display(Markdown(f'**{f.name}**')); display(Image(str(f), width=900))"),
])

classical = nb([
    ("md", "# 02 Classical ML baseline: TF-IDF + Logistic Regression\nWord (1-2) and character (2-5, char_wb) "
           "TF-IDF n-grams, class-balanced Logistic Regression; Linear SVM and Multinomial NB as extra experiments. "
           "Hyper-parameters are tuned by 5-fold stratified CV on the training split only."),
    ("code", SETUP),
    ("code", "from src import train_classical\ntrain_classical.main()"),
    ("md", "## Test-set evaluation (same held-out test split as the transformer)"),
    ("code", "res = json.load(open('results/metrics/test_metrics.json'))\n"
             "from sklearn.metrics import classification_report\n"
             "for m in ['logreg','naive_bayes','linear_svm']:\n"
             "    o = res[m]['overall']\n"
             "    print(f\"{res[m]['display_name']:32s} acc={o['accuracy']:.4f} macroF1={o['macro_f1']:.4f} "
             "phishing recall={o['per_class']['phishing']['recall']:.3f} Hinglish macroF1={res[m]['hinglish']['macro_f1']:.3f}\")"),
    ("code", "display(Image('results/confusion_matrices/cm_logreg.png', width=800))"),
    ("md", "## Most informative n-grams per class"),
    ("code", "import joblib, numpy as np\npipe = joblib.load('models/classical/logreg.joblib')\n"
             "names = pipe.named_steps['features'].get_feature_names_out(); clf = pipe.named_steps['clf']\n"
             "for i, c in enumerate(clf.classes_):\n"
             "    top = np.argsort(clf.coef_[i])[::-1][:15]\n"
             "    print(c.upper(), ':', ', '.join(names[j].split('__',1)[1] for j in top))"),
    ("md", "## Predictions on the demo messages (synthetic examples)"),
    ("code", "from src.predict import load_predictor\nfrom src.explainability import logreg_top_features\n"
             "p = load_predictor('logreg')\n"
             "msgs = ['Your appointment is confirmed for tomorrow at 10 AM.', 'Congratulations! You have won a free gift. Claim now.',\n"
             "        'Your bank account will be blocked today. Update KYC immediately using this link.',\n"
             "        'Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.',\n"
             "        'Kal college lecture 10 baje hai, time pe aa jana.']\n"
             "for m, r in zip(msgs, p.predict(msgs)):\n"
             "    print(f\"[{r['label']:8s} {r['confidence']:.2f}] {m}\\n     top n-grams: {logreg_top_features(p.pipe, m, r['label'], 4)}\")"),
])

transformer = nb([
    ("md", "# 03 Lightweight transformer: multilingual MiniLM-L12\nModel selection survey, fine-tuning, "
           "evaluation and ONNX/INT8/vocabulary-pruned export for mobile.\n\n"
           "Fine-tuning was run with `python -m src.train_transformer` (see `logs/`). Set `RUN_TRAINING = True` "
           "to re-train inside the notebook (about 5 minutes on a GPU / Apple MPS, longer on CPU)."),
    ("code", SETUP + "\nRUN_TRAINING = False"),
    ("md", "## 1. Candidate survey (size, tokenisation of Hinglish, CPU latency)"),
    ("code", "sv = json.load(open('results/metrics/model_selection_survey.json'))\n"
             "pd.DataFrame({k: {'params (M)': v['params_millions'], 'embedding share %': v['embedding_share_pct'],\n"
             "  'fp32 MB': v['fp32_size_mb'], 'CPU ms (batch 1)': v['cpu_latency_ms_batch1'],\n"
             "  'Hinglish pieces/word': v['tokenisation_hinglish']['pieces_per_word'],\n"
             "  'Hinglish p99 tokens': v['tokenisation_hinglish']['p99_tokens']} for k, v in sv.items()}).T"),
    ("code", "for k, v in sv.items(): print(f'{k:40s}', ' '.join(v['sample_tokens']))"),
    ("md", "## 2. Fine-tuning"),
    ("code", "if RUN_TRAINING:\n    from src.train_transformer import train\n"
             "    train('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', lr=8e-5)\n"
             "for name in ['paraphrase-multilingual-MiniLM-L12-v2', 'distilbert-base-multilingual-cased']:\n"
             "    f = Path(f'results/metrics/transformer_training_{name}.json')\n"
             "    if f.exists():\n        info = json.load(open(f))\n"
             "        print(name, '| device', info['device'], '| epochs', info['epochs_run'], '| train time', info['train_time_s'], 's')\n"
             "        print('   best val:', info['val_metrics_best'])"),
    ("code", "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(1, 2, figsize=(11, 3.5))\n"
             "for name in ['paraphrase-multilingual-MiniLM-L12-v2', 'distilbert-base-multilingual-cased']:\n"
             "    f = Path(f'results/metrics/transformer_training_{name}.json')\n"
             "    if not f.exists(): continue\n    h = json.load(open(f))['history']\n"
             "    ev = [x for x in h if 'eval_macro_f1' in x]; tr = [x for x in h if 'loss' in x and 'eval_loss' not in x]\n"
             "    ax[0].plot([x['epoch'] for x in tr], [x['loss'] for x in tr], 'o-', label=name[:22]+' train')\n"
             "    ax[0].plot([x['epoch'] for x in ev], [x['eval_loss'] for x in ev], 's--', label=name[:22]+' val')\n"
             "    ax[1].plot([x['epoch'] for x in ev], [x['eval_macro_f1'] for x in ev], 'o-', label=name[:22])\n"
             "ax[0].set_title('Loss'); ax[1].set_title('Validation macro-F1'); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)\n"
             "for a in ax: a.set_xlabel('epoch')\nplt.tight_layout(); plt.show()"),
    ("md", "## 3. Test-set comparison (all models, same test split)"),
    ("code", "print(open('results/metrics/comparison_table.md').read())"),
    ("code", "display(Image('results/graphs/model_comparison.png', width=950)); "
             "display(Image('results/graphs/pr_curves_phishing.png', width=500))"),
    ("code", "display(Image('results/confusion_matrices/cm_transformer.png', width=800)); "
             "display(Image('results/confusion_matrices/cm_transformer_hinglish.png', width=800))"),
    ("md", "## 4. Mobile optimisation (ONNX, INT8, vocabulary pruning) and verification"),
    ("code", "rep = json.load(open('results/metrics/onnx_export_report.json'))\n"
             "print(json.dumps({k: rep[k] for k in ['pytorch_fp32_weights_mb','onnx_sizes_mb','vocab_pruning','verification']}, indent=1))\n"
             "print(json.dumps(rep.get('end_to_end_model', {}), indent=1))"),
    ("code", "import subprocess\nprint(subprocess.run([sys.executable, 'deployment/mobile_inference.py',\n"
             "  'Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.',\n"
             "  'Kal college lecture 10 baje hai, time pe aa jana.', '--benchmark'], capture_output=True, text=True).stdout)"),
])

COLAB_REPO = "https://github.com/<your-username>/sms-spam-phishing-detection.git"
colab = nb([
    ("md", "# SMS Spam & Phishing Detection: full pipeline on Google Colab\n"
           "Niraj Bankar (EN25552002) & Zaheer Mulani (EN25552018), MTech NLP mini-project.\n\n"
           "**Before you start:** Runtime -> Change runtime type -> **T4 GPU**. Then run all cells.\n\n"
           "The notebook clones the repository, downloads the public datasets, trains both models, evaluates them, "
           "exports the mobile ONNX model, rebuilds the report and slides from *your* results, and starts the Gradio "
           "demo with a public link. Results are saved to Google Drive so a disconnect does not lose them.\n\n"
           "> GPU training is not bit-for-bit reproducible, so your numbers can differ slightly from the ones in the "
           "submitted report. Run the last cells to regenerate the report/slides from this run."),
    ("md", "## 1. Get the code\nReplace the URL with your GitHub repository (or upload the project ZIP to Colab "
           "and unzip it instead)."),
    ("code", f"REPO_URL = '{COLAB_REPO}'\n"
             "import os\nif not os.path.exists('sms-spam-phishing-detection'):\n"
             "    if os.path.exists('/content/SMS_Spam_Phishing_Detection_NLP_Project.zip'):\n"
             "        !unzip -q /content/SMS_Spam_Phishing_Detection_NLP_Project.zip -d /content\n"
             "    else:\n        !git clone $REPO_URL\n%cd /content/sms-spam-phishing-detection"),
    ("code", "!pip -q install -r requirements.txt onnxruntime-extensions\nimport torch; print('GPU:', torch.cuda.is_available())"),
    ("md", "## 2. Optional: save outputs to Google Drive"),
    ("code", "SAVE_TO_DRIVE = True\nif SAVE_TO_DRIVE:\n    from google.colab import drive; drive.mount('/content/drive')\n"
             "    DRIVE_DIR = '/content/drive/MyDrive/sms-phishing-results'; os.makedirs(DRIVE_DIR, exist_ok=True)"),
    ("md", "## 3. Data: download, mask personal data, label mapping, split"),
    ("code", "!python -m src.data_loader | tail -40\n!python -m src.eda"),
    ("md", "## 4. Classical baseline (TF-IDF + Logistic Regression, + SVM / NB)"),
    ("code", "!python -m src.train_classical"),
    ("md", "## 5. Transformer: model survey and fine-tuning (about 5 minutes on a T4)"),
    ("code", "!python -m src.model_selection\n"
             "!python -m src.train_transformer --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --lr 8e-5\n"
             "# optional comparison model:\n"
             "# !python -m src.train_transformer --model distilbert-base-multilingual-cased --lr 3e-5"),
    ("md", "## 6. Mobile export: ONNX, INT8, vocabulary pruning, end-to-end model"),
    ("code", "!python -m deployment.export_onnx\n!python -m deployment.build_e2e_model"),
    ("md", "## 7. Evaluation on the held-out test set"),
    ("code", "!python -m src.evaluate --reset --models logreg naive_bayes linear_svm transformer onnx_int8 onnx_pruned_int8\n"
             "print(open('results/metrics/comparison_table.md').read())"),
    ("code", "!python -m pytest -q tests"),
    ("md", "## 8. Rebuild report and presentation from this run's results"),
    ("code", "!python docs/build_diagrams.py && python docs/build_report.py && python docs/build_presentation.py\n"
             "if SAVE_TO_DRIVE:\n    !cp -r results docs/*.docx docs/*.pptx models/optimized $DRIVE_DIR/"),
    ("md", "## 9. Demo (public link)\nOpens a Gradio app. The `*.gradio.live` link works on a phone too; it "
           "expires after 72 hours or when the Colab session stops."),
    ("code", "!python app/app.py --share"),
])

for name, n in [("01_eda.ipynb", eda), ("02_classical_ml.ipynb", classical),
                ("03_transformer_training.ipynb", transformer), ("SMS_Phishing_Colab.ipynb", colab)]:
    nbf.write(n, NB / name)
    print("wrote", NB / name)
