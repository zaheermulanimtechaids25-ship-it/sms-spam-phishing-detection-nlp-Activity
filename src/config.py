"""Central configuration: paths, labels, seeds and dataset sources."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HINGLISH_DIR = DATA_DIR / "hinglish"
MODELS_DIR = ROOT / "models"
CLASSICAL_DIR = MODELS_DIR / "classical"
TRANSFORMER_DIR = MODELS_DIR / "transformer"
OPTIMIZED_DIR = MODELS_DIR / "optimized"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CM_DIR = RESULTS_DIR / "confusion_matrices"
GRAPHS_DIR = RESULTS_DIR / "graphs"

SEED = 42

# Three-class scheme used throughout the project
LABELS = ["ham", "spam", "phishing"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

DATASETS = {
    "uci": {
        "name": "UCI SMS Spam Collection",
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "page": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "licence": "CC BY 4.0",
        "citation": "T. A. Almeida, J. M. G. Hidalgo and A. Yamakami, \"Contributions to the study of SMS spam "
                    "filtering: new collection and results,\" in Proc. ACM DocEng, 2011, pp. 259-262.",
    },
    "mishra_soni": {
        "name": "SMS Phishing Dataset for Machine Learning and Pattern Recognition (Mishra & Soni)",
        "url": "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
               "edb361de-918d-469f-9106-e84823830665/file_downloaded",
        "page": "https://data.mendeley.com/datasets/f45bkkt8pr/1",
        "sha256": "9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3",
        "licence": "CC BY 4.0",
        "citation": "S. Mishra and D. Soni, \"SMS Phishing Dataset for Machine Learning and Pattern Recognition,\" "
                    "Mendeley Data, V1, 2022, doi:10.17632/f45bkkt8pr.1.",
    },
}

for _d in (RAW_DIR, PROCESSED_DIR, HINGLISH_DIR, CLASSICAL_DIR, TRANSFORMER_DIR, OPTIMIZED_DIR,
           METRICS_DIR, CM_DIR, GRAPHS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
