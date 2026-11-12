import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASSICAL_DIR, LABELS, OPTIMIZED_DIR, TRANSFORMER_DIR  # noqa: E402
from src.explainability import suspicious_indicators  # noqa: E402
from src.predict import DEFAULT_TRANSFORMER, load_predictor  # noqa: E402

MESSAGES = [
    "Your appointment is confirmed for tomorrow at 10 AM.",
    "Congratulations! You have won a free gift. Claim now.",
    "Your bank account will be blocked today. Update KYC immediately using this link.",
    "Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.",
    "Kal college lecture 10 baje hai, time pe aa jana.",
]

AVAILABLE = [m for m, ok in [
    ("logreg", (CLASSICAL_DIR / "logreg.joblib").exists()),
    ("transformer", (TRANSFORMER_DIR / DEFAULT_TRANSFORMER).exists()),
    ("onnx_pruned_int8", (OPTIMIZED_DIR / "model_pruned_int8.onnx").exists()),
] if ok]


@pytest.fixture(scope="module", params=AVAILABLE or [None])
def predictor(request):
    if request.param is None:
        pytest.skip("no trained model available")
    return load_predictor(request.param)


def test_output_format_and_probabilities(predictor):
    out = predictor.predict(MESSAGES)
    assert len(out) == len(MESSAGES)
    for r in out:
        assert r["label"] in LABELS
        assert abs(sum(r["scores"].values()) - 1) < 1e-4
        assert 0 <= r["confidence"] <= 1


def test_clear_ham_messages(predictor):
    out = predictor.predict([MESSAGES[0], MESSAGES[4]])
    assert all(r["label"] == "ham" for r in out)


def test_hinglish_kyc_phishing_flagged(predictor):
    assert predictor.predict([MESSAGES[3]])[0]["label"] == "phishing"


def test_batch_and_single_agree(predictor):
    batch = predictor.predict_proba(MESSAGES)
    single = np.vstack([predictor.predict_proba([m]) for m in MESSAGES])
    assert np.allclose(batch, single, atol=1e-3)


def test_indicators_detect_without_opening_links():
    inds = {f["indicator"] for f in suspicious_indicators(
        "URGENT: SBI KYC pending, account band ho jayega. Click http://kyc-x.example and share OTP")}
    assert {"Contains a link", "Asks for OTP / PIN / password / card details", "Urgency or deadline"} <= inds
