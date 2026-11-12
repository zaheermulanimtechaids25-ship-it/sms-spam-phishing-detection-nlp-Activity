import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_text, is_hinglish, mask_pii, transformer_text  # noqa: E402


def test_masks_url_phone_email_and_numbers():
    s = "Call +91 5555501234 or 09061743386, mail a.b@example.com, visit http://x.example/kyc, OTP 482913"
    m = mask_pii(s)
    assert "<phone>" in m and "<email>" in m and "<url>" in m and "<num>" in m
    for leaked in ("5555501234", "09061743386", "a.b@", "x.example", "482913"):
        assert leaked not in m


def test_bare_domains_are_masked():
    assert "<url>" in mask_pii("go to bit-abc12.example now")
    assert "<url>" in mask_pii("visit www.win-82050.co.uk")


def test_hinglish_words_are_preserved_not_translated():
    out = clean_text("Aapka SBI account band ho jayega. KYC update karo")
    for w in ("aapka", "band", "ho", "jayega", "karo"):
        assert w in out


def test_hinglish_spelling_normalisation():
    assert "nahi" in clean_text("aap ka kaam nhi hua hai bhai")
    assert "karo" in clean_text("abhi payment kro yaar jaldi, aap ka account")


def test_english_he_is_not_rewritten():
    assert " he " in f" {clean_text('he said he will come to the office')} "


def test_repeated_characters_and_punctuation():
    out = clean_text("Sooooo happy!!!!")
    assert "soo happy!" in out


def test_no_stopword_removal_keeps_phishing_cues():
    out = clean_text("Your account will be blocked now, update your details")
    assert "your" in out and "now" in out


@pytest.mark.parametrize("text,expected", [
    ("Kal college lecture 10 baje hai, time pe aa jana.", True),
    ("Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.", True),
    ("Your appointment is confirmed for tomorrow at 10 AM.", False),
    ("Congratulations! You have won a free gift. Claim now.", False),
])
def test_language_detector(text, expected):
    assert is_hinglish(text) is expected


def test_transformer_text_keeps_case():
    assert "URGENT" in transformer_text("URGENT!!! call now")


def test_processed_data_contains_no_raw_phone_numbers_or_urls():
    proc = ROOT / "data" / "processed"
    if not (proc / "train.csv").exists():
        pytest.skip("processed data not built")
    df = pd.concat(pd.read_csv(proc / f"{s}.csv", keep_default_na=False) for s in ("train", "val", "test"))
    assert not df.text.str.contains(r"\d{9,}", regex=True).any()
    assert not df.text.str.contains(r"https?://", regex=True).any()
    assert not df.text.str.contains(r"\.example\b", regex=True).any()


def test_mobile_preprocessing_matches_training_preprocessing():
    sys.path.insert(0, str(ROOT / "deployment"))
    from mobile_inference import preprocess  # noqa: E402
    samples = ["URGENT!!! Your SBI a/c will be BLOCKED. Update KYC at http://sbi.example/verify or call +91 5555501234",
               "Sooo happy, mail me at x@y.com", "Ur OTP is 482913 Â£100 prize ï¿½ claim",
               "Aapka account band ho jayega www.kyc-abc.example"]
    proc = ROOT / "data" / "processed" / "test.csv"
    if proc.exists():
        samples += pd.read_csv(proc, keep_default_na=False).text.tolist()[:300]
    for s in samples:
        assert preprocess(s) == transformer_text(s)
