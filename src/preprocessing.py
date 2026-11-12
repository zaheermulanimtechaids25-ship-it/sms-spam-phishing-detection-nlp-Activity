"""Text preprocessing for English and Romanized Hindi-English (Hinglish) SMS.

Two levels of processing are used:

* ``mask_pii``  - applied to every message before it is stored in ``data/processed``.
  It removes sensitive information (URLs, e-mail addresses, phone numbers, long digit
  sequences such as account numbers or OTPs) and replaces them with placeholder tokens.
* ``clean_text`` - model-ready normalisation for the classical baseline (lower-casing,
  repeated-character squashing, light Hinglish/SMS spelling normalisation).
  The transformer receives ``mask_pii`` output with original casing, because casing
  ("URGENT", "WINNER") is itself a useful signal and the tokenizer is cased.

Stemming and stop-word removal are deliberately NOT applied: words such as "your",
"now", "aap", "karo" or "abhi" are strong phishing indicators and would be lost.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------- placeholders
URL_TOKEN = "<url>"
EMAIL_TOKEN = "<email>"
PHONE_TOKEN = "<phone>"
NUM_TOKEN = "<num>"
PLACEHOLDERS = (URL_TOKEN, EMAIL_TOKEN, PHONE_TOKEN, NUM_TOKEN)

# ---------------------------------------------------------------- regexes
# Common mojibake found in the public datasets (latin-1 / cp1252 round trips)
MOJIBAKE = {
    "ï¿½": "", "�": "", "Â£": "£", "Â": "", "â‚¬": "€", "â€™": "'", "â€˜": "'",
    "â€œ": '"', "â€\x9d": '"', "â€“": "-", "â€”": "-", "â€¦": "...", "&lt;": "<", "&gt;": ">",
    "&amp;": "&", "&#039;": "'",
}
URL_RE = re.compile(
    r"(?:(?:https?://|www\.)\S*)"                                  # explicit scheme / www
    r"|(?:\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|in|co|uk|io|ly|me|info|biz|mobi|xyz|"
    r"top|online|site|click|link|example|test|do|gl|cc|tk|ml|ga|cf|gq|app|live|club|win)"
    r"(?:/\S*)?)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# phone numbers: +91 98xxx xxxxx, 09061743386, 0800 123 4567 ...  (>= 9 digits incl. separators)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s-]{7,}\d)(?!\w)")
# remaining standalone digit runs of 4+ digits (OTPs, account fragments, short codes, PINs)
# 7+ digit runs glued to letters, e.g. "call09050032" or "TsCs0871474032" (premium/phone numbers)
GLUED_NUM_RE = re.compile(r"\d{7,}")
LONGNUM_RE = re.compile(r"(?<![\w<])\d{4,}(?![\w>])")
REPEAT_RE = re.compile(r"(.)\1{2,}")                      # "soooo" -> "soo"
PUNCT_REPEAT_RE = re.compile(r"([!?.])\1+")               # "!!!" -> "!"
ALLOWED_RE = re.compile(r"[^\w\s<>£$€₹%!?.,:'/&+-]")       # keep currency + useful punctuation
SPACE_RE = re.compile(r"\s+")

# ---------------------------------------------------------------- normalisation dictionaries
# Only frequent, unambiguous spelling variants. Romanized Hindi words are kept, NOT translated.
HINGLISH_NORMALISE = {
    "nhi": "nahi", "nahin": "nahi", "nai": "nahi", "nhin": "nahi",
    "kro": "karo", "krna": "karna", "kr": "kar", "krein": "karein", "kare": "karein",
    "bhjo": "bhejo", "bhejiye": "bhejo", "bhejein": "bhejo",
    "ap": "aap", "apka": "aapka", "apke": "aapke", "apki": "aapki",
    "h": "hai", "hy": "hai", "he": "hai",
    "kya": "kya", "kia": "kya", "kyaa": "kya",
    "abi": "abhi", "jaldi": "jaldi", "turant": "turant",
    "paise": "paisa", "pese": "paisa", "rupaye": "rupees", "rs": "rs", "rs.": "rs",
}
SMS_NORMALISE = {
    "u": "you", "ur": "your", "urs": "yours", "r": "are", "pls": "please", "plz": "please",
    "plzz": "please", "txt": "text", "msg": "message", "msgs": "messages", "2day": "today",
    "2morrow": "tomorrow", "tmrw": "tomorrow", "4": "for", "b4": "before", "thx": "thanks",
    "acc": "account", "a/c": "account", "ac": "account", "acct": "account",
}
# "he" is English too; only treat it as Hinglish "hai" inside messages already flagged as Hinglish.
_AMBIGUOUS_HINGLISH = {"h", "he", "hy", "ap", "kr"}

# Function words that are very frequent in Romanized Hindi and rare in English SMS.
HINGLISH_MARKERS = {
    "hai", "hain", "ho", "hoga", "hogi", "hoge", "ka", "ki", "ke", "ko", "se", "mein", "mai", "main",
    "aap", "aapka", "aapke", "aapki", "apka", "tum", "tumhara", "kya", "kyu", "kyun", "nahi", "nhi",
    "karo", "kare", "karein", "karna", "kar", "kro", "bhejo", "abhi", "kal", "aaj", "jaldi", "turant",
    "band", "jayega", "jaega", "jaayega", "liye", "wala", "wali", "raha", "rahi", "rahe", "gaya",
    "gayi", "diya", "mil", "milega", "milegi", "paisa", "paise", "yaar", "bhai", "haan", "accha",
    "acha", "theek", "thik", "batao", "bata", "ghar", "aana", "jana", "aa", "ja", "lo", "do", "dena",
    "lena", "par", "pe", "sab", "kuch", "baje", "kab", "kaha", "kahan", "kyunki", "lekin", "aur",
    "agar", "toh", "to", "bhi", "sirf", "apna", "apne", "humara", "hamara", "beta", "mummy", "papa",
}
_STRONG_MARKERS = HINGLISH_MARKERS - {"to", "do", "par", "pe", "lo", "aa", "ja", "main", "band", "mil", "kar"}


def fix_encoding(text: str) -> str:
    """Repair common mojibake and normalise unicode (NFKC)."""
    for bad, good in MOJIBAKE.items():
        text = text.replace(bad, good)
    return unicodedata.normalize("NFKC", text)


def mask_pii(text: str) -> str:
    """Replace URLs, e-mails, phone numbers and long digit sequences with placeholders.

    This is the privacy step: the processed dataset never contains these values.
    Order matters: e-mails first (so their domain is not read as a URL), then URLs,
    then phone numbers, then remaining long digit runs.
    """
    if not isinstance(text, str):
        return ""
    text = fix_encoding(text).strip()
    text = EMAIL_RE.sub(f" {EMAIL_TOKEN} ", text)
    text = URL_RE.sub(f" {URL_TOKEN} ", text)
    text = PHONE_RE.sub(f" {PHONE_TOKEN} ", text)
    text = GLUED_NUM_RE.sub(f" {PHONE_TOKEN} ", text)
    text = LONGNUM_RE.sub(f" {NUM_TOKEN} ", text)
    return SPACE_RE.sub(" ", text).strip()


def is_hinglish(text: str, threshold: int = 2) -> bool:
    """Heuristic code-mixing detector for Romanized Hindi.

    A message is flagged as Hinglish when it contains at least ``threshold`` distinct
    Romanized-Hindi function words (with at least one "strong" marker that is not also
    a common English word). This is a heuristic for EDA and for the demo's language
    hint, not a validated language-identification model.
    """
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    hits = tokens & HINGLISH_MARKERS
    return len(hits) >= threshold and bool(hits & _STRONG_MARKERS)


def clean_text(text: str, hinglish: bool | None = None) -> str:
    """Normalise a (PII-masked) message for the TF-IDF baseline."""
    text = mask_pii(text)
    if hinglish is None:
        hinglish = is_hinglish(text)
    text = text.lower()
    text = REPEAT_RE.sub(r"\1\1", text)
    text = PUNCT_REPEAT_RE.sub(r"\1", text)
    text = ALLOWED_RE.sub(" ", text)
    out = []
    for tok in text.split():
        if tok in PLACEHOLDERS:
            out.append(tok)
            continue
        core = tok.strip(".,:!?'")
        if hinglish and core in HINGLISH_NORMALISE:
            tok = tok.replace(core, HINGLISH_NORMALISE[core])
        elif not hinglish and core in HINGLISH_NORMALISE and core not in _AMBIGUOUS_HINGLISH:
            tok = tok.replace(core, HINGLISH_NORMALISE[core])
        elif core in SMS_NORMALISE:
            tok = tok.replace(core, SMS_NORMALISE[core])
        out.append(tok)
    return SPACE_RE.sub(" ", " ".join(out)).strip()


def transformer_text(text: str) -> str:
    """Input for the transformer: PII-masked, repeated characters squashed, casing kept."""
    text = mask_pii(text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = PUNCT_REPEAT_RE.sub(r"\1", text)
    return SPACE_RE.sub(" ", text).strip()


if __name__ == "__main__":
    for s in [
        "URGENT!!! Your SBI a/c will be BLOCKED. Update KYC at http://sbi-kyc.example/verify or call +91 5555501234",
        "Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo.",
        "Kal college lecture 10 baje hai, time pe aa jana.",
        "Your OTP is 482913. Do not share it with anyone.",
        "Sooooo happy!!! Call me at 09061743386 or mail test.user@example.com",
    ]:
        print(is_hinglish(s), "|", mask_pii(s), "|", clean_text(s))
