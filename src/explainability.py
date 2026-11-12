"""Human-readable explanations of why an SMS looks suspicious.

Two complementary sources:
1. Rule-based indicators (URLs, OTP/PIN/KYC requests, urgency, threats, prizes ...) for
   English and Romanized Hindi. These are shown to the user as warnings; they do not
   change the model's prediction.
2. For the Logistic Regression baseline, the n-gram features that contributed most to
   the predicted class (TF-IDF value x class coefficient).

Links in a message are only detected, never opened or fetched.
"""
from __future__ import annotations

import re

import numpy as np

from src.preprocessing import EMAIL_RE, PHONE_RE, URL_RE, fix_encoding

INDICATORS = [
    ("Contains a link", "A link in an unexpected SMS is the most common phishing delivery method.",
     None),  # handled separately with URL_RE
    ("Shortened / disguised link", "Short or look-alike links hide the real destination.",
     r"\b(bit\.ly|tinyurl|goo\.gl|t\.co|bit\.do|bit-\w+|smsg\.io|is\.gd|cutt\.ly)\b"),
    ("Asks for OTP / PIN / password / card details",
     "Banks never ask for OTP, PIN, CVV or passwords over SMS.",
     r"\b(otp|pin|cvv|password|passcode|card (number|details|no)|account (number|details|no)|net ?banking|"
     r"upi pin|login details|card ki last|bank details)\b"),
    ("KYC / Aadhaar / PAN update request", "Fake KYC updates are a common Indian SMS scam.",
     r"\b(kyc|aadhaar|aadhar|pan card|pan link|re-?kyc)\b"),
    ("Urgency or deadline", "Scammers create pressure so you act without thinking.",
     r"\b(urgent|immediately|right now|today|within \d+ ?(hours|hrs|mins?)|expire[sd]?|last chance|"
     r"abhi|turant|jaldi|aaj raat|warna|\d+ ghante|final notice|act now)\b"),
    ("Threat of blocking / suspension", "Threats of account blocking are used to scare victims.",
     r"\b(block(ed)?|suspend(ed)?|deactivat\w*|band ho|kat jayega|kat jaega|disconnect\w*|freez\w*|"
     r"terminated?|legal action|arrest)\b"),
    ("Prize / lottery / reward claim", "Unexpected prizes and refunds are bait for fraud.",
     r"\b(won|winner|win|prize|lottery|lucky draw|jackpot|cash ?back|reward|refund|jeet\w*|inaam|"
     r"claim|selected|congratulations|badhai)\b"),
    ("Asks you to pay a fee", "Legitimate prizes, jobs or loans never need an upfront fee.",
     r"\b(processing (fee|charge)|registration fee|security deposit|fee bharo|pay rs|customs charge|"
     r"charge pending)\b"),
    ("Contains a phone number to call/reply", "Premium-rate or fake support numbers are common in scams.",
     None),  # handled with PHONE_RE
    ("Impersonates bank / authority", "Messages pretending to be banks, police or tax office.",
     r"\b(sbi|hdfc|icici|axis|pnb|kotak|bank of baroda|customer care|income tax|cyber cell|police|"
     r"rbi|electricity|bijli)\b"),
]
_COMPILED = [(n, d, re.compile(p, re.I) if p else None) for n, d, p in INDICATORS]


def suspicious_indicators(text: str) -> list[dict]:
    """Return the list of indicators found in the raw message (links are never opened)."""
    text = fix_encoding(text or "")
    found = []
    for name, desc, rx in _COMPILED:
        if name == "Contains a link":
            hits = [m.group(0) for m in URL_RE.finditer(EMAIL_RE.sub(" ", text))]
        elif name == "Contains a phone number to call/reply":
            hits = [m.group(0) for m in PHONE_RE.finditer(text)]
        else:
            hits = [m.group(0) for m in rx.finditer(text)]
        if hits:
            found.append({"indicator": name, "why": desc, "evidence": sorted(set(h.strip() for h in hits))[:4]})
    return found


def logreg_top_features(pipe, text: str, label: str, k: int = 6) -> list[tuple[str, float]]:
    """Top n-gram contributions towards ``label`` for a fitted TF-IDF + LR pipeline."""
    features, clf = pipe.named_steps["features"], pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    x = features.transform([text])
    names = features.get_feature_names_out()
    cls = list(clf.classes_).index(label)
    contrib = x.multiply(clf.coef_[cls]).tocsr()
    idx = contrib.indices[np.argsort(contrib.data)[::-1][:k * 3]]
    out, seen = [], set()
    for i in idx:
        v = float(contrib[0, i])
        if v <= 0:
            continue
        kind, feat = names[i].split("__", 1)
        feat = feat.strip()
        if len(feat) < 3 or feat in seen:
            continue
        seen.add(feat)
        out.append((f"{feat} ({'word' if kind == 'word' else 'chars'})", round(v, 3)))
        if len(out) == k:
            break
    return out


if __name__ == "__main__":
    for s in ["Your bank account will be blocked today. Update KYC immediately using this link.",
              "Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo bit-xy12.example",
              "Kal college lecture 10 baje hai, time pe aa jana."]:
        print(s)
        for f in suspicious_indicators(s):
            print("   -", f["indicator"], f["evidence"])
