"""Start the Gradio demo, type the demo messages, click "Analyze SMS" and save real screenshots.

Also records the models' predictions on the demo messages (results/metrics/demo_predictions.json).
Requires: pip install playwright  (uses the locally installed Google Chrome: channel="chrome";
or run `playwright install chromium` and set CHANNEL=None).

Run:  python tools/capture_screenshots.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
PORT = 7861
CHANNEL = "chrome"

DEMO = [
    ("ham_english", "Your appointment is confirmed for tomorrow at 10 AM."),
    ("spam_english", "Congratulations! You have won a free gift. Claim now."),
    ("phishing_english", "Your bank account will be blocked today. Update KYC immediately using this link."),
    ("phishing_hinglish", "Aapka SBI account band ho jayega. KYC update karne ke liye abhi link open karo."),
    ("ham_hinglish", "Kal college lecture 10 baje hai, time pe aa jana."),
]


def record_predictions():
    from src.predict import load_predictor
    from src.preprocessing import is_hinglish
    models = {"logreg": load_predictor("logreg"), "transformer": load_predictor("transformer"),
              "onnx_pruned_int8": load_predictor("onnx_pruned_int8")}
    expected = {"ham_english": "ham", "spam_english": "spam", "phishing_english": "phishing",
                "phishing_hinglish": "phishing", "ham_hinglish": "ham"}
    rows = []
    for key, text in DEMO:
        row = {"id": key, "text": text, "expected_by_brief": expected[key],
               "language_detected": "hinglish" if is_hinglish(text) else "english"}
        for name, m in models.items():
            r = m.predict([text])[0]
            row[name] = {"label": r["label"], "confidence": round(r["confidence"], 4)}
        rows.append(row)
    (ROOT / "results" / "metrics" / "demo_predictions.json").write_text(json.dumps(rows, indent=2))
    return rows


def wait_up(url, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1)
    return False


def capture():
    from playwright.sync_api import sync_playwright
    proc = subprocess.Popen([sys.executable, "app/app.py", "--port", str(PORT)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{PORT}"
        if not wait_up(url):
            raise RuntimeError("app did not start")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=CHANNEL, headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1.5)
            page.goto(url)
            page.wait_for_selector("textarea")
            page.screenshot(path=str(OUT / "00_home.png"), full_page=True)
            for key, text in DEMO:
                before = page.inner_html("#verdict")
                box = page.locator("textarea").first
                box.fill(text)
                page.get_by_role("button", name="Analyze SMS").click()
                page.wait_for_function("b => document.querySelector('#verdict').innerHTML !== b && "
                                       "document.querySelector('#verdict').innerText.includes('confidence')",
                                       arg=before, timeout=60000)
                page.wait_for_timeout(1500)
                page.screenshot(path=str(OUT / f"demo_{key}.png"), full_page=True)
                print("saved", key)
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=20)


if __name__ == "__main__":
    for r in record_predictions():
        print(r["id"], {k: r[k]["label"] for k in ("logreg", "transformer", "onnx_pruned_int8")})
    capture()
