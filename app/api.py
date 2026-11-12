"""Optional local REST API (FastAPI).

Run:   uvicorn app.api:app --port 8000
Test:  curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
            -d '{"text": "Aapka SBI account band ho jayega", "model": "onnx_pruned_int8"}'
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.explainability import suspicious_indicators
from src.predict import load_predictor
from src.preprocessing import is_hinglish

app = FastAPI(title="SMS Spam & Phishing Detection API", version="1.0")
_CACHE: dict = {}
ALLOWED = {"logreg", "naive_bayes", "linear_svm", "transformer", "onnx_int8", "onnx_pruned_int8"}


class SMSIn(BaseModel):
    text: str
    model: str = "onnx_pruned_int8"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: SMSIn):
    if req.model not in ALLOWED:
        raise HTTPException(400, f"model must be one of {sorted(ALLOWED)}")
    if req.model not in _CACHE:
        _CACHE[req.model] = load_predictor(req.model)
    res = _CACHE[req.model].predict([req.text])[0]
    return {"model": req.model, "language": "hinglish" if is_hinglish(req.text) else "english", **res,
            "indicators": suspicious_indicators(req.text)}
