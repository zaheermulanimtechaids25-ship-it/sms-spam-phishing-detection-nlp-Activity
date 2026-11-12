#!/usr/bin/env bash
# Reproduce the whole project end-to-end (CPU works; GPU/Apple MPS is used automatically if present).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1

$PY -m src.data_loader            # download (UCI + Mishra & Soni), mask PII, map labels, split
$PY -m src.eda                    # figures -> results/graphs
$PY -m src.train_classical        # TF-IDF + LR / SVM / NB
$PY -m src.model_selection        # transformer candidate survey
$PY -m src.train_transformer --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --lr 8e-5
$PY -m src.train_transformer --model distilbert-base-multilingual-cased --lr 3e-5   # comparison model
$PY -m deployment.export_onnx     # ONNX fp32 / int8 / pruned int8 + verification
$PY -m deployment.build_e2e_model # single-file Android model (tokenizer inside) + verification
$PY -m src.evaluate --reset --models logreg naive_bayes linear_svm transformer \
    transformer:distilbert-base-multilingual-cased onnx_int8 onnx_pruned_int8
$PY -m pytest -q tests
$PY docs/build_diagrams.py && $PY docs/build_report.py && $PY docs/build_presentation.py
echo "Done. Start the demo with:  $PY app/app.py"
