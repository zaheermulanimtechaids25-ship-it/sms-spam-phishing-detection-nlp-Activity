"""Build ONE end-to-end ONNX model for Android: raw (pre-processed) SMS string -> logits.

    sms_classifier_e2e.onnx
      input : "text"   string tensor, shape [1]   (output of the preprocessing/masking step)
      output: "logits" float tensor,  shape [1, 3] (ham, spam, phishing)

The SentencePiece tokenizer is embedded as an ONNX graph via onnxruntime-extensions
(ai.onnx.contrib domain), followed by a small glue graph (tokens -> input_ids/attention_mask)
and the vocabulary-pruned INT8 classifier. On Android this needs:
    com.microsoft.onnxruntime:onnxruntime-android
    com.microsoft.onnxruntime:onnxruntime-extensions-android
Verification: the merged model is run with ONNX Runtime + the extensions custom-op library
(the same runtime and ops that the Android packages ship) on the full test set and compared
with the Python OnnxPredictor.

Run (after export_onnx.py):  python -m deployment.build_e2e_model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from onnx import TensorProto, compose, helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import LABELS, METRICS_DIR, OPTIMIZED_DIR, PROCESSED_DIR, TRANSFORMER_DIR  # noqa: E402
from src.predict import DEFAULT_TRANSFORMER, OnnxPredictor  # noqa: E402
from src.preprocessing import transformer_text  # noqa: E402

MAX_LEN = 128


def build() -> Path:
    from onnxruntime_extensions import gen_processing_models
    from transformers import XLMRobertaTokenizer

    tok = XLMRobertaTokenizer.from_pretrained(TRANSFORMER_DIR / DEFAULT_TRANSFORMER) \
        if (TRANSFORMER_DIR / DEFAULT_TRANSFORMER / "sentencepiece.bpe.model").exists() \
        else XLMRobertaTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    tok_graph, _ = gen_processing_models(tok, pre_kwargs={})
    tok_in = tok_graph.graph.input[0].name
    tok_out = tok_graph.graph.output[0].name          # 1-D int64 token ids incl. <s> ... </s>

    clf = onnx.load(str(OPTIMIZED_DIR / "model_pruned_int8.onnx"))

    # glue: ids[N] (= <s> ... </s>) -> truncate like HF (first MAX_LEN-1 ids, then </s>)
    #       -> input_ids[1, n], attention_mask[1, n]
    eos = tok.eos_token_id
    tok_type = tok_graph.graph.output[0].type.tensor_type.elem_type   # int32 from the tokenizer op
    nodes = [
        helper.make_node("Cast", ["g_ids_raw"], ["g_ids"], to=TensorProto.INT64),
        helper.make_node("Shape", ["g_ids"], ["g_len"]),
        helper.make_node("Constant", [], ["g_max"], value=helper.make_tensor("v", TensorProto.INT64, [1], [MAX_LEN])),
        helper.make_node("Min", ["g_len", "g_max"], ["g_cut"]),
        helper.make_node("Constant", [], ["g_zero"], value=helper.make_tensor("z", TensorProto.INT64, [1], [0])),
        helper.make_node("Constant", [], ["g_one"], value=helper.make_tensor("o", TensorProto.INT64, [1], [1])),
        helper.make_node("Sub", ["g_cut", "g_one"], ["g_body_end"]),
        helper.make_node("Slice", ["g_ids", "g_zero", "g_body_end"], ["g_body"]),       # drop original </s>
        helper.make_node("Constant", [], ["g_eos"], value=helper.make_tensor("e", TensorProto.INT64, [1], [eos])),
        helper.make_node("Concat", ["g_body", "g_eos"], ["g_seq"], axis=0),
        helper.make_node("Unsqueeze", ["g_seq", "g_zero"], ["input_ids"]),
        helper.make_node("Shape", ["input_ids"], ["g_shape"]),
        helper.make_node("ConstantOfShape", ["g_shape"], ["attention_mask"],
                         value=helper.make_tensor("one", TensorProto.INT64, [1], [1])),
    ]
    glue = helper.make_graph(
        nodes, "glue", [helper.make_tensor_value_info("g_ids_raw", tok_type, ["n"])],
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "m"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "m"])])
    opset = [helper.make_opsetid("", 17)]
    glue_model = helper.make_model(glue, opset_imports=opset, ir_version=clf.ir_version)

    # align IR/opsets before merging
    for m in (tok_graph, glue_model):
        m.ir_version = clf.ir_version
    tok_graph = onnx.version_converter.convert_version(tok_graph, 17) if any(
        o.domain in ("", "ai.onnx") and o.version < 17 for o in tok_graph.opset_import) else tok_graph

    a = compose.merge_models(tok_graph, glue_model, io_map=[(tok_out, "g_ids_raw")], prefix1="tok_", prefix2="glue_")
    e2e = compose.merge_models(a, clf, io_map=[("glue_input_ids", "input_ids"),
                                               ("glue_attention_mask", "attention_mask")], prefix2="clf_")
    # rename IO to friendly names
    for i in e2e.graph.input:
        if i.name.endswith(tok_in):
            old = i.name
            i.name = "text"
            for n in e2e.graph.node:
                n.input[:] = ["text" if x == old else x for x in n.input]
    for o in e2e.graph.output:
        if o.name.endswith("logits"):
            old = o.name
            o.name = "logits"
            for n in e2e.graph.node:
                n.output[:] = ["logits" if x == old else x for x in n.output]
    keep = [o for o in e2e.graph.output if o.name == "logits"]
    del e2e.graph.output[:]
    e2e.graph.output.extend(keep)
    out = OPTIMIZED_DIR / "sms_classifier_e2e.onnx"
    onnx.save(e2e, str(out))
    return out


def verify(path: Path) -> dict:
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path
    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    test = pd.read_csv(PROCESSED_DIR / "test.csv", keep_default_na=False)
    ref = OnnxPredictor("model_pruned_int8").predict_proba(test.text.tolist())
    probs = []
    for t in test.text:
        lg = sess.run(["logits"], {"text": np.array([transformer_text(t)])})[0][0]
        p = np.exp(lg - lg.max())
        probs.append(p / p.sum())
    probs = np.array(probs)
    # latency (batch 1) for 1/2/4 threads and session memory (RSS delta after imports)
    import time

    import psutil
    proc = psutil.Process()
    lat = {}
    x = {"text": np.array([transformer_text(test.text.iloc[0])])}
    for th in (1, 2, 4):
        so2 = ort.SessionOptions()
        so2.register_custom_ops_library(get_library_path())
        so2.intra_op_num_threads = th
        before = proc.memory_info().rss
        s2 = ort.InferenceSession(str(path), so2, providers=["CPUExecutionProvider"])
        for _ in range(10):
            s2.run(None, x)
        ts = []
        for t in test.text.iloc[:200]:
            xx = {"text": np.array([transformer_text(t)])}
            t0 = time.perf_counter()
            s2.run(None, xx)
            ts.append((time.perf_counter() - t0) * 1000)
        lat[f"{th}_threads"] = {"median_ms": round(float(np.median(ts)), 2),
                                "p95_ms": round(float(np.percentile(ts, 95)), 2)}
        if th == 1:
            session_mb = round((proc.memory_info().rss - before) / 1024 ** 2, 1)
        del s2
    res = {"file": path.name, "size_mb": round(path.stat().st_size / 1024 ** 2, 2),
           "inputs": [(i.name, i.type, i.shape) for i in sess.get_inputs()],
           "outputs": [(o.name, o.type, o.shape) for o in sess.get_outputs()],
           "test_messages": len(test),
           "label_agreement_with_python_onnx_pct": round(100 * float((probs.argmax(1) == ref.argmax(1)).mean()), 2),
           "max_abs_prob_diff": round(float(np.abs(probs - ref).max()), 5),
           "test_accuracy": round(float((np.array(LABELS)[probs.argmax(1)] == test.label.values).mean()), 4),
           "latency_cpu": lat, "session_rss_increase_mb": session_mb}
    return res


if __name__ == "__main__":
    p = build()
    r = verify(p)
    rep_path = METRICS_DIR / "onnx_export_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    rep["end_to_end_model"] = r
    rep_path.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(r, indent=2, default=str))
