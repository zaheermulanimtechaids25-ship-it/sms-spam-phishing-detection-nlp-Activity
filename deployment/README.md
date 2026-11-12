# Mobile deployment

## Files (models/optimized/)

| File | Size | Purpose |
|---|---|---|
| `model_fp32.onnx` | 449 MB | plain ONNX export (reference) |
| `model_int8.onnx` | 113 MB | dynamic INT8 quantization |
| `model_pruned_int8.onnx` | 32.3 MB | INT8 + vocabulary pruning (**deployed classifier**) |
| `sms_classifier_e2e.onnx` | 37.2 MB | tokenizer + classifier in one graph; input: string, output: logits (**Android**) |
| `sentencepiece.bpe.model` | 4.8 MB | tokenizer for `model_pruned_int8.onnx` (XLM-R id convention) |
| `labels.json` | <1 KB | label order, max length, tokenizer convention |

## How the model was optimised

1. **Export:** `torch.onnx.export`, opset 17, dynamic batch and sequence axes.
2. **Dynamic INT8 quantization:** `onnxruntime.quantization.quantize_dynamic` on MatMul/Gemm/Gather
   (weights int8, activations quantized at run time).
3. **Vocabulary pruning:** keep the 20,000 most frequent SentencePiece pieces plus all pieces that
   occur in the *training* split (25,103 of 250,037 rows). A lookup table inside the graph maps
   original token IDs to the new rows, so the original tokenizer still works. The trade-off for other
   sizes is in `results/metrics/pruning_ablation.json`.
4. **Embedded tokenizer:** `onnxruntime_extensions.gen_processing_models` turns the SentencePiece
   tokenizer into ONNX operators. A small glue graph (cast, truncate to 128, add `</s>`, build the
   attention mask) and the classifier are merged with `onnx.compose`.

## Verification (all 945 test messages, `results/metrics/onnx_export_report.json`)

| Check | Result |
|---|---|
| ONNX fp32 vs PyTorch | 100% same labels |
| ONNX INT8 vs PyTorch | 99.89% same labels |
| pruned INT8 vs PyTorch | 99.58% same labels; test accuracy 0.9619 (PyTorch 0.9630) |
| end-to-end file vs Python pruned-INT8 pipeline | 100% same labels, max probability difference 0.0 |
| SentencePiece ids vs training tokenizer | identical on all 5,364 train + test messages |
| Kotlin/Python masking parity | the Python mirror (`mobile_inference.preprocess`) is unit-tested against training preprocessing; the Kotlin port uses the same regexes but is untested |

Latency of `sms_classifier_e2e.onnx` (desktop CPU, batch 1, 200 test SMS): 2.39 ms (1 thread),
1.88 ms (2 threads), 1.67 ms (4 threads); about 52 MB of session RAM.

**Note on padding:** dynamic INT8 quantization computes activation scales over the whole input
tensor, so a padded message in a batch can get slightly different scores. On a phone, messages are
classified one at a time without padding, and all ONNX numbers are measured that way.

## Running it without PyTorch

```bash
pip install onnxruntime sentencepiece numpy
python deployment/mobile_inference.py "Aapka SBI account band ho jayega. KYC update karo" --benchmark
```

## Android integration (reference code, not device-tested)

1. Copy `models/optimized/sms_classifier_e2e.onnx` to `app/src/main/assets/`.
2. Add the dependencies to `app/build.gradle.kts`:
   ```kotlin
   implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
   implementation("com.microsoft.onnxruntime:onnxruntime-extensions-android:0.13.0")
   ```
   Both versions exist on Maven Central. The model was verified with onnxruntime 1.31.1 and
   onnxruntime-extensions 0.15.2 (Python). 0.13.0 is the newest Android extensions release, and the
   SentencePiece tokenizer op it needs is long-established, but compatibility has not been tested on a device.
3. Use `deployment/android/SmsPhishingClassifier.kt`:
   ```kotlin
   val clf = SmsPhishingClassifier(context)
   val r = clf.classify(smsBody)          // r.label, r.confidence, r.scores
   if (r.label == "phishing") showWarning()
   ```
4. Run classification off the main thread (e.g. a coroutine on `Dispatchers.Default`). Never open
   links from the SMS automatically.

**What is and is not verified.** The ONNX files were built here and checked with ONNX Runtime and
the extensions custom-op library on a desktop CPU. The Kotlin file was written as integration
guidance. It has **not** been compiled or run on an Android device or emulator by the project team,
so its on-device latency and memory are not measured.

## Alternatives considered

* TensorFlow Lite: would require converting PyTorch to TF; ONNX Runtime Mobile runs the exported model directly.
* Keeping `tokenizer.json` (Hugging Face `tokenizers`): about 350 MB of RAM to load the 250k-entry
  vocabulary, against about 43 MB for the 4.8 MB SentencePiece model.
