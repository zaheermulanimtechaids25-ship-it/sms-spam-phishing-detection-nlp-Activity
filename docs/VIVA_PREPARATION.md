# Viva preparation: 36 questions with answers

Numbers refer to `results/metrics/comparison_table.md` (test set, 945 messages).

## A. NLP fundamentals and text classification

**1. What is text classification? What kind is this project?**
Assigning a predefined label to a piece of text. This is single-label, multi-class (3 classes:
ham, spam, phishing) classification of short texts.

**2. Why is SMS harder than normal text?**
SMS messages are short (median 13 words; 11 for ham, 23 for spam and phishing), with little context. They contain abbreviations
("u", "ur", "2day"), creative spelling ("FREEE"), numbers, links and code-mixing. The classes are
also imbalanced: about 82% ham.

**3. What is code-mixing / Hinglish, and why is it difficult?**
Hinglish is switching between Hindi and English inside one message, with Hindi written in the
Roman script. There is no standard spelling ("nahi", "nhi", "nahin"), most NLP tools assume one
language, and Romanized Hindi is rare in pre-training corpora.

**4. Why didn't you translate Hinglish into English first?**
Translation tools handle Romanized Hindi poorly and can drop the exact scam cues ("band ho
jayega", "abhi"). We preserve the original words, normalise only frequent spelling variants, and
let character n-grams and sub-word tokenization handle the rest.

**5. Why no stemming or stop-word removal?**
Function words carry signal here: "your", "now", "abhi", "karo" and "aapka" are typical of
phishing. Stemmers are built for English and would corrupt Hindi words.

## B. TF-IDF, n-grams, Logistic Regression

**6. Explain TF-IDF.**
`tfidf(t, d) = tf(t, d) * log(N / df(t))`, normalised per document. A term gets a high weight if it
is frequent in the message but rare in the corpus. We use sub-linear tf (1 + log tf).

**7. What are n-grams? Why word (1-2) and character (2-5)?**
An n-gram is a sequence of n tokens or characters. Word bigrams capture phrases ("claim your",
"update karo"). Character n-grams are robust to misspellings and Hinglish variants, because "nahi"
and "nhi" share "nh" and "hi"; they also capture sub-word cues like "ww." or "£".

**8. How does Logistic Regression do multi-class classification?**
It learns one weight vector per class. Softmax of the linear scores w_k·x + b_k gives class
probabilities, and training minimises cross-entropy with L2 regularisation (strength 1/C;
C = 1 was chosen by 5-fold CV).

**9. Why is TF-IDF + LR a good baseline?**
It is fast (0.45 ms per SMS), tiny (1.1 MB) and interpretable: every n-gram has a weight, which
we show in the demo. It is a strong reference on short-text tasks. Here it actually had the best
test macro-F1 (0.894).

**10. How did you handle class imbalance?**
With `class_weight="balanced"` for LR/SVM and a weighted cross-entropy for the transformer
(weights inversely proportional to class frequency). We also report macro metrics, which weight
the small classes equally.

## C. Transformers, tokenization, attention, fine-tuning

**11. Explain self-attention.**
Each token is projected to query, key and value vectors, and attention = softmax(QK^T / sqrt(d)) V.
Every token gathers information from every other token, weighted by relevance. Multi-head
attention does this in several sub-spaces in parallel.

**12. Transformer encoder layer?**
Multi-head self-attention, then add & layer-norm, then a position-wise feed-forward network, then
add & layer-norm. Positional embeddings give word order. MiniLM-L12 stacks 12 such layers with hidden size 384.

**13. What is MiniLM, and why this model?**
MiniLM is a small transformer distilled from a large one by mimicking its self-attention.
paraphrase-multilingual-MiniLM-L12-v2 was further distilled for 50+ languages. Among four
candidates it was the fastest (13.6 ms on CPU), produced the fewest pieces per Hinglish word (1.61),
had no [UNK] tokens, and 82% of its size is the vocabulary table, which we could prune.

**14. How does the tokenizer handle Hinglish?**
It uses SentencePiece unigram with the XLM-R vocabulary (250k pieces). Unknown words are split
into known sub-words ("jayega" -> "▁ja", "y", "ega"), so nothing becomes [UNK]. This does not mean
the model "understands" Hindi; that is why we measure Hinglish performance separately.

**15. What is fine-tuning, and what settings did you use?**
Continuing to train a pre-trained model on the labelled task, with a new classification head.
Settings: AdamW, lr 8e-5 with warm-up and linear decay, batch 32, max 128 tokens (99% of messages
are under 70 tokens), class-weighted loss, early stopping (patience 2) on validation macro-F1.
It ran 6 epochs in 4.6 minutes on the Apple GPU.

**16. Multilingual model vs Hinglish-specific model (HingBERT)?**
HingBERT is pre-trained on Romanized Hinglish, which is attractive. But it is a full BERT-base
(768 hidden, 31 ms on CPU), more than twice as slow as MiniLM, and less suited to a phone. It is a
good candidate for future work.

**17. Why did DistilBERT lose?**
Validation macro-F1 was lower (0.915 vs 0.925), training was 3x slower, CPU latency was higher, and
test Hinglish macro-F1 was much lower (0.79). It did have higher validation phishing recall (0.92),
which we report.

## D. Evaluation

**18. Define precision, recall and F1.**
Precision = TP / (TP + FP): of messages flagged phishing, how many are phishing.
Recall = TP / (TP + FN): of real phishing messages, how many we caught.
F1 = 2PR / (P + R).

**19. Why macro-F1 and not accuracy?**
A model that says "ham" for everything scores about 82% accuracy. Macro-F1 averages F1 over classes
equally, so the small spam and phishing classes count as much as ham.

**20. Why emphasise phishing recall and false negatives?**
A missed phishing SMS can cause direct financial loss, while a false alarm is only an annoyance.
We also report "phishing predicted as ham" separately: a phishing message predicted as spam is still
flagged, but one predicted as ham reaches the user unflagged. That happened for only 1 (LR) and
1-2 (transformer) of 83 test phishing messages.

**21. Read your confusion matrix.**
Rows are true classes and columns are predictions. Ham is almost always right (false-alarm rate:
LR 1.2%, transformer 0.5%). Most errors are in the spam/phishing block, because the source labels
are inconsistent there (e.g. a "new Nokia phone, reply or call" offer is labelled smishing).

**22. What do ROC-AUC and PR-AUC tell you here?**
ROC-AUC (about 0.99) measures how well the scores rank classes across all thresholds. PR-AUC for
phishing (0.93-0.94) is more informative for a rare class, because it ignores the many true negatives.

**23. Which model is better?**
It depends on the criterion. The baseline has better macro-F1 (0.894 vs 0.880) and phishing recall
(0.892 vs 0.819). The transformer has better phishing precision (0.87 vs 0.81), fewer false alarms,
and better Hinglish macro-F1 on unseen templates (0.89 vs 0.85). A practical system can run both and
flag a message if either says fraud.

**24. Did you try to raise phishing recall?**
Yes. We tuned a lower phishing threshold on the validation set. It helped only a little (transformer
test recall 0.807 -> 0.831), because the transformer is very confident and its misses are confident
"spam" predictions.

## E. Data and leakage

**25. What is data leakage? How did you prevent it?**
Information from the test set influencing training, which inflates scores. We:
* de-duplicate on PII-masked text, so messages that differ only in phone numbers are treated as duplicates;
* tune hyper-parameters only on train (CV) or validation, and touch the test set once;
* split synthetic Hinglish **by template**, so test templates are never seen in training;
* build the pruned vocabulary from the training split only.
Checks show 0 shared texts and 0 shared templates.

**26. How did you get a phishing label if UCI only has ham/spam?**
We did not relabel UCI ourselves. The Mishra & Soni dataset (CC BY 4.0) contains most UCI messages
with author-assigned smishing labels; we use those and keep the original labels in separate columns.
Limitation: their smishing definition includes prize fraud, and the spam/smishing boundary is noisy.

**27. Is the Hinglish data real?**
No. It is synthetic, generated from 90 templates written to imitate publicly reported Indian scam
patterns. Every row is marked `synthetic_hinglish`. Links use `.example` and phone numbers use an
unallocated range. Results on it show generalisation to new templates, not real-world accuracy.

**28. How did you remove personal information?**
URLs, e-mails, phone numbers (including digits glued to words) and any 4+ digit run are replaced by
`<url>`, `<email>`, `<phone>` and `<num>` before saving. A unit test fails if any 9+ digit number or URL
remains. Names in the public data were not removed, which is a stated limitation.

## F. Optimisation and mobile inference

**29. What is quantization? Which type did you use?**
Representing weights and activations with fewer bits. We used dynamic INT8: weights stored as int8,
activations quantized on the fly per input, with no calibration data. Size went from 449 MB to 113 MB
with 99.89% identical labels.

**30. What is vocabulary pruning, and is it safe?**
We deleted embedding rows for sub-words that are neither in the 20k most frequent pieces nor in our
training data (25,103 of 250,037 rows kept). A lookup table maps original IDs to new rows, and
removed IDs map to `<unk>`. Size dropped to 32.3 MB with 99.58% identical labels. 2.1% of test tokens
(4.6% of Hinglish tokens) became `<unk>`. The ablation shows macro-F1 is almost flat from 26 to 60 MB.

**31. What is ONNX? Why ONNX Runtime Mobile?**
ONNX is an open format for neural-network graphs. ONNX Runtime runs it on CPU, GPU and mobile, and
the Android package can run our exported model directly without converting to TensorFlow.

**32. How does the phone tokenize text?**
The SentencePiece tokenizer is converted into ONNX operators (onnxruntime-extensions) and merged
into the same file, so the app passes a string and gets three logits back. We verified this file
gives identical results to the Python pipeline on all 945 test messages.

**33. Did you test it on a phone?**
No. We verified the ONNX models in ONNX Runtime on a desktop CPU (1.7-2.4 ms per SMS, about 52 MB
session RAM) and wrote the Kotlin integration, but it has not been compiled or run on a device.
That is listed as future work.

**34. Why are ONNX results measured one message at a time?**
With dynamic INT8, activation scales are computed over the whole input tensor, so padding a message
inside a batch changes its scores slightly. Phones classify one message at a time, so we evaluate
that way.

## G. Limitations and ethics

**35. Main limitations?**
* Synthetic Hinglish data, and only 50 Hinglish test messages.
* Broad and noisy smishing labels.
* The English data is from 2011-12 (few UPI/KYC scams).
* The language detector is a heuristic.
* No on-device test.
* Names are not masked.

**36. Ethical and safety considerations?**
* We used only openly licensed data and cited it.
* We masked PII and created synthetic data without real people's details.
* The demo never opens links and shows defanged URLs.
* Predictions can be wrong, so the app shows evidence and advises users to verify with the bank
  through official channels.

---

## How each member explains their contribution

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for the matrix and a 2-3 minute script for each member.

* **Niraj:** dataset sources and licences, the label-mapping decision, PII masking (show the unit
  test), the Hinglish template design and the template-level split, the TF-IDF features and grid
  search, and how `src/evaluate.py` computes every metric. Demo: `notebooks/02_classical_ml.ipynb`,
  the top n-grams per class, and the confusion matrix.
* **Zaheer:** the model-selection table, the fine-tuning configuration and training curves, ONNX
  export, quantization, vocabulary pruning, the ablation, the end-to-end model and its verification,
  the Kotlin integration, and the Gradio app. Demo: run `python app/app.py` live and
  `python deployment/mobile_inference.py --benchmark`.
* **Both:** be ready to explain the other person's model at a high level. Questions 6-10 and 11-15
  are the minimum each of you should be able to answer.
