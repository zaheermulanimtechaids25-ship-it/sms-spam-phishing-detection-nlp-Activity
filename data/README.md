# Data card

All data is produced by `python -m src.data_loader`. Raw third-party files are downloaded into
`data/raw/` (git-ignored) and are **not** redistributed in this repository.

## Sources

| Source | Labels | Size (raw) | Licence | How obtained |
|---|---|---|---|---|
| UCI SMS Spam Collection [1] | ham / spam | 5,574 (4,827 ham, 747 spam) | CC BY 4.0 | automatic download from archive.ics.uci.edu |
| Mishra & Soni SMS Phishing Dataset [2] | ham / spam / smishing | 5,971 (4,844 ham, 489 spam, 638 smishing) | CC BY 4.0 | automatic download from Mendeley Data (SHA-256 verified) |
| Synthetic Hinglish SMS (this project) | ham / spam / phishing | 447 generated -> 357 after de-dup | MIT (this repo) | `src/hinglish_generator.py`, saved to `data/hinglish/` |

[1] T. A. Almeida, J. M. G. Hidalgo, A. Yamakami, "Contributions to the study of SMS spam filtering: new collection and results," ACM DocEng 2011.
[2] S. Mishra, D. Soni, "SMS Phishing Dataset for Machine Learning and Pattern Recognition," Mendeley Data, V1, 2022, doi:10.17632/f45bkkt8pr.1.

## Three-class label scheme and mapping

| Class | Meaning in this project |
|---|---|
| `ham` | legitimate message |
| `spam` | unwanted promotional / bulk message |
| `phishing` | fraud that tries to obtain credentials, OTPs, banking details, money or other sensitive data |

* The UCI dataset only has ham/spam. **We do not relabel UCI spam as phishing ourselves.**
* The Mishra & Soni dataset re-uses most UCI messages and adds author-assigned `smishing` labels.
  Where a message appears there, that label is used (`smishing` -> `phishing`).
* Messages found only in UCI keep their UCI label.
* Original labels are preserved in the columns `label_uci` and `label_ms`.
* 37 messages that carry conflicting labels inside the Mishra & Soni file were dropped.
  No ham-vs-non-ham conflicts were found between the two sources.

**Limitation.** In the English data, "phishing" follows Mishra & Soni's definition of smishing.
That definition is broader than credential theft and includes prize/lottery fraud ("You have won
£1000, call 09061..."). The spam/smishing boundary in that dataset is also noisy. A model trained
on it learns that boundary, so a message such as "Congratulations! You have won a free gift.
Claim now." is likely to be predicted *phishing* rather than *spam*.

## Synthetic Hinglish supplement

Generated with 90 templates written by the project team (32 phishing, 26 spam, 32 ham). The
templates imitate publicly reported Indian SMS scam patterns: fake KYC/PAN/Aadhaar updates,
UPI collect-request and refund fraud, account/SIM/electricity blocking threats, OTP requests,
fake delivery/customs fees, lottery/KBC prizes, fake jobs, tax refunds, loan fees and family/police
impersonation. Ham templates cover college, family, friends, work, appointments and legitimate
transaction alerts, including OTP and debit alerts that look similar to scams.

* **These messages are synthetic, not real collected SMS.** Each row has `source = synthetic_hinglish`.
* Slot filling (bank, wallet, courier, amount, time, ...) and style perturbations (`nahi/nhi`,
  `karo/kro`, upper/lower case) create variants.
* URLs use the reserved `.example` domain, phone numbers use the unallocated `+91 55555 xxxxx`
  range, and there are no personal names.
* **Leakage control:** splits are made by `template_id`, so variants of one template never appear
  in both training and test data.

## Privacy and PII handling

Before anything is written to `data/processed/`:

| Pattern | Replaced by |
|---|---|
| e-mail addresses | `<email>` |
| URLs (scheme, `www.`, bare domains) | `<url>` |
| phone numbers (9+ digits, incl. separators and digits glued to words) | `<phone>` |
| remaining digit runs of 4+ (OTPs, PINs, account fragments, short codes) | `<num>` |

`tests/test_preprocessing.py::test_processed_data_contains_no_raw_phone_numbers_or_urls`
checks the processed files. Personal first names that occur in the public UCI messages were
not removed, because no reliable name recogniser exists for SMS slang and Hinglish. Both public
datasets are already anonymised by their publishers.

## Final processed dataset

| Split | Total | ham | spam | phishing | English | Hinglish (synthetic) |
|---|---|---|---|---|---|---|
| train | 4,419 | 3,624 | 403 | 392 | 4,174 | 245 |
| val | 957 | 780 | 88 | 89 | 895 | 62 |
| test | 945 | 779 | 83 | 83 | 895 | 50 |

* English messages after de-duplication: 5,964 (5,057 ham, 474 spam, 433 phishing).
* De-duplication uses the PII-masked, lower-cased text, so messages that differ only in a phone
  number or amount count as duplicates.
* Leakage checks: 0 identical texts and 0 shared Hinglish templates across splits
  (`results/metrics/dataset_stats.json`).
* The Hinglish test subset is small (50 messages: 20 ham, 12 spam, 18 phishing, from 14 templates never seen in training), so per-class Hinglish
  scores carry high variance.

## Files

| File | Description |
|---|---|
| `data/processed/{train,val,test}.csv` | `text` (masked), `label`, `language`, `source`, `category`, `template_id`, `label_uci`, `label_ms`, `hinglish_detected` |
| `data/hinglish/hinglish_sms_synthetic.csv` | the generated synthetic messages before masking (placeholder links/numbers only) |
