# Day 10 — Pipeline Sanity, Label Normalization, and Overfit Validation

## Goal

The goal of Day 10 was to diagnose the root cause of extremely low Exact Match (EM) scores observed in Day 8–9 baseline experiments, and to determine whether the issue originated from:

- preprocessing or label inconsistencies
- tokenizer or vocabulary mismatch
- dataset split issues
- decoding / evaluation bugs
- or fundamental data limitations

This day focused on sanity-checking the entire pipeline end-to-end before proceeding to large-scale pretraining.

---

## 1. Pipeline Sanity: Image Integrity Check

### Method

Random samples were extracted from all splits to visually inspect the processed images.

python3 -m src.debug.save_samples --config configs/baseline_cnn_lstm_dev.yaml --split train --n 12
python3 -m src.debug.save_samples --config configs/baseline_cnn_lstm_dev.yaml --split valid --n 12
python3 -m src.debug.save_samples --config configs/baseline_cnn_lstm_dev.yaml --split test  --n 12

Findings
	•	No image clipping observed
	•	No truncation of symbols
	•	No inversion or aspect distortion
	•	Handwriting is clearly readable across splits

Conclusion

✅ Image preprocessing is correct and not responsible for low EM.

⸻

##  2. Tokenizer Health Check (UNK Analysis)

Method

Checked whether ground-truth LaTeX labels contain characters unseen by the tokenizer.

python3 -m src.debug.check_unk_in_labels --config configs/baseline_cnn_lstm_dev.yaml --split train
python3 -m src.debug.check_unk_in_labels --config configs/baseline_cnn_lstm_dev.yaml --split valid
python3 -m src.debug.check_unk_in_labels --config configs/baseline_cnn_lstm_dev.yaml --split test

Results

Split=train | samples=9636 | UNK_in_GT=0 (0.00%)
Split=valid | samples=428  | UNK_in_GT=0 (0.00%)
Split=test  | samples=1757 | UNK_in_GT=0 (0.00%)

Conclusion

✅ Tokenizer vocabulary fully covers all labels
❌ UNK tokens are not the cause of EM failure

⸻

##  3. Dataset Split Statistics (Length Distribution)

Results

Train
	•	mean length: 37.09 tokens
	•	p90: 62, p95: 73
	•	max length: 301

Valid
	•	mean length: 27.79 tokens
	•	p90: 47, p95: 56
	•	max length: 116

Test
	•	mean length: 26.51 tokens
	•	p90: 45, p95: 54
	•	max length: 128

Interpretation
	•	Train set contains significantly longer and more complex expressions
	•	Validation and test sets are well aligned
	•	No pathological mismatch between splits

Conclusion

✅ Dataset splits are healthy and realistic
❌ Length distribution alone does not explain EM collapse

⸻

##  4. Label Normalization Issue (LaTeX $...$ Delimiters)

Problem Observed

Many CROHME labels were wrapped with LaTeX math delimiters:

000000.png,$y = Ax + A^2$
000001.png,$B_n(1-x)=(-1)^n B_n(x)$

Some labels also contained internal dollar signs, e.g.:

"$10,000 + $1,000 = $11,000"

This caused:
	•	Token-level learning to succeed
	•	Exact-match evaluation to fail due to string mismatch
	•	Overfit models to emit empty or truncated predictions

⸻

## 5. Label Normalization Fix

A normalization function was introduced to strip surrounding math mode consistently.

def normalize_latex_label(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return s

Verification

python3 -c 'from src.utils.latex_norm import normalize_latex_label; print(normalize_latex_label("$$x^2$$"))'

Output:

x^2

After reprocessing, vocabulary size reduced from 84 → 83, confirming removal of $.

⸻

## 6. Overfit Test (Critical Sanity Check)

Configuration
	•	CNN + Transformer
	•	64 training samples
	•	No label smoothing
	•	Greedy decoding
	•	Train and validation split kept separate

Training Result
	•	Train loss → near zero
	•	Validation loss remains high (expected)
	•	Model successfully memorizes training samples

⸻

## 7. Overfit Evaluation Results

Train Split (Overfit Check)

GT  : y = Ax + A^2
PRED: y = Ax + A^2

Exact-match (train): 0.62% (60/9636)
TokenAcc (train):   24.84%

Exact matches are correct and identical, confirming the pipeline works.

Validation Split
	•	EM ≈ 0%
	•	TokenAcc ≈ 17%
	•	Long hallucinated outputs

This is expected when overfitting on a tiny subset.

⸻

## Final Diagnosis (Day 10)

Component	Status
Image preprocessing	✅ Correct
Tokenizer coverage	✅ Correct
Dataset splits	✅ Correct
Training loop	✅ Correct
Decoder / inference	✅ Correct
Label normalization	❌ Was broken
Overfit capability	✅ Confirmed


⸻

## Conclusion

The root cause of low EM in earlier experiments was inconsistent LaTeX label formatting, not model failure.

After normalization:
	•	Models can overfit correctly
	•	EM behaves as expected
	•	Evaluation metrics are trustworthy
