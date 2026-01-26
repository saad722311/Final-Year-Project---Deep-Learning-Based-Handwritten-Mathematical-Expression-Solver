# Day 9 — Full-Split Evaluation (Valid + Test) for CROHME23 Baselines

## Goal
Move beyond partial evaluation (`--max_eval 300`) and report **full-split** performance for the Day 8 baselines on CROHME23 processed dataset:

- **CNN Encoder + LSTM Decoder (attention)**
- **CNN Encoder + Transformer Decoder**

Evaluate on:
- **Valid split (n=428)**
- **Test split (n=1757)**

Metrics reported:
- **EM**: Exact Match (string match after basic trimming)
- **nEM**: Normalized Exact Match (removes surrounding `$...$` and whitespace)
- **TokenAcc**: Teacher-forcing token accuracy (ignoring PAD)
- **BraceOK**: Brace-balance rate (well-formed `{}` count)
- **AvgLen**: Average decoded length (tokens)

---

## Experiment Setup

### Model Checkpoints (Day 8)
- **LSTM**: `results/crohme_cnn_lstm_dev_day_8/best.pt`
- **Transformer**: `results/crohme_cnn_transformer_dev_day8/best.pt`

### Common Decoding Constraints
Used for all evaluations:

- `max_len = 256`
- `min_len = 10`
- `repetition_penalty = 1.15`
- `no_repeat_ngram_size = 3`
- `forbid_unk = True`

Decoding modes:
- **Greedy**
- **Beam search** (`beam_size=5`, `alpha=0.6`)

Outputs saved under:
- `results/compare/day9/`

---

## Commands Used (Full-Split Evaluation)

### LSTM — Valid (Full)
```bash
python3 -m src.train.infer \
  --config configs/baseline_cnn_lstm_dev.yaml \
  --ckpt results/crohme_cnn_lstm_dev_day_8/best.pt \
  --split valid \
  --eval_all \
  --decode greedy \
  --max_len 256 \
  --min_len 10 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --forbid_unk \
  --out results/compare/day9/lstm_valid_greedy_full.txt

python3 -m src.train.infer \
  --config configs/baseline_cnn_lstm_dev.yaml \
  --ckpt results/crohme_cnn_lstm_dev_day_8/best.pt \
  --split valid \
  --eval_all \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --max_len 256 \
  --min_len 10 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --forbid_unk \
  --out results/compare/day9/lstm_valid_beam5a06_full.txt

```bash

### LSTM — Test (Full)

```bash
python3 -m src.train.infer \
  --config configs/baseline_cnn_lstm_dev.yaml \
  --ckpt results/crohme_cnn_lstm_dev_day_8/best.pt \
  --split test \
  --eval_all \
  --decode greedy \
  --max_len 256 \
  --min_len 10 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --forbid_unk \
  --out results/compare/day9/lstm_test_greedy_full.txt

python3 -m src.train.infer \
  --config configs/baseline_cnn_lstm_dev.yaml \
  --ckpt results/crohme_cnn_lstm_dev_day_8/best.pt \
  --split test \
  --eval_all \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --max_len 256 \
  --min_len 10 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --forbid_unk \
  --out results/compare/day9/lstm_test_beam5a06_full.txt
```bash

### Transformer — Valid (Full)

```bash
python3 -m src.train.infer \
  --config configs/baseline_cnn_transformer_dev.yaml \
  --ckpt results/crohme_cnn_transformer_dev_day8/best.pt \
  --split valid \
  --eval_all \
  --decode greedy \
  --max_len 256 \
  --min_len 10 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --forbid_unk \
  --out results/compare/day9/transformer_valid_greedy_full.txt

python3 -m src.train.infer \
  --config configs/baseline_cnn_transformer_dev.yaml \
  --ckpt results/crohme_cnn_transformer_dev_day8/best.pt \
  --split valid \
  --eval_all \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --max_len 256 \
  --min_len 10 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --forbid_unk \
  --out results/compare/day9/transformer_valid_beam5a06_full.txt
```bash

### Transformer — Test (Full)

```bash
python3 -m src.train.infer \
  --config configs/baseline_cnn_transformer_dev.yaml \
  --ckpt results/crohme_cnn_transformer_dev_day8/best.pt \
  --split test \
  --eval_all \
  --decode greedy \
  --max_len 256 \
  --min_len 10 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --forbid_unk \
  --out results/compare/day9/transformer_test_greedy_full.txt

python3 -m src.train.infer \
  --config configs/baseline_cnn_transformer_dev.yaml \
  --ckpt results/crohme_cnn_transformer_dev_day8/best.pt \
  --split test \
  --eval_all \
  --decode beam \
  --beam_size 5 \
  --alpha 0.6 \
  --max_len 256 \
  --min_len 10 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --forbid_unk \
  --out results/compare/day9/transformer_test_beam5a06_full.txt
```bash

## Results — Validation Set (Full, n=428)

| Model | Decode | EM | nEM | TokenAcc | BraceOK | AvgLen |
|------|--------|----|-----|----------|---------|--------|
| CNN + LSTM (Day8) | Greedy | 0.23% (1/428) | 0.23% (1/428) | 66.33% (7889/11893) | 92.29% (395/428) | 23.4 |
| CNN + LSTM (Day8) | Beam (5, α=0.6) | 0.23% (1/428) | **0.93% (4/428)** | 66.33% (7889/11893) | 93.93% (402/428) | 21.9 |
| CNN + Transformer (Day8) | Greedy | 0.23% (1/428) | 0.47% (2/428) | **66.59% (7920/11893)** | 86.92% (372/428) | 26.4 |
| CNN + Transformer (Day8) | Beam (5, α=0.6) | 0.00% (0/428) | 0.47% (2/428) | **66.59% (7920/11893)** | **94.16% (403/428)** | 21.9 |

---

## Results — Test Set (Full, n=1757) ✅ *(Main Result)*

| Model | Decode | EM | nEM | TokenAcc | BraceOK | AvgLen |
|------|--------|----|-----|----------|---------|--------|
| CNN + LSTM (Day8) | Greedy | 0.11% (2/1757) | 0.17% (3/1757) | 61.06% (28444/46584) | 90.55% (1591/1757) | 23.7 |
| CNN + LSTM (Day8) | Beam (5, α=0.6) | 0.11% (2/1757) | 0.17% (3/1757) | 61.06% (28444/46584) | 94.71% (1664/1757) | 22.2 |
| CNN + Transformer (Day8) | Greedy | 0.11% (2/1757) | 0.28% (5/1757) | **62.49% (29109/46584)** | 87.31% (1534/1757) | 26.3 |
| CNN + Transformer (Day8) | Beam (5, α=0.6) | **0.23% (4/1757)** | **0.40% (7/1757)** | **62.49% (29109/46584)** | **95.22% (1673/1757)** | 21.3 |

---

## Key Findings

### 1) Full-split evaluation confirms the same pattern

Compared to the earlier *first-300* experiments, the absolute values remain low, but full evaluation confirms:

- **TokenAcc ≈ 61–66%** across validation and test
- **EM remains near 0%** for both baselines

This indicates the models are learning **token-level structure**, but are not yet producing **fully correct LaTeX sequences end-to-end**.

---

### 2) Best overall test result: Transformer + Beam

On the **test set**, the strongest configuration is:

- **CNN + Transformer + Beam (k=5, α=0.6)**
- **EM:** 0.23% (4/1757)
- **nEM:** 0.40% (7/1757)
- **TokenAcc:** 62.49%
- **BraceOK:** 95.22%

This suggests that **Transformer decoding combined with beam search** is currently the best baseline choice.

---

### 3) Beam search mainly improves syntax validity and compactness

Beam search consistently improves brace balance and reduces decoded length:

- **LSTM (test):** BraceOK 90.55% → 94.71%
- **Transformer (test):** BraceOK 87.31% → 95.22%
- **AvgLen decreases** for both models (beam produces more compact outputs)

Thus, beam search acts primarily as a **stability / structural improvement**, even when EM does not increase significantly.

---

### 4) Generalization gap (Validation → Test)

Token accuracy drops from validation (~66%) to test (~61–62%), indicating a moderate **generalization gap**:

- **LSTM:** 66.33% → 61.06%
- **Transformer:** 66.59% → 62.49%

This is expected given that:
- CROHME is a **small dataset**
- handwriting variability is high
- **Exact Match is extremely strict** for long structured expressions

---

## Conclusion (Day 9)

Day 9 successfully produced **trustworthy baseline benchmarks** by evaluating both models on **full validation and test splits**.

- EM remains extremely low for both baselines
- Transformer shows slightly better token accuracy and the **best test EM/nEM when paired with beam**
- Beam search primarily improves **structural validity (BraceOK)** and prevents overly long predictions

### ✅ Current best baseline configuration (for reporting / future integration)

**CNN + Transformer (Day 8 checkpoint) + Beam search (k=5, α=0.6)**

---
