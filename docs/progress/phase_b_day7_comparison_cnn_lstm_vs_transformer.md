# Day 7 — Baseline Comparison (CNN + LSTM vs CNN + Transformer)

## Goal
Compare two baseline decoders for handwritten mathematical expression recognition on the CROHME23 processed dataset:

- **CNN Encoder + LSTM Decoder (with attention)**
- **CNN Encoder + Transformer Decoder**

Evaluation metrics:
- **Exact Match (EM)** — full LaTeX sequence must match
- **Normalized Exact Match (nEM)** — ignores whitespace and surrounding `$...$`
- **Teacher-forcing Token Accuracy (TokenAcc)**

All evaluations are performed on the **first 300 samples** (`--max_eval 300`).

---

## Experiment Setup

### Common Inference Settings
All models were evaluated using identical decoding constraints:

- `max_len = 256`
- `min_len = 10`
- `repetition_penalty = 1.15`
- `no_repeat_ngram_size = 3`
- `forbid_unk = True`

Decoding strategies:
- **Greedy**
- **Beam Search (beam_size = 5, alpha = 0.6)**

Saved outputs:
- `results/compare/transformer_valid_greedy_300.txt`
- `results/compare/transformer_valid_beam5a06_300.txt`
- `results/compare/lstm_valid_greedy_300.txt`
- `results/compare/lstm_valid_beam5a06_300.txt`

---

## Training Observation (Overfitting)
During training, validation loss decreased until a certain epoch and then started increasing, while training loss continued to decrease.

This indicates **overfitting**:
- the model increasingly memorizes training data
- generalization to validation samples degrades after the best epoch

**Action:** all evaluations are performed using `best.pt` (lowest validation loss).

---

## Results — Validation Set (First 300 Samples)

| Model | Decode | EM | nEM | TokenAcc |
|------|--------|----|-----|----------|
| CNN + Transformer | Greedy | 0.00% (0/300) | 0.00% (0/300) | 61.96% (5078/8195) |
| CNN + Transformer | Beam (5, α=0.6) | 0.33% (1/300) | 0.33% (1/300) | 61.96% (5078/8195) |
| CNN + LSTM | Greedy | 0.00% (0/300) | 0.33% (1/300) | 57.62% (4722/8195) |
| CNN + LSTM | Beam (5, α=0.6) | 0.00% (0/300) | 0.33% (1/300) | 57.62% (4722/8195) |

---

## Results — Training Set (First 300 Samples)

| Model | Decode | EM | nEM | TokenAcc |
|------|--------|----|-----|----------|
| CNN + LSTM | Greedy | 3.33% (10/300) | 8.00% (24/300) | 82.35% |
| CNN + LSTM | Beam (5, α=0.6) | **6.67% (20/300)** | **12.00% (36/300)** | 82.35% |
| CNN + Transformer | Greedy | 4.00% (12/300) | 4.00% (12/300) | **90.14%** |
| CNN + Transformer | Beam (5, α=0.6) | 3.33% (10/300) | 3.33% (10/300) | **90.14%** |

---

## Key Findings

### 1) Token Accuracy vs Exact Match
Both models achieve relatively high token accuracy but extremely low exact match.

This is expected in handwritten math recognition:
- EM requires **every token** to be correct
- a single incorrect symbol breaks the entire sequence
- TokenAcc can improve while EM remains near zero

---

### 2) Transformer vs LSTM
- Transformer consistently achieves **higher token accuracy**
- LSTM shows slightly higher EM on the training set with beam search
- On validation data, neither model achieves strong EM yet

This suggests the Transformer is a stronger baseline architecture, but decoding stability and generalization remain limiting factors.

---

### 3) Train vs Validation Gap
- Train EM > Validation EM for both models
- Indicates the models are learning, but overfitting occurs
- Sequence-level generalization remains weak

---

## Conclusion (Day 7)
- Both CNN+LSTM and CNN+Transformer baselines learn token-level structure but struggle with full-expression correctness.
- Transformer decoder is the more promising baseline due to higher token accuracy.
- Improving exact match will require better generalization, decoding strategies, and possibly stronger encoders.

---

## Next Steps

### A) Improve Generalization
- Increase regularization (dropout, weight decay)
- Apply early stopping
- Introduce learning-rate scheduling

### B) Improve Sequence-Level Accuracy
- Label smoothing
- Scheduled sampling
- Coverage-aware beam search

### C) Application Pipeline
If baseline EM remains limited, integrate a pretrained HMER model and focus on:
1. Stylus-based input interface
2. Image preprocessing and LaTeX recognition
3. LLM-based step-by-step solution generation
4. End-to-end interactive system