## Day 8 — Regularization + Structure Metrics (CNN+LSTM vs CNN+Transformer)

### What changed in Day 8
We introduced training regularization and improved evaluation visibility:
- weight_decay = 0.01
- dropout = 0.2
- label_smoothing = 0.1
- Added structure metrics during inference:
  - Brace-balance rate
  - Avg decoded length

All results below are on first 300 samples.

---

## Results (Valid, first 300)

| Model | Decode | EM | nEM | TokenAcc | Brace-balance | Avg len |
|------|--------|----|-----|----------|--------------|--------|
| CNN+LSTM (Day8) | Greedy | 0.33% (1/300) | 0.33% (1/300) | 64.39% (5277/8195) | 93.00% (279/300) | 23.2 |
| CNN+LSTM (Day8) | Beam (5, α=0.6) | 0.33% (1/300) | 1.00% (3/300) | 64.39% (5277/8195) | 93.00% (279/300) | 21.3 |
| CNN+Transformer (Day8) | Greedy | 0.33% (1/300) | 0.67% (2/300) | 65.53% (5370/8195) | 86.67% (260/300) | 25.7 |
| CNN+Transformer (Day8) | Beam (5, α=0.6) | 0.00% (0/300) | 0.67% (2/300) | 65.53% (5370/8195) | 93.00% (279/300) | 22.0 |

---

## Results (Train, first 300)

| Model | Decode | EM | nEM | TokenAcc | Brace-balance | Avg len |
|------|--------|----|-----|----------|--------------|--------|
| CNN+LSTM (Day8) | Greedy | 11.67% (35/300) | 12.00% (36/300) | 91.73% (10827/11803) | 82.00% (246/300) | 47.1 |
| CNN+LSTM (Day8) | Beam (5, α=0.6) | 11.67% (35/300) | 16.67% (50/300) | 91.73% (10827/11803) | 87.33% (262/300) | 40.0 |
| CNN+Transformer (Day8) | Greedy | 11.33% (34/300) | 11.33% (34/300) | 91.15% (10759/11803) | 69.33% (208/300) | 40.0 |
| CNN+Transformer (Day8) | Beam (5, α=0.6) | 11.67% (35/300) | 13.67% (41/300) | 91.15% (10759/11803) | 71.00% (213/300) | 34.9 |

---

## Interpretation
1) Both models learn well on training data (TokenAcc ≈ 91%, EM ≈ 11–12%), but generalization remains weak (valid TokenAcc ≈ 64–66%, EM ≈ 0–0.33%).
2) Transformer achieves slightly higher valid TokenAcc, while LSTM shows stronger brace-balance stability under greedy decoding.
3) Beam search improves structure and sometimes nEM, but may reduce EM if it prefers shorter / generic sequences. Further tuning (alpha/min_len/coverage) is required.