# Day 7 — Baseline Comparison (CNN+LSTM vs CNN+Transformer)

## Goal
Compare two baseline decoders on CROHME23 processed dataset:
- **CNN Encoder + LSTM Decoder (attention)**
- **CNN Encoder + Transformer Decoder**

Evaluation uses:
- Exact Match (EM)
- Normalized Exact Match (nEM) — ignores whitespace / surrounding `$...$`
- Teacher-forcing Token Accuracy (TokenAcc)

All results below are computed on **valid split**, first **300** samples (`--max_eval 300`).

---

## Experiment Setup

### Common Inference Settings
All models were evaluated with the same decoding constraints:

- `max_len = 256`
- `min_len = 10`
- `repetition_penalty = 1.15`
- `no_repeat_ngram_size = 3`
- `forbid_unk = True`

Decoding modes:
- **Greedy**
- **Beam search (beam_size=5, alpha=0.6)**

Outputs saved to:
- `results/compare/transformer_valid_greedy_300.txt`
- `results/compare/transformer_valid_beam5a06_300.txt`
- `results/compare/lstm_valid_greedy_300.txt`
- `results/compare/lstm_valid_beam5a06_300.txt`

---

## Training Observation (Overfitting)
During training, validation loss improved up to a certain epoch and then started increasing while training loss continued decreasing.

This indicates **overfitting**:
- the model keeps memorizing training data
- generalization to validation gets worse after the best epoch

**Action:** always evaluate using `best.pt` (lowest valid loss), and prefer early stopping / regularization.

---

## Results Summary (Valid split, first 300)

| Model | Decode | EM | nEM | TokenAcc |
|------|--------|----|-----|----------|
| CNN+Transformer | Greedy | 0.00% (0/300) | 0.00% (0/300) | 61.96% (5078/8195) |
| CNN+Transformer | Beam (5, α=0.6) | 0.33% (1/300) | 0.33% (1/300) | 61.96% (5078/8195) |
| CNN+LSTM | Greedy | 0.00% (0/300) | 0.33% (1/300) | 57.62% (4722/8195) |
| CNN+LSTM | Beam (5, α=0.6) | 0.00% (0/300) | 0.33% (1/300) | 57.62% (4722/8195) |

---

## Key Findings

### 1) Transformer > LSTM on Token Accuracy (so far)
Transformer token accuracy is higher on the same validation slice:
- Transformer: **61.96%**
- LSTM: **57.62%**

This suggests Transformer is learning better token-level patterns.

### 2) Exact Match remains extremely low for both
Even with beam decoding + constraints, EM is near 0.

Reason:
- EM requires the **entire LaTeX sequence** to be correct.
- One missing brace, wrong superscript, or token mismatch makes the whole expression incorrect.
- TokenAcc can improve while EM stays low, especially with long sequences.

### 3) Beam search helps slightly (Transformer)
Transformer beam produced:
- EM **0.33%** vs greedy **0.00%**
LSTM beam did not improve EM in this run.

---

## Conclusion (Day 7)
- The system is learning (loss decreases; token accuracy improves), but it does not yet generalize well enough to produce full-expression exact matches.
- Transformer baseline currently looks stronger than LSTM in token-level accuracy.
- Next work should focus on improving generalization + decoding stability to increase EM.

---

## Next Steps (Planned Improvements)

### A) Training improvements (generalization)
- Add regularization for Transformer: weight decay (e.g. 0.01) and higher dropout (e.g. 0.2)
- Use early stopping (stop near best epoch instead of training long after val loss rises)
- Consider LR scheduling (ReduceLROnPlateau / cosine decay)

### B) Better evaluation signals (beyond EM)
Track additional metrics more sensitive than EM:
- Edit distance / CER on token sequences
- Expression-level F1 on symbols
- Bracket-balance rate (how many predictions have valid LaTeX structure)

### C) Data / tokenization normalization
- Normalize spacing consistently
- Ensure stable tokenization for braces, commands, subscripts/superscripts
- Investigate common failure patterns (e.g., unmatched `}`)

### D) UI pipeline planning
Build an interface where:
1. User writes expression using stylus
2. Model predicts LaTeX
3. LaTeX is sent to an LLM for step-by-step solution/explanation
4. Show final rendered expression + steps

This will be implemented after improving baseline performance and stabilizing decoding outputs.