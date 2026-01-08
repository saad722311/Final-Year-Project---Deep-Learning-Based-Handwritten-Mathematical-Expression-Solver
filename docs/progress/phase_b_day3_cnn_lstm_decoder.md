# Phase B — Day 3: CNN–LSTM Decoder + End-to-End Forward Pass

**Phase:** B  
**Day:** 3  
**Objective:** Implement an attention-based LSTM decoder and validate the first complete end-to-end handwritten mathematical expression recognition pipeline.

---

## Work completed

### 1) LSTM Decoder with Attention
**File:** `src/models/decoder_lstm.py`

- Implemented an LSTM-based sequence decoder with **additive (Bahdanau) attention**
- Decoder operates under **teacher forcing**
- At each decoding step:
  - attends over encoder feature sequence `(B, T, D)`
  - predicts next LaTeX token from vocabulary

---

### 2) End-to-End HMER Model
**File:** `src/models/hmer_model.py`

- Integrated:
  - CNN encoder (image → feature sequence)
  - LSTM decoder (feature sequence → LaTeX tokens)
- Forward pipeline:
images → encoder → memory → decoder → logits

---

### 3) Forward-Pass Smoke Test
**File:** `src/train/smoke_test_model_forward.py`

- Validated:
- dataset loading
- tokenizer integration
- encoder–decoder compatibility
- loss computation with padding ignored

**Command used:**
```bash
python3 -m src.train.smoke_test_model_forward

Outcome

✅ End-to-end forward pass executed successfully
✅ Logits produced with correct shape (B, L, V)
✅ Cross-entropy loss computed without errors
✅ Model handles variable-width images and variable-length labels

This confirms the baseline CNN–LSTM recognition model is structurally correct and ready for training.