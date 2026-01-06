
# Phase B — Day 1: Data Pipeline + Smoke Test (CROHME)

**Date:** (fill in today’s date)  
**Repo:** Final-Year-Project---Deep-Learning-Based-Handwritten-Mathematical-Expression-Solver  
**Phase:** B (Baseline recognition pipeline)

## Objective (today)
Build and validate the foundational data pipeline required for the baseline HMER model:
- Image → tensor pipeline (grayscale, fixed height)
- LaTeX → token IDs pipeline (char-level tokenizer)
- Dataset + batching (padding for variable widths and label lengths)
- Smoke test to confirm everything works before implementing CNN/LSTM/Transformer models

---

## Work completed

### 1) Folder + file scaffolding created
Created a structured Phase B codebase under `src/`:
- `src/data/` for tokenizer, transforms, dataset, collate
- `src/models/` reserved for CNN encoder + decoders
- `src/train/` reserved for training/eval/inference scripts
- `src/utils/` reserved for metrics/checkpoint helpers
- `configs/` reserved for YAML configs
- `results/` reserved for generated outputs (not committed)

---

### 2) Implemented char-level LaTeX tokenizer
**File:** `src/data/tokenizer.py`  
**What it does:**
- Builds a character vocabulary from CROHME training labels
- Encodes LaTeX strings into token IDs
- Decodes token IDs back into LaTeX
- Uses special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`

**Reason for char-level baseline:**
- Simple and robust (works for any LaTeX)
- Easy to debug and verify correctness

---

### 3) Implemented image preprocessing transforms
**File:** `src/data/transforms.py`  
**What it does:**
- Loads PNG as grayscale
- Resizes to fixed height (128) while preserving aspect ratio
- Converts to torch tensor and normalizes to roughly [-1, 1]
- Keeps variable width (to be padded later in collate)

---

### 4) Implemented CROHME processed dataset loader
**File:** `src/data/datasets.py`  
**What it does:**
- Loads MathWriting-style processed CROHME splits:
  - `processed/{split}_images/*.png`
  - `processed/{split}_labels.csv`
- Returns (per sample):
  - image tensor `(1, H, W)`
  - `input_ids` and `target_ids` (teacher forcing shift)
  - raw label string for debugging

Teacher forcing setup:
- `input_ids` = `<SOS> ...`
- `target_ids` = `... <EOS>`

---

### 5) Implemented batch collate with padding
**File:** `src/data/collate.py`  
**What it does:**
- Pads images in batch to the max width (Wmax)
- Pads label token sequences to max length (Lmax)
- Outputs a batch dictionary with:
  - `images: (B, 1, 128, Wmax)`
  - `input_ids: (B, Lmax)`
  - `target_ids: (B, Lmax)`
  - plus lengths + filenames

---

### 6) Smoke test created + executed successfully
**File:** `src/train/smoke_test_data.py`  
**Command used (IMPORTANT):**
```bash
python3 -m src.train.smoke_test_data

## Smoke Test Outcome

**Status:** ✅ SUCCESS

The data pipeline and tokenizer were validated successfully.

### Verified components
- Tokenizer vocabulary built correctly from CROHME training labels
- Images loaded and preprocessed correctly (grayscale, resized, normalized)
- Batch collation and padding for variable-width images worked as intended
- LaTeX encoding → decoding reproduced the original labels exactly

### Observed output snapshot
- **Vocabulary size:** 84
- **Images tensor shape:** `torch.Size([4, 1, 128, 612])`
- **Input token IDs shape:** `torch.Size([4, 72])`
- **Target token IDs shape:** `torch.Size([4, 72])`
- **Decoded LaTeX:** matched the original sample label with no loss or corruption

This confirms the data pipeline is stable and ready for CNN–LSTM and CNN–Transformer model implementation.