# Day 5 — Masked Encoder→Decoder Attention + Inference Metrics

## Goal
Wire padding/masking end-to-end for variable-width images, then verify inference works and report basic evaluation metrics.

## What I implemented

### 1) Collate: pad image width + return true widths
- Images are padded to the max width in the batch.
- Added `image_widths` (true width before padding) to the batch dict.
- This enables correct masking later in the encoder/attention.

Files:
- `src/data/collate.py`

### 2) CNN Encoder: return memory + memory_mask
- Encoder outputs:
  - `memory`: (B, T, D)
  - `memory_mask`: (B, T) boolean mask where True=valid timestep, False=padding
- `memory_mask` is computed from `image_widths` and the CNN pooling schedule.

Files:
- `src/models/cnn_encoder.py`

### 3) Model forward/generate: pass widths → encoder mask → decoder attention
- Updated `HMERModel.forward()` to call encoder with `image_widths` and pass `memory_mask` into decoder.
- Updated `HMERModel.generate()` similarly.

Files:
- `src/models/hmer_model.py`

### 4) Decoder: attention masking + faster teacher-forcing forward
- Attention uses `memory_mask` to block padded timesteps.
- Teacher-forcing forward was vectorized (no python loop over sequence length), improving training speed.

Files:
- `src/models/decoder_lstm.py`

### 5) Inference script: smart checkpoint loading + metrics
Implemented a single clean checkpoint loader:
- Loads dict checkpoint
- Uses `"model_state"` if present, otherwise falls back to `"model"` or raw state_dict

Added evaluation metrics:
- Exact-match (EM): strict string equality
- Normalized exact-match (nEM): removes `$...$` wrappers and whitespace
- Token accuracy (teacher forcing): % token match ignoring PAD

Files:
- `src/train/infer.py`

## Results (CROHME processed)

Config used: `crohme_cnn_lstm_dev_512`  
Evaluation size: first 300 samples (train + valid)

### Valid
- Exact-match (valid): **0.00%** (0/300)
- Norm exact-match (valid): **0.00%** (0/300)
- Token accuracy (valid): **56.25%** (4610/8195)

Output saved:
- `results/crohme_cnn_lstm_dev_512/valid_samples.txt`

### Train
- Exact-match (train): **0.00%** (0/300)
- Norm exact-match (train): **0.00%** (0/300)
- Token accuracy (train): **72.86%** (8600/11803)

Output saved:
- `results/crohme_cnn_lstm_dev_512/train_samples.txt`

## Interpretation
- Token accuracy indicates the model is learning meaningful local token patterns (well above random baseline).
- However, expression-level exact match remains 0% at this stage, which is expected for early sequence models: small token mistakes break full-sequence equality.
- Next steps will focus on stronger decoding, better training objectives, and improving sequence-level correctness.

## Next steps (Day 6+)
1. Add attention regularization / better decoding (beam search).
2. Move to a stronger encoder (ResNet-style) and/or Transformer decoder.
3. Add label smoothing + scheduled sampling.
4. Use sequence-level metrics like edit distance and CER/WER-style measures.
5. Train longer on full width (max_width=1024) and add LR scheduling.