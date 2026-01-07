# Phase B — Day 2: CNN Encoder (Image → Feature Sequence)

**Phase:** B  
**Day:** 2  
**Objective:** Implement and validate a CNN encoder that converts handwritten mathematical expression images into a sequential feature representation usable by both CNN–LSTM and CNN–Transformer decoders.

---

## Motivation

Handwritten mathematical expressions are variable-width images.  
Sequence decoders (LSTM / Transformer) require **sequential inputs** rather than raw pixel grids.

The CNN encoder bridges this gap by:
- extracting visual features from images
- compressing spatial information
- converting the image into a left-to-right feature sequence

---

## Encoder design

### Input
- Image tensor of shape:
(B, 1, 128, W)

Where:
- `B` = batch size
- `1` = grayscale channel
- `128` = fixed image height
- `W` = variable image width

---

### CNN architecture

The encoder consists of:
- stacked convolution blocks (`Conv → BatchNorm → ReLU`)
- progressive downsampling using max pooling
- aggressive height reduction
- moderate width reduction

This design preserves left-to-right structure while compressing vertical information.

---

### Feature sequence construction

After CNN processing, the feature map has shape:
(B, C, H’, W’)

The feature map is converted into a sequence by:
- treating each column along `W'` as a time step
- flattening `(C × H')` into a single feature vector
- projecting features to a fixed dimension (`d_model = 256`)

Final encoder output:
memory: (B, T, D)

Where:
- `T = W'` (time steps across width)
- `D = 256` (feature dimension)

This representation is shared by both LSTM and Transformer decoders.

---

## Implementation

**File created:**
src/models/cnn_encoder.py

Key characteristics:
- Works with variable-width images
- Produces consistent feature size per time step
- Decoder-agnostic (reusable backbone)

---

## Encoder smoke test

**File created:**
src/train/smoke_test_encoder.py

The smoke test:
- loads a CROHME training batch
- passes images through the CNN encoder
- prints tensor shapes for verification

---

## Observed output (validated)
Vocab size: 84
Input images: torch.Size([4, 1, 128, 847])
Encoder memory: torch.Size([4, 52, 256])

### Interpretation
- Batch size: `4`
- Input image width reduced from `847 → 52` time steps
- Each time step represented by a `256`-dimensional feature vector

This confirms:
- CNN downsampling is correct
- sequence length scales with image width
- encoder output matches expected `(B, T, D)` format

---

## Outcome

✅ CNN encoder implemented successfully  
✅ Encoder output verified via smoke test  
✅ Feature sequence suitable for both LSTM and Transformer decoders  

The model pipeline is now ready for sequence decoding.

---

## Next step (Day 3)

**Decoder #1: CNN–LSTM with attention**
- Token embedding
- LSTM decoder
- Attention over encoder memory
- Forward pass and loss computation

This will complete the first end-to-end recognition baseline.