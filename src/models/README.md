# Models Package - CNN-LSTM Architecture for HMER

This package contains the complete neural network architecture for Handwritten Mathematical Expression Recognition (HMER).

## 📁 Package Structure

```
models/
├── __init__.py          # Package initialization
├── encoder.py           # CNN encoders (ResNet18, DenseNet121)
├── attention.py         # Attention mechanisms
├── decoder.py           # LSTM decoder
├── hmer_model.py        # Complete end-to-end model
└── README.md           # This file
```

## 🏗️ Architecture Overview

### Complete Pipeline:

```
Input Image [B, 1, 256, 256]
         ↓
┌────────────────────────┐
│   CNN Encoder          │
│   (ResNet18/DenseNet)  │
└────────────────────────┘
         ↓
Feature Maps [B, 512, 8, 8]
         ↓
┌────────────────────────┐
│   Flatten to Sequence  │
│   [B, 64, 512]         │
└────────────────────────┘
         ↓
┌────────────────────────┐
│   LSTM Decoder         │
│   + Attention          │
└────────────────────────┘
         ↓
LaTeX Tokens [B, max_len, vocab_size]
```

## 🔧 Components

### 1. **encoder.py** - Visual Feature Extraction

**CNNEncoder (ResNet18-based)**
- Input: Grayscale images [B, 1, 256, 256]
- Output: Feature maps [B, 512, 8, 8]
- Pretrained on ImageNet (adapted for grayscale)
- Preserves spatial structure for attention

**CNNEncoderDenseNet (DenseNet121-based)**
- Alternative encoder with better parameter efficiency
- Output: Feature maps [B, 1024, 8, 8]

### 2. **attention.py** - Attention Mechanisms

**BahdanauAttention** (Default)
- Additive attention from "Neural Machine Translation" paper
- Used in original WAP model for HMER
- Computes alignment between decoder state and all encoder positions

**DotProductAttention**
- Scaled dot-product attention from Transformer
- Faster but may be less expressive than Bahdanau

**AdaptiveAttention**
- Includes coverage mechanism
- Prevents attending to same region repeatedly

### 3. **decoder.py** - Sequence Generation

**LSTMDecoder**
- Generates LaTeX tokens autoregressively
- Uses attention to focus on relevant image regions
- Implements teacher forcing for training
- Supports beam search for inference (future work)

Key Features:
- Token embedding layer
- LSTM cell with attention
- Output projection to vocabulary
- Dropout for regularization

### 4. **hmer_model.py** - Complete Model

**HMERModel**
- Combines encoder + decoder
- Handles training and inference
- Device management (CPU/GPU/MPS)

## 💻 Usage

### Quick Start

```python
from models import create_hmer_model

# Create model
model = create_hmer_model(
    vocab_size=346,
    embed_dim=256,
    decoder_dim=512,
    encoder_dim=512,
    attention_dim=512,
    encoder_type='resnet',      # 'resnet' or 'densenet'
    attention_type='bahdanau',  # 'bahdanau', 'dot_product', 'adaptive'
    pretrained=True,
    dropout=0.5,
    device='mps'                # 'cuda', 'mps', or 'cpu'
)

# Training
predictions, alphas = model(images, captions, caption_lengths)

# Inference
tokens, alphas = model.generate_latex(image, max_len=50)
```

### Custom Configuration

```python
from models import create_encoder, create_attention, LSTMDecoder, HMERModel

# Create components separately
encoder = create_encoder('resnet', pretrained=True)
attention = create_attention('bahdanau', encoder_dim=512, decoder_dim=512)
decoder = LSTMDecoder(vocab_size=346, embed_dim=256, decoder_dim=512, 
                      encoder_dim=512, attention=attention)

# Combine
model = HMERModel(encoder, decoder, device='mps')
```

## 🧪 Testing

Each module includes standalone tests:

```bash
# Test encoder
python models/encoder.py

# Test attention
python models/attention.py

# Test decoder
python models/decoder.py

# Test complete model
python models/hmer_model.py
```

## 📊 Model Statistics

### Default Configuration (ResNet18 + LSTM)

```
Total parameters: ~17.8M
- CNN Encoder: ~11.2M
- LSTM Decoder: ~6.6M

Model size: ~71 MB (fp32)
```

### Memory Requirements

**Training (batch_size=32)**
- GPU Memory: ~4-6 GB
- Training time: ~2-3 hours/epoch on M1/M2 Mac

**Inference**
- Single image: ~100-200ms on CPU
- Single image: ~20-50ms on GPU/MPS

## 🎯 Design Decisions

### Why ResNet18?
- Proven performance on image recognition
- Pretrained weights help with limited data
- Balance between accuracy and speed
- 8×8 feature maps preserve spatial information

### Why Bahdanau Attention?
- Standard in seq2seq models
- Used in original HMER papers (WAP)
- More interpretable than dot-product
- Better for structured spatial data

### Why LSTM?
- Handles variable-length sequences
- Good at capturing sequential dependencies
- Proven effective for LaTeX generation
- Simpler than Transformer for this task

## 🔄 Future Improvements

- [ ] Transformer decoder option
- [ ] Beam search decoding
- [ ] Coverage mechanism (already in AdaptiveAttention)
- [ ] Multi-head attention
- [ ] Positional encoding for spatial features

## 📚 References

1. **Watch, Attend and Parse** (Zhang et al., 2017)
   - Foundation for encoder-decoder HMER
   
2. **Bahdanau Attention** (2015)
   - "Neural Machine Translation by Jointly Learning to Align and Translate"

3. **Deep Residual Learning** (He et al., 2016)
   - ResNet architecture

## ⚙️ Hyperparameters

Recommended values based on MathWriting dataset:

```python
VOCAB_SIZE = 346          # From vocabulary analysis
EMBED_DIM = 256          # Token embedding size
DECODER_DIM = 512        # LSTM hidden size
ENCODER_DIM = 512        # ResNet18 output
ATTENTION_DIM = 512      # Attention hidden size
DROPOUT = 0.5            # Regularization
MAX_SEQ_LEN = 46         # 99th percentile + SOS/EOS
```

## 🐛 Troubleshooting

**Out of Memory Error**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

**Poor Attention Alignment**
- Visualize attention weights
- Try AdaptiveAttention with coverage
- Adjust attention_dim

**Slow Training**
- Use MPS on Mac (2-3x faster than CPU)
- Reduce image size (but affects accuracy)
- Use smaller encoder (MobileNet)

---

**Created for**: HWU Final Year Project - Deep Learning-Based Handwritten Mathematical Expression Solver  
**Author**: Muhammad Saad Muhammad Noman  
**Date**: February 2026