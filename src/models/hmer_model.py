"""
Complete HMER Model
===================
Combines CNN Encoder, Attention, and LSTM Decoder into a complete
end-to-end model for Handwritten Mathematical Expression Recognition.
"""

import torch
import torch.nn as nn


class HMERModel(nn.Module):
    """
    Complete HMER model: Image → LaTeX
    
    Architecture:
        Input: [B, 1, 256, 256] grayscale images
        ↓
        CNN Encoder (ResNet18)
        ↓
        Features: [B, 512, 8, 8]
        ↓
        LSTM Decoder with Attention
        ↓
        Output: LaTeX token sequences
    
    Args:
        encoder (nn.Module): CNN encoder
        decoder (nn.Module): LSTM decoder with attention
        device (str): 'cuda', 'mps', or 'cpu'
    """
    
    def __init__(self, encoder, decoder, device='cpu'):
        super(HMERModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Move model to device
        self.to(device)
        
        print(f"\n✓ Complete HMER Model initialized")
        print(f"  - Device: {device}")
        print(f"  - Encoder: {encoder.__class__.__name__}")
        print(f"  - Decoder: {decoder.__class__.__name__}")
    
    def forward(self, images, captions, caption_lengths, teacher_forcing_ratio=1.0):
        """
        Forward pass (training with optional scheduled sampling).
        
        Args:
            images (torch.Tensor): Input images [B, 1, 256, 256]
            captions (torch.Tensor): Ground truth token IDs [B, max_len]
            caption_lengths (torch.Tensor): Actual lengths [B]
            teacher_forcing_ratio (float): Probability of using ground truth (default: 1.0)
        
        Returns:
            predictions (torch.Tensor): [B, max_len, vocab_size]
            alphas (torch.Tensor): [B, max_len, num_pixels]
        """
        # Encode images
        encoder_out = self.encoder(images)  # [B, 512, 8, 8]
        
        # Decode to LaTeX (with scheduled sampling)
        predictions, alphas = self.decoder(
            encoder_out, captions, caption_lengths, 
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        return predictions, alphas
    
    def generate_latex(self, image, max_len=50, sos_token=1, eos_token=2, beam_width=1):
        """
        Generate LaTeX from a single image (inference).
        
        Args:
            image (torch.Tensor): Single image [1, 1, 256, 256]
            max_len (int): Maximum sequence length
            sos_token (int): Start token ID
            eos_token (int): End token ID
            beam_width (int): Beam width (1 = greedy, >1 = beam search)
        
        Returns:
            tokens (list): Generated token IDs
            score (float): Log probability (only for beam search)
        """
        self.eval()
        with torch.no_grad():
            # Encode image
            encoder_out = self.encoder(image)
            
            if beam_width == 1:
                # Greedy decoding
                tokens, alphas = self.decoder.generate(
                    encoder_out,
                    max_len=max_len,
                    sos_token=sos_token,
                    eos_token=eos_token
                )
                return tokens, None
            else:
                # Beam search
                tokens, score = self.decoder.beam_search(
                    encoder_out,
                    beam_width=beam_width,
                    max_len=max_len,
                    sos_token=sos_token,
                    eos_token=eos_token
                )
                return tokens, score
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_hmer_model(vocab_size, embed_dim=256, decoder_dim=512, 
                     encoder_dim=512, attention_dim=512,
                     encoder_type='resnet', attention_type='bahdanau',
                     pretrained=True, dropout=0.5, device='cpu'):
    """
    Factory function to create complete HMER model.
    
    Args:
        vocab_size (int): Size of vocabulary (346 for MathWriting)
        embed_dim (int): Token embedding dimension
        decoder_dim (int): LSTM hidden dimension
        encoder_dim (int): CNN encoder output dimension
        attention_dim (int): Attention hidden dimension
        encoder_type (str): 'resnet' or 'densenet'
        attention_type (str): 'bahdanau', 'dot_product', or 'adaptive'
        pretrained (bool): Use pretrained CNN encoder
        dropout (float): Dropout probability
        device (str): 'cuda', 'mps', or 'cpu'
    
    Returns:
        HMERModel: Complete model ready for training
    """
    # Import components
    import os
    import sys
    
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add models directory to path if not already there
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from encoder import create_encoder
    from attention import create_attention
    from decoder import LSTMDecoder
    
    # Create encoder
    encoder = create_encoder(encoder_type=encoder_type, pretrained=pretrained)
    
    # Get actual encoder dimension
    if encoder_type == 'densenet':
        encoder_dim = 1024
    else:
        encoder_dim = 512
    
    # Create attention
    attention = create_attention(
        attention_type=attention_type,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        attention_dim=attention_dim
    )
    
    # Create decoder
    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        encoder_dim=encoder_dim,
        attention=attention,
        dropout=dropout
    )
    
    # Create complete model
    model = HMERModel(encoder, decoder, device=device)
    
    return model