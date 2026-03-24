"""
Attention Mechanism for HMER
============================
Implements the attention mechanism that allows the decoder to focus
on relevant parts of the encoded image features at each decoding step.

This is crucial for HMER because different symbols appear in different
spatial locations in the image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.
    
    Also known as "Additive Attention" or "Concat Attention".
    Used in the original "Watch, Attend and Parse" (WAP) model.
    
    Args:
        encoder_dim (int): Dimension of encoder output features (512 for ResNet)
        decoder_dim (int): Dimension of decoder hidden state
        attention_dim (int): Dimension of attention hidden layer
    
    Reference:
        Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        
        # Linear layers for attention score computation
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden):
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_out (torch.Tensor): Encoded features [B, num_pixels, encoder_dim]
                                        where num_pixels = H * W (e.g., 8*8=64)
            decoder_hidden (torch.Tensor): Decoder hidden state [B, decoder_dim]
        
        Returns:
            context (torch.Tensor): Context vector [B, encoder_dim]
            alpha (torch.Tensor): Attention weights [B, num_pixels]
        """
        # Transform encoder output
        att1 = self.encoder_att(encoder_out)  # [B, num_pixels, attention_dim]
        
        # Transform decoder hidden state
        att2 = self.decoder_att(decoder_hidden)  # [B, attention_dim]
        
        # Add and apply non-linearity
        # att2 needs to be broadcast to all pixels
        att2 = att2.unsqueeze(1)  # [B, 1, attention_dim]
        att = self.full_att(self.relu(att1 + att2))  # [B, num_pixels, 1]
        att = att.squeeze(2)  # [B, num_pixels]
        
        # Compute attention weights
        alpha = self.softmax(att)  # [B, num_pixels]
        
        # Compute context vector (weighted sum of encoder outputs)
        # alpha: [B, num_pixels] → [B, num_pixels, 1]
        # encoder_out: [B, num_pixels, encoder_dim]
        # context: [B, encoder_dim]
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        
        return context, alpha


class DotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Simpler and faster than Bahdanau attention.
    Used in Transformer architectures.
    
    Args:
        encoder_dim (int): Dimension of encoder output features
        decoder_dim (int): Dimension of decoder hidden state
    """
    
    def __init__(self, encoder_dim, decoder_dim):
        super(DotProductAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # Project decoder hidden to same dimension as encoder
        self.query_proj = nn.Linear(decoder_dim, encoder_dim)
        self.softmax = nn.Softmax(dim=1)
        
        # Scaling factor for numerical stability
        self.scale = encoder_dim ** -0.5
    
    def forward(self, encoder_out, decoder_hidden):
        """
        Compute attention using scaled dot-product.
        
        Args:
            encoder_out (torch.Tensor): [B, num_pixels, encoder_dim]
            decoder_hidden (torch.Tensor): [B, decoder_dim]
        
        Returns:
            context (torch.Tensor): [B, encoder_dim]
            alpha (torch.Tensor): [B, num_pixels]
        """
        # Project decoder hidden to encoder dimension
        query = self.query_proj(decoder_hidden)  # [B, encoder_dim]
        query = query.unsqueeze(2)  # [B, encoder_dim, 1]
        
        # Compute attention scores
        # encoder_out: [B, num_pixels, encoder_dim]
        # query: [B, encoder_dim, 1]
        scores = torch.bmm(encoder_out, query)  # [B, num_pixels, 1]
        scores = scores.squeeze(2) * self.scale  # [B, num_pixels]
        
        # Apply softmax
        alpha = self.softmax(scores)  # [B, num_pixels]
        
        # Compute context
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [B, encoder_dim]
        
        return context, alpha


class AdaptiveAttention(nn.Module):
    """
    Adaptive Attention with coverage mechanism.
    
    Helps prevent the model from attending to the same region repeatedly.
    Maintains a coverage vector that accumulates attention weights over time.
    
    Args:
        encoder_dim (int): Dimension of encoder output
        decoder_dim (int): Dimension of decoder hidden state
        attention_dim (int): Dimension of attention hidden layer
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(AdaptiveAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.coverage_att = nn.Linear(1, attention_dim)  # Coverage input
        self.full_att = nn.Linear(attention_dim, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden, coverage=None):
        """
        Compute attention with coverage.
        
        Args:
            encoder_out (torch.Tensor): [B, num_pixels, encoder_dim]
            decoder_hidden (torch.Tensor): [B, decoder_dim]
            coverage (torch.Tensor, optional): [B, num_pixels]
                                               Accumulated attention from previous steps
        
        Returns:
            context (torch.Tensor): [B, encoder_dim]
            alpha (torch.Tensor): [B, num_pixels]
            coverage (torch.Tensor): [B, num_pixels] - Updated coverage
        """
        batch_size, num_pixels, _ = encoder_out.size()
        
        # Initialize coverage if None
        if coverage is None:
            coverage = torch.zeros(batch_size, num_pixels).to(encoder_out.device)
        
        # Transform inputs
        att1 = self.encoder_att(encoder_out)  # [B, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # [B, 1, attention_dim]
        
        # Add coverage component
        coverage_input = coverage.unsqueeze(2)  # [B, num_pixels, 1]
        att3 = self.coverage_att(coverage_input)  # [B, num_pixels, attention_dim]
        
        # Combine and compute attention scores
        att = self.full_att(self.relu(att1 + att2 + att3))  # [B, num_pixels, 1]
        att = att.squeeze(2)  # [B, num_pixels]
        
        # Attention weights
        alpha = self.softmax(att)  # [B, num_pixels]
        
        # Update coverage
        coverage = coverage + alpha
        
        # Context vector
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [B, encoder_dim]
        
        return context, alpha, coverage


# === HELPER FUNCTION ===

def create_attention(attention_type='bahdanau', encoder_dim=512, decoder_dim=512, attention_dim=512):
    """
    Factory function to create attention mechanism.
    
    Args:
        attention_type (str): 'bahdanau', 'dot_product', or 'adaptive'
        encoder_dim (int): Encoder output dimension
        decoder_dim (int): Decoder hidden dimension
        attention_dim (int): Attention layer dimension
    
    Returns:
        nn.Module: Attention mechanism
    """
    if attention_type.lower() == 'bahdanau':
        return BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
    elif attention_type.lower() == 'dot_product':
        return DotProductAttention(encoder_dim, decoder_dim)
    elif attention_type.lower() == 'adaptive':
        return AdaptiveAttention(encoder_dim, decoder_dim, attention_dim)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# === TESTING ===

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Testing Attention Mechanisms")
    print("="*70 + "\n")
    
    # Test parameters
    batch_size = 4
    num_pixels = 64  # 8x8 feature map
    encoder_dim = 512
    decoder_dim = 512
    attention_dim = 512
    
    # Create dummy inputs
    encoder_out = torch.randn(batch_size, num_pixels, encoder_dim)
    decoder_hidden = torch.randn(batch_size, decoder_dim)
    
    print("Input shapes:")
    print(f"  Encoder output: {encoder_out.shape}")
    print(f"  Decoder hidden: {decoder_hidden.shape}\n")
    
    # Test Bahdanau Attention
    print("1. Testing Bahdanau Attention:")
    bahdanau_att = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
    context, alpha = bahdanau_att(encoder_out, decoder_hidden)
    print(f"   Context shape: {context.shape}")
    print(f"   Alpha shape: {alpha.shape}")
    print(f"   Alpha sum: {alpha.sum(dim=1)[0]:.4f} (should be 1.0)")
    
    # Test Dot-Product Attention
    print("\n2. Testing Dot-Product Attention:")
    dot_att = DotProductAttention(encoder_dim, decoder_dim)
    context, alpha = dot_att(encoder_out, decoder_hidden)
    print(f"   Context shape: {context.shape}")
    print(f"   Alpha shape: {alpha.shape}")
    print(f"   Alpha sum: {alpha.sum(dim=1)[0]:.4f} (should be 1.0)")
    
    # Test Adaptive Attention
    print("\n3. Testing Adaptive Attention:")
    adaptive_att = AdaptiveAttention(encoder_dim, decoder_dim, attention_dim)
    context, alpha, coverage = adaptive_att(encoder_out, decoder_hidden)
    print(f"   Context shape: {context.shape}")
    print(f"   Alpha shape: {alpha.shape}")
    print(f"   Coverage shape: {coverage.shape}")
    print(f"   Alpha sum: {alpha.sum(dim=1)[0]:.4f} (should be 1.0)")
    
    print("\n" + "="*70)
    print("✅ All attention mechanisms tested successfully!")
    print("="*70)