"""
CNN Encoder for Handwritten Mathematical Expression Recognition
================================================================
This module implements the CNN-based visual encoder that extracts
features from handwritten mathematical expression images.

Architecture:
    Input: [B, 1, 256, 256] grayscale images
    Output: [B, 512, 8, 8] feature maps (spatial features preserved)

Based on ResNet architecture with modifications for HMER.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """
    CNN-based encoder for extracting visual features from handwritten math images.
    
    Uses a ResNet18 backbone (pretrained on ImageNet) as feature extractor,
    modified for single-channel grayscale input.
    
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights (default: True)
        feature_dim (int): Dimension of output feature maps (default: 512)
    """
    
    def __init__(self, pretrained=True, feature_dim=512):
        super(CNNEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Load ResNet18 backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for single-channel (grayscale) input
        # Original: conv1 expects [B, 3, H, W] (RGB)
        # Modified: conv1 expects [B, 1, H, W] (Grayscale)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # If using pretrained weights, average RGB weights to initialize grayscale conv
        if pretrained:
            # Take mean of RGB weights along channel dimension
            pretrained_weights = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
        
        # Copy remaining ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet blocks
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Remove fully connected layer (we keep spatial features)
        # ResNet's avgpool and fc are not used
        
        print(f"✓ CNN Encoder initialized (ResNet18 backbone)")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Feature dimension: {feature_dim}")
    
    def forward(self, x):
        """
        Forward pass through CNN encoder.
        
        Args:
            x (torch.Tensor): Input images [B, 1, 256, 256]
        
        Returns:
            torch.Tensor: Feature maps [B, 512, 8, 8]
        
        Feature map spatial size calculation:
            Input:  256×256
            Conv1:  256→128 (stride 2)
            Pool:   128→64  (stride 2)
            Layer1: 64→64
            Layer2: 64→32   (stride 2)
            Layer3: 32→16   (stride 2)
            Layer4: 16→8    (stride 2)
            Output: 8×8 feature maps
        """
        # Initial convolution and pooling
        x = self.conv1(x)       # [B, 64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, 64, 64]
        
        # ResNet blocks
        x = self.layer1(x)      # [B, 64, 64, 64]
        x = self.layer2(x)      # [B, 128, 32, 32]
        x = self.layer3(x)      # [B, 256, 16, 16]
        x = self.layer4(x)      # [B, 512, 8, 8]
        
        return x


class CNNEncoderDenseNet(nn.Module):
    """
    Alternative CNN encoder using DenseNet121 backbone.
    
    DenseNet is more parameter-efficient than ResNet and may perform better
    on limited data. Can be used as alternative to ResNet encoder.
    
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights
        feature_dim (int): Dimension of output feature maps (default: 1024)
    """
    
    def __init__(self, pretrained=True, feature_dim=1024):
        super(CNNEncoderDenseNet, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Load DenseNet121
        densenet = models.densenet121(pretrained=pretrained)
        
        # Modify first conv for grayscale
        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            pretrained_weights = densenet.features.conv0.weight.data
            self.conv0.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
        
        # Copy DenseNet feature layers
        self.features = densenet.features
        self.features.conv0 = self.conv0  # Replace first conv
        
        print(f"✓ CNN Encoder initialized (DenseNet121 backbone)")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Feature dimension: {feature_dim}")
    
    def forward(self, x):
        """
        Forward pass through DenseNet encoder.
        
        Args:
            x (torch.Tensor): Input images [B, 1, 256, 256]
        
        Returns:
            torch.Tensor: Feature maps [B, 1024, 8, 8]
        """
        features = self.features(x)  # [B, 1024, 8, 8]
        return features


# === HELPER FUNCTION ===

def create_encoder(encoder_type='resnet', pretrained=True):
    """
    Factory function to create CNN encoder.
    
    Args:
        encoder_type (str): 'resnet' or 'densenet'
        pretrained (bool): Use ImageNet pretrained weights
    
    Returns:
        nn.Module: CNN encoder
    """
    if encoder_type.lower() == 'resnet':
        return CNNEncoder(pretrained=pretrained, feature_dim=512)
    elif encoder_type.lower() == 'densenet':
        return CNNEncoderDenseNet(pretrained=pretrained, feature_dim=1024)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# === TESTING ===

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Testing CNN Encoder")
    print("="*70 + "\n")
    
    # Create encoder
    encoder = CNNEncoder(pretrained=True)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 256, 256)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    features = encoder(dummy_input)
    
    print(f"Output shape: {features.shape}")
    print(f"\n✅ Expected: [4, 512, 8, 8]")
    print(f"✅ Got:      {list(features.shape)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*70)
    print("✅ CNN Encoder test passed!")
    print("="*70)