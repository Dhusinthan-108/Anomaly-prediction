"""
Feature Extractor using EfficientNet-B0
Pre-trained on ImageNet for robust feature extraction
"""
import torch
import torch.nn as nn
import timm

class FeatureExtractor(nn.Module):
    """
    EfficientNet-B0 based feature extractor
    Outputs 1280-dimensional feature vectors per frame
    """
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(FeatureExtractor, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Freeze backbone for faster training
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Adaptive pooling to get fixed-size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension
        self.feature_dim = 1280
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            features: Feature tensor of shape (batch_size, feature_dim)
        """
        # Extract features
        features = self.backbone(x)  # (B, 1280, H', W')
        
        # Global average pooling
        features = self.adaptive_pool(features)  # (B, 1280, 1, 1)
        features = features.flatten(1)  # (B, 1280)
        
        return features
    
    def unfreeze(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test the feature extractor
    model = FeatureExtractor(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    features = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {model.feature_dim}")
