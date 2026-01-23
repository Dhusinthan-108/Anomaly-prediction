"""
Complete Autoencoder for Anomaly Detection
Combines feature extraction, temporal encoding, and reconstruction
"""
import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .temporal_encoder import TemporalEncoder

class Decoder(nn.Module):
    """
    Decoder network with skip connections
    Reconstructs features from encoded representation
    """
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
        )
        
    def forward(self, x):
        """
        Args:
            x: Encoded tensor (batch_size, seq_len, hidden_dim)
        Returns:
            reconstructed: Reconstructed features (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape for processing
        x = x.reshape(batch_size * seq_len, hidden_dim)
        
        # Decode
        reconstructed = self.decoder(x)
        
        # Reshape back
        reconstructed = reconstructed.reshape(batch_size, seq_len, -1)
        
        return reconstructed


class AnomalyAutoencoder(nn.Module):
    """
    Complete Autoencoder for Video Anomaly Detection
    Architecture: FeatureExtractor -> TemporalEncoder -> Decoder
    """
    def __init__(self, config):
        super(AnomalyAutoencoder, self).__init__()
        
        self.config = config
        
        # Feature extractor (EfficientNet-B0)
        self.feature_extractor = FeatureExtractor(
            pretrained=True,
            freeze_backbone=True
        )
        
        # Temporal encoder (Bidirectional ConvLSTM)
        self.temporal_encoder = TemporalEncoder(
            input_dim=config['feature_dim'],
            hidden_dim=config['lstm_hidden_dim'],
            num_layers=config['lstm_layers'],
            dropout=config['dropout']
        )
        
        # Decoder
        self.decoder = Decoder(
            hidden_dim=config['lstm_hidden_dim'],
            output_dim=config['feature_dim']
        )
        
    def extract_features(self, frames):
        """
        Extract features from video frames
        Args:
            frames: Tensor of shape (batch_size, seq_len, C, H, W)
        Returns:
            features: Tensor of shape (batch_size, seq_len, feature_dim)
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        # Reshape for feature extraction
        frames = frames.reshape(batch_size * seq_len, C, H, W)
        
        # Extract features
        with torch.set_grad_enabled(self.training):
            features = self.feature_extractor(frames)  # (B*T, feature_dim)
        
        # Reshape back
        features = features.reshape(batch_size, seq_len, -1)
        
        return features
    
    def forward(self, frames):
        """
        Forward pass through the autoencoder
        Args:
            frames: Input video frames (batch_size, seq_len, C, H, W)
        Returns:
            reconstructed: Reconstructed features (batch_size, seq_len, feature_dim)
            encoded: Encoded representation (batch_size, seq_len, hidden_dim)
        """
        # Extract features
        features = self.extract_features(frames)  # (B, T, feature_dim)
        
        # Temporal encoding
        encoded = self.temporal_encoder(features)  # (B, T, hidden_dim)
        
        # Decode
        reconstructed = self.decoder(encoded)  # (B, T, feature_dim)
        
        return reconstructed, encoded, features
    
    def get_reconstruction_error(self, frames):
        """
        Compute reconstruction error for anomaly detection
        Args:
            frames: Input video frames (batch_size, seq_len, C, H, W)
        Returns:
            error: Reconstruction error per frame (batch_size, seq_len)
        """
        reconstructed, _, features = self.forward(frames)
        
        # Compute MSE per frame
        error = torch.mean((features - reconstructed) ** 2, dim=-1)  # (B, T)
        
        return error
    
    def unfreeze_backbone(self):
        """Unfreeze feature extractor for fine-tuning"""
        self.feature_extractor.unfreeze()


if __name__ == "__main__":
    # Test the autoencoder
    from config import MODEL_CONFIG
    
    model = AnomalyAutoencoder(MODEL_CONFIG)
    
    # Test input
    frames = torch.randn(2, 16, 3, 224, 224)  # (batch, seq_len, C, H, W)
    
    # Forward pass
    reconstructed, encoded, features = model(frames)
    
    print(f"Input shape: {frames.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test reconstruction error
    error = model.get_reconstruction_error(frames)
    print(f"Reconstruction error shape: {error.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
