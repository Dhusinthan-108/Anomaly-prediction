"""
Anomaly Scorer
Multi-scale anomaly scoring with adaptive thresholding
"""
import torch
import torch.nn as nn
import numpy as np

class AnomalyScorer:
    """
    Computes anomaly scores from reconstruction errors
    Implements adaptive thresholding and temporal smoothing
    """
    def __init__(self, threshold=0.5, smoothing_window=5):
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.score_history = []
        
    def compute_reconstruction_loss(self, original, reconstructed):
        """
        Compute reconstruction loss (MSE)
        Args:
            original: Original features (batch_size, seq_len, feature_dim)
            reconstructed: Reconstructed features (batch_size, seq_len, feature_dim)
        Returns:
            loss: MSE loss per frame (batch_size, seq_len)
        """
        mse = torch.mean((original - reconstructed) ** 2, dim=-1)
        return mse
    
    def compute_temporal_consistency_loss(self, features):
        """
        Compute temporal consistency loss
        Penalizes sudden changes in features
        Args:
            features: Feature tensor (batch_size, seq_len, feature_dim)
        Returns:
            loss: Temporal consistency loss (batch_size, seq_len-1)
        """
        # Compute differences between consecutive frames
        diff = features[:, 1:, :] - features[:, :-1, :]
        temporal_loss = torch.mean(diff ** 2, dim=-1)
        return temporal_loss
    
    def compute_anomaly_score(self, original, reconstructed):
        """
        Compute final anomaly score
        Args:
            original: Original features (batch_size, seq_len, feature_dim)
            reconstructed: Reconstructed features (batch_size, seq_len, feature_dim)
        Returns:
            scores: Anomaly scores (batch_size, seq_len)
        """
        # Reconstruction error
        recon_error = self.compute_reconstruction_loss(original, reconstructed)
        
        # Temporal consistency (optional, can be weighted)
        # temporal_error = self.compute_temporal_consistency_loss(reconstructed)
        
        # Normalize scores to [0, 1]
        scores = recon_error
        
        return scores
    
    def temporal_smoothing(self, scores):
        """
        Apply temporal smoothing using moving average
        Args:
            scores: Anomaly scores (seq_len,) or (batch_size, seq_len)
        Returns:
            smoothed: Smoothed scores
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        if len(scores.shape) == 1:
            scores = scores[np.newaxis, :]
        
        batch_size, seq_len = scores.shape
        smoothed = np.zeros_like(scores)
        
        for b in range(batch_size):
            for i in range(seq_len):
                start = max(0, i - self.smoothing_window // 2)
                end = min(seq_len, i + self.smoothing_window // 2 + 1)
                smoothed[b, i] = np.mean(scores[b, start:end])
        
        return torch.from_numpy(smoothed).float()
    
    def adaptive_threshold(self, scores, percentile=95):
        """
        Compute adaptive threshold based on score distribution
        Args:
            scores: Anomaly scores (batch_size, seq_len)
            percentile: Percentile for threshold (default: 95)
        Returns:
            threshold: Adaptive threshold value
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        threshold = np.percentile(scores.flatten(), percentile)
        return threshold
    
    def detect_anomalies(self, scores, threshold=None):
        """
        Detect anomalies based on threshold
        Args:
            scores: Anomaly scores (batch_size, seq_len)
            threshold: Threshold value (if None, use adaptive)
        Returns:
            anomalies: Binary mask (batch_size, seq_len)
            threshold: Used threshold
        """
        if threshold is None:
            threshold = self.adaptive_threshold(scores)
        
        if isinstance(scores, torch.Tensor):
            anomalies = (scores > threshold).float()
        else:
            anomalies = (scores > threshold).astype(np.float32)
        
        return anomalies, threshold
    
    def get_anomaly_segments(self, anomalies):
        """
        Get continuous anomaly segments
        Args:
            anomalies: Binary anomaly mask (seq_len,)
        Returns:
            segments: List of (start, end) tuples
        """
        if isinstance(anomalies, torch.Tensor):
            anomalies = anomalies.cpu().numpy()
        
        segments = []
        in_segment = False
        start = 0
        
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly and not in_segment:
                start = i
                in_segment = True
            elif not is_anomaly and in_segment:
                segments.append((start, i))
                in_segment = False
        
        # Handle case where anomaly extends to end
        if in_segment:
            segments.append((start, len(anomalies)))
        
        return segments
    
    def compute_confidence(self, scores, threshold):
        """
        Compute confidence scores for anomalies
        Args:
            scores: Anomaly scores (batch_size, seq_len)
            threshold: Threshold value
        Returns:
            confidence: Confidence scores [0, 1]
        """
        # Normalize scores relative to threshold
        confidence = torch.clamp((scores - threshold) / (threshold + 1e-8), 0, 1)
        return confidence


if __name__ == "__main__":
    # Test the anomaly scorer
    scorer = AnomalyScorer(threshold=0.5, smoothing_window=5)
    
    # Generate dummy data
    original = torch.randn(2, 16, 1280)
    reconstructed = original + torch.randn(2, 16, 1280) * 0.1
    
    # Compute scores
    scores = scorer.compute_anomaly_score(original, reconstructed)
    print(f"Anomaly scores shape: {scores.shape}")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Smooth scores
    smoothed = scorer.temporal_smoothing(scores)
    print(f"Smoothed scores shape: {smoothed.shape}")
    
    # Detect anomalies
    anomalies, threshold = scorer.detect_anomalies(scores)
    print(f"Threshold: {threshold:.4f}")
    print(f"Anomalies detected: {anomalies.sum().item()}")
    
    # Get segments
    segments = scorer.get_anomaly_segments(anomalies[0])
    print(f"Anomaly segments: {segments}")
