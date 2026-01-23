"""
Post-processing utilities for anomaly detection
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

class PostProcessor:
    """
    Post-process anomaly detection results
    """
    def __init__(self, smoothing_sigma=2.0, min_segment_length=3):
        """
        Args:
            smoothing_sigma: Sigma for Gaussian smoothing
            min_segment_length: Minimum length of anomaly segments
        """
        self.smoothing_sigma = smoothing_sigma
        self.min_segment_length = min_segment_length
    
    def gaussian_smoothing(self, scores):
        """
        Apply Gaussian smoothing to scores
        Args:
            scores: Anomaly scores (numpy array or tensor)
        Returns:
            smoothed: Smoothed scores
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        smoothed = gaussian_filter1d(scores, sigma=self.smoothing_sigma)
        
        return smoothed
    
    def remove_short_segments(self, anomalies, min_length=None):
        """
        Remove anomaly segments shorter than threshold
        Args:
            anomalies: Binary anomaly mask
            min_length: Minimum segment length (default: self.min_segment_length)
        Returns:
            filtered: Filtered anomaly mask
        """
        if min_length is None:
            min_length = self.min_segment_length
        
        if isinstance(anomalies, torch.Tensor):
            anomalies = anomalies.cpu().numpy()
        
        filtered = anomalies.copy()
        
        # Find segments
        in_segment = False
        start = 0
        
        for i in range(len(anomalies)):
            if anomalies[i] and not in_segment:
                start = i
                in_segment = True
            elif not anomalies[i] and in_segment:
                # End of segment
                if i - start < min_length:
                    # Remove short segment
                    filtered[start:i] = 0
                in_segment = False
        
        # Handle last segment
        if in_segment and len(anomalies) - start < min_length:
            filtered[start:] = 0
        
        return filtered
    
    def merge_close_segments(self, anomalies, max_gap=5):
        """
        Merge anomaly segments that are close together
        Args:
            anomalies: Binary anomaly mask
            max_gap: Maximum gap to merge
        Returns:
            merged: Merged anomaly mask
        """
        if isinstance(anomalies, torch.Tensor):
            anomalies = anomalies.cpu().numpy()
        
        merged = anomalies.copy()
        
        # Find gaps between anomalies
        in_gap = False
        gap_start = 0
        
        for i in range(len(anomalies)):
            if not anomalies[i] and i > 0 and anomalies[i-1]:
                # Start of gap
                gap_start = i
                in_gap = True
            elif anomalies[i] and in_gap:
                # End of gap
                gap_length = i - gap_start
                if gap_length <= max_gap:
                    # Merge by filling gap
                    merged[gap_start:i] = 1
                in_gap = False
        
        return merged
    
    def compute_segment_confidence(self, scores, segments):
        """
        Compute confidence for each anomaly segment
        Args:
            scores: Anomaly scores
            segments: List of (start, end) tuples
        Returns:
            confidences: List of confidence scores
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        confidences = []
        
        for start, end in segments:
            segment_scores = scores[start:end]
            confidence = np.mean(segment_scores)
            confidences.append(float(confidence))
        
        return confidences
    
    def rank_segments(self, scores, segments):
        """
        Rank anomaly segments by severity
        Args:
            scores: Anomaly scores
            segments: List of (start, end) tuples
        Returns:
            ranked_segments: List of (start, end, score) sorted by score
        """
        confidences = self.compute_segment_confidence(scores, segments)
        
        ranked = [(seg[0], seg[1], conf) 
                 for seg, conf in zip(segments, confidences)]
        ranked.sort(key=lambda x: x[2], reverse=True)
        
        return ranked
    
    def process(self, scores, anomalies):
        """
        Complete post-processing pipeline
        Args:
            scores: Anomaly scores
            anomalies: Binary anomaly mask
        Returns:
            processed_scores: Smoothed scores
            processed_anomalies: Filtered anomalies
        """
        # Smooth scores
        processed_scores = self.gaussian_smoothing(scores)
        
        # Remove short segments
        processed_anomalies = self.remove_short_segments(anomalies)
        
        # Merge close segments
        processed_anomalies = self.merge_close_segments(processed_anomalies)
        
        return processed_scores, processed_anomalies


if __name__ == "__main__":
    # Test post-processor
    processor = PostProcessor()
    
    # Generate dummy data
    scores = np.random.rand(100)
    anomalies = np.zeros(100)
    anomalies[20:25] = 1  # Short segment
    anomalies[30:45] = 1  # Long segment
    anomalies[50:52] = 1  # Very short segment
    anomalies[60:70] = 1  # Another long segment
    
    print(f"Original anomalies: {anomalies.sum()}")
    
    # Process
    smoothed_scores, filtered_anomalies = processor.process(scores, anomalies)
    
    print(f"Filtered anomalies: {filtered_anomalies.sum()}")
    print(f"Score smoothing applied: ✅")
