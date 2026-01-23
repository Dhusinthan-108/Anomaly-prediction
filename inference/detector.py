"""
Anomaly Detection Engine
Real-time inference on video sequences
"""
import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

from config import DEVICE, MODEL_CONFIG, INFERENCE_CONFIG
from models import AnomalyAutoencoder, AnomalyScorer
from utils.preprocessing import VideoPreprocessor

class AnomalyDetector:
    """
    Real-time anomaly detection engine
    """
    def __init__(self, model_path=None, config=None):
        """
        Args:
            model_path: Path to trained model checkpoint
            config: Inference configuration
        """
        self.config = config or INFERENCE_CONFIG
        self.device = DEVICE
        
        # Initialize model
        self.model = AnomalyAutoencoder(MODEL_CONFIG).to(self.device)
        
        # Load checkpoint if provided
        if model_path:
            self.load_checkpoint(model_path)
        
        self.model.eval()
        
        # Initialize scorer
        self.scorer = AnomalyScorer(
            threshold=self.config['threshold'],
            smoothing_window=self.config.get('smoothing_window', 5)
        )
        
        # Preprocessor
        self.preprocessor = VideoPreprocessor()
        
        print(f"✅ Anomaly Detector initialized")
        print(f"   Device: {self.device}")
        print(f"   Threshold: {self.config['threshold']}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        
        if 'best_val_loss' in checkpoint:
            print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    def detect_video(self, video_path, return_details=False):
        """
        Detect anomalies in a video file
        Args:
            video_path: Path to video file
            return_details: Whether to return detailed results
        Returns:
            results: Dictionary with detection results
        """
        print(f"\n🎬 Processing video: {video_path}")
        
        # Load video
        frames = self.preprocessor.load_video(video_path)
        print(f"   Loaded {len(frames)} frames")
        
        # Preprocess
        frames_tensor = self.preprocessor.preprocess_frames(frames)
        
        # Create sequences
        sequences = self.preprocessor.create_sequences(
            frames_tensor,
            temporal_window=MODEL_CONFIG['temporal_window'],
            stride=MODEL_CONFIG['temporal_window'] // 2
        )
        
        print(f"   Created {len(sequences)} sequences")
        
        # Detect anomalies
        all_scores = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), self.config['batch_size']),
                         desc="Detecting anomalies"):
                batch = sequences[i:i + self.config['batch_size']]
                batch = torch.stack(batch).to(self.device)
                
                # Get reconstruction errors
                errors = self.model.get_reconstruction_error(batch)
                all_scores.append(errors.cpu())
        
        # Combine scores
        all_scores = torch.cat(all_scores, dim=0)  # (num_sequences, temporal_window)
        
        # Average overlapping predictions
        frame_scores = self._aggregate_scores(all_scores, len(frames))
        
        # Smooth scores
        smoothed_scores = self.scorer.temporal_smoothing(frame_scores)
        
        # Detect anomalies
        anomalies, threshold = self.scorer.detect_anomalies(
            smoothed_scores,
            threshold=self.config.get('threshold')
        )
        
        # Get anomaly segments
        segments = self.scorer.get_anomaly_segments(anomalies.squeeze())
        
        # Compute confidence
        confidence = self.scorer.compute_confidence(smoothed_scores, threshold)
        
        # Results
        results = {
            'num_frames': len(frames),
            'num_anomalies': int(anomalies.sum().item()),
            'anomaly_ratio': float(anomalies.sum().item() / len(frames)),
            'threshold': float(threshold),
            'segments': segments,
            'max_score': float(smoothed_scores.max().item()),
            'mean_score': float(smoothed_scores.mean().item()),
        }
        
        if return_details:
            results['scores'] = smoothed_scores.squeeze().numpy()
            results['anomalies'] = anomalies.squeeze().numpy()
            results['confidence'] = confidence.squeeze().numpy()
            results['frames'] = frames
        
        print(f"\n📊 Detection Results:")
        print(f"   Total frames: {results['num_frames']}")
        print(f"   Anomalies detected: {results['num_anomalies']}")
        print(f"   Anomaly ratio: {results['anomaly_ratio']:.2%}")
        print(f"   Threshold: {results['threshold']:.4f}")
        print(f"   Anomaly segments: {len(segments)}")
        
        return results
    
    def _aggregate_scores(self, sequence_scores, num_frames):
        """
        Aggregate overlapping sequence scores to frame-level scores
        Args:
            sequence_scores: Scores from sequences (num_sequences, temporal_window)
            num_frames: Total number of frames
        Returns:
            frame_scores: Frame-level scores (num_frames,)
        """
        stride = MODEL_CONFIG['temporal_window'] // 2
        temporal_window = MODEL_CONFIG['temporal_window']
        
        frame_scores = torch.zeros(num_frames)
        frame_counts = torch.zeros(num_frames)
        
        for seq_idx, seq_scores in enumerate(sequence_scores):
            start_frame = seq_idx * stride
            end_frame = min(start_frame + temporal_window, num_frames)
            
            for i, score in enumerate(seq_scores):
                frame_idx = start_frame + i
                if frame_idx < num_frames:
                    frame_scores[frame_idx] += score
                    frame_counts[frame_idx] += 1
        
        # Average scores
        frame_scores = frame_scores / (frame_counts + 1e-8)
        
        return frame_scores
    
    def detect_frames(self, frames_tensor):
        """
        Detect anomalies in preprocessed frames
        Args:
            frames_tensor: Tensor of frames (num_frames, C, H, W)
        Returns:
            scores: Anomaly scores per frame
        """
        sequences = self.preprocessor.create_sequences(
            frames_tensor,
            temporal_window=MODEL_CONFIG['temporal_window'],
            stride=MODEL_CONFIG['temporal_window'] // 2
        )
        
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), self.config['batch_size']):
                batch = sequences[i:i + self.config['batch_size']]
                batch = torch.stack(batch).to(self.device)
                
                errors = self.model.get_reconstruction_error(batch)
                all_scores.append(errors.cpu())
        
        all_scores = torch.cat(all_scores, dim=0)
        frame_scores = self._aggregate_scores(all_scores, len(frames_tensor))
        
        return frame_scores


if __name__ == "__main__":
    # Test detector
    detector = AnomalyDetector()
    print("✅ Detector initialized successfully")
