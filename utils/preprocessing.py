"""
Video Preprocessing Utilities
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from config import MODEL_CONFIG, DATA_CONFIG

class VideoPreprocessor:
    """
    Preprocess videos for anomaly detection
    """
    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(DATA_CONFIG['normalize_mean'])
        self.std = np.array(DATA_CONFIG['normalize_std'])
    
    def load_video(self, video_path, max_frames=None):
        """
        Load video and extract frames
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
        Returns:
            frames: List of frames (H, W, C)
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        return frames
    
    def resize_frame(self, frame):
        """Resize frame to target size"""
        return cv2.resize(frame, self.target_size)
    
    def normalize_frame(self, frame):
        """
        Normalize frame with ImageNet statistics
        Args:
            frame: Frame as numpy array (H, W, C) in range [0, 255]
        Returns:
            normalized: Normalized frame in range [-1, 1]
        """
        # Convert to float [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Normalize
        frame = (frame - self.mean) / self.std
        
        return frame
    
    def preprocess_frame(self, frame):
        """
        Complete preprocessing pipeline for a single frame
        Args:
            frame: Input frame (H, W, C)
        Returns:
            processed: Preprocessed frame
        """
        # Resize
        frame = self.resize_frame(frame)
        
        # Normalize
        if self.normalize:
            frame = self.normalize_frame(frame)
        else:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def preprocess_frames(self, frames):
        """
        Preprocess a list of frames
        Args:
            frames: List of frames
        Returns:
            tensor: Tensor of shape (T, C, H, W)
        """
        processed = []
        
        for frame in frames:
            processed_frame = self.preprocess_frame(frame)
            processed.append(processed_frame)
        
        # Stack and convert to tensor
        processed = np.stack(processed, axis=0)  # (T, H, W, C)
        processed = torch.from_numpy(processed).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        
        return processed
    
    def create_sequences(self, frames, temporal_window=16, stride=8):
        """
        Create overlapping sequences from frames
        Args:
            frames: Tensor of frames (T, C, H, W)
            temporal_window: Length of each sequence
            stride: Stride between sequences
        Returns:
            sequences: List of sequences (temporal_window, C, H, W)
        """
        T = frames.shape[0]
        sequences = []
        
        for i in range(0, T - temporal_window + 1, stride):
            sequence = frames[i:i + temporal_window]
            sequences.append(sequence)
        
        return sequences
    
    def denormalize_frame(self, frame):
        """
        Denormalize frame for visualization
        Args:
            frame: Normalized frame (C, H, W) or (H, W, C)
        Returns:
            denormalized: Frame in range [0, 255]
        """
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        # Handle different shapes
        if frame.shape[0] == 3:  # (C, H, W)
            frame = frame.transpose(1, 2, 0)  # (H, W, C)
        
        # Denormalize
        frame = frame * self.std + self.mean
        
        # Clip and convert to uint8
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        
        return frame


if __name__ == "__main__":
    # Test preprocessor
    preprocessor = VideoPreprocessor()
    
    # Create dummy frames
    dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(32)]
    
    print(f"Input: {len(dummy_frames)} frames of shape {dummy_frames[0].shape}")
    
    # Preprocess
    processed = preprocessor.preprocess_frames(dummy_frames)
    print(f"Processed shape: {processed.shape}")
    print(f"Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Create sequences
    sequences = preprocessor.create_sequences(processed, temporal_window=16, stride=8)
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences[0].shape}")
    
    # Test denormalization
    denorm = preprocessor.denormalize_frame(processed[0])
    print(f"Denormalized shape: {denorm.shape}")
    print(f"Denormalized range: [{denorm.min()}, {denorm.max()}]")
