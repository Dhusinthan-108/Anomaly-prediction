"""
Data Augmentation for Video Sequences
"""
import numpy as np
import torch
import cv2

class VideoAugmentation:
    """
    Augmentation techniques for video sequences
    """
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of applying each augmentation
        """
        self.p = p
    
    def temporal_jitter(self, frames, max_jitter=2):
        """
        Randomly skip frames for temporal jittering
        Args:
            frames: Tensor (T, C, H, W)
            max_jitter: Maximum frames to skip
        Returns:
            jittered: Jittered frames
        """
        if np.random.rand() > self.p:
            return frames
        
        T = frames.shape[0]
        indices = []
        i = 0
        
        while i < T:
            indices.append(i)
            jitter = np.random.randint(1, max_jitter + 1)
            i += jitter
        
        # Ensure we have enough frames
        if len(indices) < T:
            # Repeat last frame
            indices += [indices[-1]] * (T - len(indices))
        else:
            indices = indices[:T]
        
        return frames[indices]
    
    def random_crop(self, frames, crop_ratio=0.9):
        """
        Random spatial crop with context preservation
        Args:
            frames: Tensor (T, C, H, W)
            crop_ratio: Ratio of crop size to original
        Returns:
            cropped: Cropped frames
        """
        if np.random.rand() > self.p:
            return frames
        
        T, C, H, W = frames.shape
        
        crop_h = int(H * crop_ratio)
        crop_w = int(W * crop_ratio)
        
        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)
        
        cropped = frames[:, :, top:top+crop_h, left:left+crop_w]
        
        # Resize back to original size
        cropped = torch.nn.functional.interpolate(
            cropped, size=(H, W), mode='bilinear', align_corners=False
        )
        
        return cropped
    
    def color_jitter(self, frames, brightness=0.2, contrast=0.2):
        """
        Random color augmentation
        Args:
            frames: Tensor (T, C, H, W)
            brightness: Brightness variation
            contrast: Contrast variation
        Returns:
            augmented: Color-augmented frames
        """
        if np.random.rand() > self.p:
            return frames
        
        # Brightness
        brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
        frames = frames * brightness_factor
        
        # Contrast
        contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
        mean = frames.mean(dim=(2, 3), keepdim=True)
        frames = (frames - mean) * contrast_factor + mean
        
        return frames
    
    def gaussian_noise(self, frames, std=0.01):
        """
        Add Gaussian noise
        Args:
            frames: Tensor (T, C, H, W)
            std: Standard deviation of noise
        Returns:
            noisy: Noisy frames
        """
        if np.random.rand() > self.p:
            return frames
        
        noise = torch.randn_like(frames) * std
        return frames + noise
    
    def horizontal_flip(self, frames):
        """
        Random horizontal flip
        Args:
            frames: Tensor (T, C, H, W)
        Returns:
            flipped: Flipped frames
        """
        if np.random.rand() > self.p:
            return frames
        
        return torch.flip(frames, dims=[3])
    
    def speed_variation(self, frames, speed_range=(0.8, 1.2)):
        """
        Simulate speed variation (slow-mo/fast-forward)
        Args:
            frames: Tensor (T, C, H, W)
            speed_range: Range of speed factors
        Returns:
            varied: Speed-varied frames
        """
        if np.random.rand() > self.p:
            return frames
        
        T = frames.shape[0]
        speed_factor = np.random.uniform(*speed_range)
        
        # Create new indices
        new_T = int(T / speed_factor)
        indices = np.linspace(0, T - 1, new_T).astype(int)
        
        # Sample frames
        varied = frames[indices]
        
        # Pad or truncate to original length
        if varied.shape[0] < T:
            # Repeat last frame
            padding = frames[-1:].repeat(T - varied.shape[0], 1, 1, 1)
            varied = torch.cat([varied, padding], dim=0)
        else:
            varied = varied[:T]
        
        return varied
    
    def __call__(self, frames):
        """
        Apply all augmentations
        Args:
            frames: Tensor (T, C, H, W)
        Returns:
            augmented: Augmented frames
        """
        frames = self.temporal_jitter(frames)
        frames = self.random_crop(frames)
        frames = self.color_jitter(frames)
        frames = self.gaussian_noise(frames)
        frames = self.horizontal_flip(frames)
        # frames = self.speed_variation(frames)  # Optional, can cause issues
        
        return frames


if __name__ == "__main__":
    # Test augmentation
    augmentation = VideoAugmentation(p=0.8)
    
    # Create dummy frames
    frames = torch.randn(16, 3, 224, 224)
    
    print(f"Original shape: {frames.shape}")
    print(f"Original range: [{frames.min():.3f}, {frames.max():.3f}]")
    
    # Apply augmentation
    augmented = augmentation(frames)
    
    print(f"Augmented shape: {augmented.shape}")
    print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
