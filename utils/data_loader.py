"""
Data Loader for UCSD Pedestrian Dataset
Auto-download and preprocessing
"""
import os
import urllib.request
import tarfile
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from tqdm import tqdm
from config import DATA_DIR, MODEL_CONFIG, DATA_CONFIG

def download_dataset(url=None, extract_dir=None):
    """
    Download and extract UCSD Pedestrian dataset
    Args:
        url: Dataset URL (default from config)
        extract_dir: Directory to extract to (default: DATA_DIR)
    """
    if url is None:
        url = DATA_CONFIG['ucsd_ped1_url']
    if extract_dir is None:
        extract_dir = DATA_DIR
    
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if (extract_dir / 'UCSD_Anomaly_Dataset.v1p2').exists():
        print("✅ Dataset already downloaded!")
        return str(extract_dir / 'UCSD_Anomaly_Dataset.v1p2')
    
    # Download
    filename = extract_dir / 'ucsd_dataset.tar.gz'
    
    print(f"📥 Downloading UCSD Pedestrian Dataset...")
    print(f"URL: {url}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end='')
    
    try:
        urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
        print("\n✅ Download complete!")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("📝 Please download manually from:")
        print("   http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm")
        return None
    
    # Extract
    print(f"📦 Extracting dataset...")
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("✅ Extraction complete!")
        
        # Remove tar file
        os.remove(filename)
        
        return str(extract_dir / 'UCSD_Anomaly_Dataset.v1p2')
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None


class UCSDDataset(Dataset):
    """
    UCSD Pedestrian Dataset for Anomaly Detection
    """
    def __init__(self, root_dir, subset='Train', temporal_window=16, transform=None):
        """
        Args:
            root_dir: Root directory of UCSD dataset
            subset: 'Train' or 'Test'
            temporal_window: Number of frames per sequence
            transform: Optional transform to apply
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.temporal_window = temporal_window
        self.transform = transform
        
        # Get video directories
        if subset == 'Train':
            self.video_dir = self.root_dir / 'UCSDped1' / 'Train'
        else:
            self.video_dir = self.root_dir / 'UCSDped1' / 'Test'
        
        # Load all video frames
        self.sequences = self._load_sequences()
        
        print(f"✅ Loaded {len(self.sequences)} sequences from {subset} set")
    
    def _load_sequences(self):
        """Load all video sequences"""
        sequences = []
        
        # Get all video folders
        video_folders = sorted([f for f in self.video_dir.iterdir() if f.is_dir()])
        
        for video_folder in tqdm(video_folders, desc=f"Loading {self.subset} videos"):
            # Get all frames in this video
            frame_files = sorted(list(video_folder.glob('*.tif')))
            
            if len(frame_files) == 0:
                continue
            
            # Create sliding windows
            for i in range(0, len(frame_files) - self.temporal_window + 1, self.temporal_window // 2):
                sequence_frames = frame_files[i:i + self.temporal_window]
                if len(sequence_frames) == self.temporal_window:
                    sequences.append(sequence_frames)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sequence of frames
        Returns:
            frames: Tensor of shape (temporal_window, C, H, W)
        """
        frame_paths = self.sequences[idx]
        frames = []
        
        for frame_path in frame_paths:
            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                # Handle missing frames
                frame = np.zeros((MODEL_CONFIG['input_size'][0], 
                                MODEL_CONFIG['input_size'][1], 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                frame = cv2.resize(frame, MODEL_CONFIG['input_size'])
            
            frames.append(frame)
        
        # Stack frames
        frames = np.stack(frames, axis=0)  # (T, H, W, C)
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        else:
            # Default: normalize and convert to tensor
            frames = frames.astype(np.float32) / 255.0
            
            # Normalize with ImageNet stats
            mean = np.array(DATA_CONFIG['normalize_mean'])
            std = np.array(DATA_CONFIG['normalize_std'])
            frames = (frames - mean) / std
            
            # Convert to tensor (T, H, W, C) -> (T, C, H, W)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        return frames


if __name__ == "__main__":
    # Test data loader
    print("Testing data loader...")
    
    # Download dataset
    dataset_path = download_dataset()
    
    if dataset_path:
        # Create dataset
        train_dataset = UCSDDataset(dataset_path, subset='Train', temporal_window=16)
        test_dataset = UCSDDataset(dataset_path, subset='Test', temporal_window=16)
        
        print(f"\nTrain dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Test loading a sample
        sample = train_dataset[0]
        print(f"\nSample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
