"""
Global Configuration for Anomaly Detection System
"""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Directory paths
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"
EXAMPLES_DIR = ROOT_DIR / "examples"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, EXAMPLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'feature_dim': 1280,  # EfficientNet-B0 output
    'temporal_window': 16,  # Number of frames per sequence
    'lstm_hidden_dim': 512,
    'lstm_layers': 2,
    'dropout': 0.2,
    'input_size': (224, 224),  # Frame size
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'epochs': 30,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'early_stopping_patience': 5,
    'validation_split': 0.2,
    'num_workers': 4,
    'mixed_precision': True,
}

# Data configuration
DATA_CONFIG = {
    'ucsd_ped1_url': 'http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz',
    'frame_rate': 10,  # FPS for video processing
    'normalize_mean': [0.485, 0.456, 0.406],  # ImageNet stats
    'normalize_std': [0.229, 0.224, 0.225],
}

# Inference configuration
INFERENCE_CONFIG = {
    'threshold': 0.5,  # Anomaly score threshold
    'smoothing_window': 5,  # Temporal smoothing window
    'confidence_threshold': 0.7,
    'batch_size': 16,
}

# UI configuration
UI_CONFIG = {
    'theme': 'soft',
    'primary_color': '#667eea',
    'secondary_color': '#764ba2',
    'success_color': '#10b981',
    'warning_color': '#f59e0b',
    'danger_color': '#ef4444',
    'background_dark': '#1a1a2e',
    'surface_dark': '#16213e',
    'text_light': '#eaeaea',
}

# Device configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")
