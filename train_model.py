"""
Training Script for Anomaly Detection Model
Train the model on UCSD Pedestrian dataset
"""

import argparse
import torch
from pathlib import Path

from config import MODEL_CONFIG, TRAINING_CONFIG, CHECKPOINT_DIR
from models import AnomalyAutoencoder
from utils import download_dataset, UCSDDataset
from training import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train Anomaly Detection Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🎓 TRAINING ANOMALY DETECTION MODEL")
    print("=" * 70)
    print()
    
    # Update config
    TRAINING_CONFIG['epochs'] = args.epochs
    TRAINING_CONFIG['batch_size'] = args.batch_size
    TRAINING_CONFIG['learning_rate'] = args.lr
    
    print("📋 Configuration:")
    print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print()
    
    # Download dataset
    print("📥 Preparing dataset...")
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = download_dataset()
    
    if dataset_path is None:
        print("❌ Failed to download dataset")
        return
    
    print(f"✅ Dataset ready at: {dataset_path}")
    print()
    
    # Create datasets
    print("🔄 Loading datasets...")
    train_dataset = UCSDDataset(
        dataset_path,
        subset='Train',
        temporal_window=MODEL_CONFIG['temporal_window']
    )
    
    val_dataset = UCSDDataset(
        dataset_path,
        subset='Test',
        temporal_window=MODEL_CONFIG['temporal_window']
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} sequences")
    print(f"✅ Val dataset: {len(val_dataset)} sequences")
    print()
    
    # Initialize model
    print("🏗️ Building model...")
    model = AnomalyAutoencoder(MODEL_CONFIG)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"📂 Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Checkpoint loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model built successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()
    
    # Initialize trainer
    print("🎯 Initializing trainer...")
    trainer = Trainer(model, train_dataset, TRAINING_CONFIG, val_dataset)
    print()
    
    # Train
    print("🚀 Starting training...")
    print("=" * 70)
    history = trainer.train()
    
    # Plot training curves
    print("\n📈 Generating training curves...")
    trainer.plot_training_curves()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"📊 Best model: {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"📈 Training curves: {trainer.visualizer.OUTPUT_DIR / 'training_curves.html'}")
    print("\n🎉 Ready for inference! Run: python app.py")
    print()

if __name__ == "__main__":
    main()
