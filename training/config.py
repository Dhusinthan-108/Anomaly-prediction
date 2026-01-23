"""
Training configuration
"""

TRAINING_CONFIG = {
    # Optimization
    'batch_size': 8,
    'learning_rate': 1e-4,
    'epochs': 30,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    
    # Regularization
    'dropout': 0.2,
    'label_smoothing': 0.1,
    
    # Learning rate schedule
    'scheduler': 'cosine',  # 'cosine', 'step', 'plateau'
    'warmup_epochs': 3,
    'min_lr': 1e-6,
    
    # Early stopping
    'early_stopping_patience': 5,
    'early_stopping_delta': 1e-4,
    
    # Data
    'validation_split': 0.2,
    'num_workers': 4,
    'pin_memory': True,
    
    # Performance
    'mixed_precision': True,
    'gradient_accumulation_steps': 1,
    
    # Checkpointing
    'save_best_only': True,
    'save_frequency': 5,  # Save every N epochs
    
    # Logging
    'log_frequency': 10,  # Log every N batches
    'plot_frequency': 1,  # Plot every N epochs
}
