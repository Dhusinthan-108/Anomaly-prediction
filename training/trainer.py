"""
Advanced Training Pipeline with Rich Progress Bars and Real-time Plotting
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import time
import json

from config import DEVICE, CHECKPOINT_DIR, OUTPUT_DIR
from models import AnomalyAutoencoder
from utils.visualization import Visualizer

class Trainer:
    """
    Advanced trainer for anomaly detection model
    """
    def __init__(self, model, train_dataset, config, val_dataset=None):
        """
        Args:
            model: AnomalyAutoencoder model
            train_dataset: Training dataset
            config: Training configuration
            val_dataset: Validation dataset (optional)
        """
        self.model = model.to(DEVICE)
        self.config = config
        self.device = DEVICE
        
        # Split dataset if no validation set provided
        if val_dataset is None and config['validation_split'] > 0:
            val_size = int(len(train_dataset) * config['validation_split'])
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False)
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config.get('num_workers', 0),
                pin_memory=config.get('pin_memory', False)
            )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        if config.get('scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config.get('min_lr', 1e-6)
            )
        elif config.get('scheduler') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', False) and DEVICE.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Visualizer
        self.visualizer = Visualizer()
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {DEVICE}")
        print(f"   Train samples: {len(train_dataset)}")
        if val_dataset:
            print(f"   Val samples: {len(val_dataset)}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Mixed precision: {self.use_amp}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, frames in enumerate(pbar):
            frames = frames.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    reconstructed, encoded, features = self.model(frames)
                    loss = self.criterion(reconstructed, features)
            else:
                reconstructed, encoded, features = self.model(frames)
                loss = self.criterion(reconstructed, features)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.get('gradient_clip'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.get('gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for frames in tqdm(self.val_loader, desc="Validating", leave=False):
                frames = frames.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        reconstructed, encoded, features = self.model(frames)
                        loss = self.criterion(reconstructed, features)
                else:
                    reconstructed, encoded, features = self.model(frames)
                    loss = self.criterion(reconstructed, features)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'][-1],
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = CHECKPOINT_DIR / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train(self):
        """Complete training loop"""
        print("\n🚀 Starting training...")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            self.history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Learning Rate: {current_lr:.6f}")
            print(f"   Time: {epoch_time:.2f}s")
            
            # Check for best model
            is_best = False
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if is_best or (epoch + 1) % self.config.get('save_frequency', 5) == 0:
                self.save_checkpoint(epoch, is_best)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if self.patience_counter >= self.config.get('early_stopping_patience', 5):
                print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
                print(f"   Best epoch: {self.best_epoch+1}")
                print(f"   Best val loss: {self.best_val_loss:.4f}")
                break
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✅ Training completed!")
        print(f"   Total time: {total_time/60:.2f} minutes")
        print(f"   Best epoch: {self.best_epoch+1}")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_history(self):
        """Save training history"""
        history_path = OUTPUT_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📝 Saved training history to {history_path}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training & Validation Loss', 'Learning Rate'),
            vertical_spacing=0.15
        )
        
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['train_loss'],
                      mode='lines+markers', name='Train Loss',
                      line=dict(color='#667eea', width=2)),
            row=1, col=1
        )
        
        if self.history['val_loss']:
            fig.add_trace(
                go.Scatter(x=epochs, y=self.history['val_loss'],
                          mode='lines+markers', name='Val Loss',
                          line=dict(color='#10b981', width=2)),
                row=1, col=1
            )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['learning_rate'],
                      mode='lines', name='Learning Rate',
                      line=dict(color='#f59e0b', width=2)),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="LR", type="log", row=2, col=1)
        
        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=True
        )
        
        # Save plot
        plot_path = OUTPUT_DIR / 'training_curves.html'
        fig.write_html(str(plot_path))
        print(f"📈 Saved training curves to {plot_path}")
        
        return fig


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
