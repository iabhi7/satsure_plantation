import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from tqdm import tqdm
import numpy as np
import wandb
from src.model import DiceLoss, MultiSourceUNet
from src.data_preprocessing import get_data_loaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config):
        """Initialize the trainer with configuration"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join(config.output_dir, 'runs'))
        
        # Initialize model
        self.model = MultiSourceUNet(
            # 4 S2 + 2 S1 + 2 MODIS channels
            n_classes=1  # Binary segmentation
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize criterion
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Load data
        self.train_loader, self.val_loader = get_data_loaders(
            config.data_dir,
            batch_size=config.batch_size
        )
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        
        # Initialize wandb
        wandb.init(project="plantation-detection")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Ensure output and target have same dimensions
                output = output.squeeze(1)  # Remove channel dim if present
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to tensorboard
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/train_step', loss.item(), step)
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        dice_scores = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output = output.squeeze(1)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                
                # Calculate Dice score
                pred = (output > 0.5).float()
                dice = (2. * (pred * target).sum()) / (pred.sum() + target.sum())
                dice_scores.append(dice.item())
        
        avg_val_loss = val_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # Save if best model
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint(epoch, is_best=True)
        
        avg_dice_score = np.mean(dice_scores)
        wandb.log({
            'val_loss': avg_val_loss,
            'dice_score': avg_dice_score
        })
        
        return avg_val_loss, avg_dice_score
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Training config: {self.config}")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            val_loss, dice_score = self.validate(epoch)
            logger.info(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
            
            # Save regular checkpoint
            if epoch % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(epoch)
            
            # Update learning rate
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * 0.95
            
            # Log metrics
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'dice_score': dice_score
            })
            
            print(f"Dice Score: {dice_score:.4f}")

class TrainingConfig:
    def __init__(self):
        # Create directories if they don't exist
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('outputs/checkpoints', exist_ok=True)
        
        self.data_dir = 'data/processed'
        self.output_dir = 'outputs'
        self.checkpoint_dir = 'outputs/checkpoints'
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.checkpoint_frequency = 5

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 