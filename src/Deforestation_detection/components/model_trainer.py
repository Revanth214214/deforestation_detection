import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from pathlib import Path
from deforestation_detection.logging import logger
from deforestation_detection.entity.config_entity import ModelTrainerConfig
from deforestation_detection.utils.common import *

class RefinedWeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=3.0, smooth=1.0):
        super(RefinedWeightedBCEDiceLoss, self).__init__()
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        
        device = inputs.device
        pos_weight_tensor = torch.tensor([self.pos_weight], device=device)

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs.view(-1), targets_flat, pos_weight=pos_weight_tensor
        )
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return bce_loss + dice_loss


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        try:
            model_path = Path(self.config.root_dir) / self.config.model_name

            # Skip training if the model checkpoint already exists
            if os.path.exists(model_path):
                logger.info(f"Model checkpoint already exists at: {model_path}. Skipping training.")
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")

            # 1. Load dataset arrays
            logger.info(f"Loading data from {self.config.train_data_path}")
            dataset_dict = torch.load(self.config.train_data_path)
            
            X_full = dataset_dict["X_full"]
            y_full = dataset_dict["y_full"]
            logger.info(f"Loaded X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")

            # 2. Train/Test Split (matching your Trial 12 setup)
            X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
            logger.info(f"Train samples: {X_tr.shape[0]}, Test samples: {X_te.shape[0]}")

            # 3. Rebuild DataLoaders
            train_dataset = TensorDataset(X_tr.float(), y_tr.float())
            test_dataset = TensorDataset(X_te.float(), y_te.float())

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

            # 4. Initialize SMP U-Net Model
            logger.info(f"Initializing U-Net with encoder: {self.config.encoder_name}")
            model = smp.Unet(
                encoder_name=self.config.encoder_name,
                encoder_weights=None,
                in_channels=self.config.in_channels,
                classes=self.config.classes
            ).to(device)

            criterion = RefinedWeightedBCEDiceLoss(pos_weight=self.config.pos_weight)
            optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

            # 5. Training Loop
            best_val_loss = float('inf')
            save_path = Path(self.config.root_dir) / self.config.model_name

            logger.info(f"Starting training for {self.config.epochs} epochs...")
            
            for epoch in range(self.config.epochs):
                model.train()
                train_loss = 0.0

                for images, masks in train_loader:
                    images, masks = images.to(device), masks.to(device)
                    if masks.ndim == 3:
                        masks = masks.unsqueeze(1)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * images.size(0)

                epoch_train_loss = train_loss / len(train_loader.dataset)

                # Validation step
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks in test_loader:
                        images, masks = images.to(device), masks.to(device)
                        if masks.ndim == 3:
                            masks = masks.unsqueeze(1)

                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item() * images.size(0)

                epoch_val_loss = val_loss / len(test_loader.dataset)

                logger.info(f"Epoch [{epoch+1}/{self.config.epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

                # Save best model checkpoint
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    os.makedirs(self.config.root_dir, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"--> Saved Best Model Checkpoint at Epoch {epoch+1}")

            logger.info(f"Model Training Complete! Best weights saved to: {save_path}")

        except Exception as e:
            raise e