import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import segmentation_models_pytorch as smp

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

class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def calculate_metrics(self, pred, target, smooth=1e-5):
        """Calculates Intersection over Union (IoU) and Dice Coefficient for segmentation."""
        pred_sigmoid = torch.sigmoid(pred)
        pred_bin = (pred_sigmoid > 0.5).float()
        
        pred_flat = pred_bin.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return iou.item(), dice.item()

    def evaluate(self):
        # 1. Load Test Dataset Arrays (matching train_test_split from training)
        dataset_dict = torch.load(self.config.test_data_path)
        X_full = dataset_dict["X_full"]
        y_full = dataset_dict["y_full"]

        _, X_te, _, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        test_dataset = TensorDataset(X_te.float(), y_te.float())
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        # 2. Reconstruct Model Architecture & Load Locked Weights
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=4,
            classes=1
        ).to(self.device)

        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.eval()

        criterion = RefinedWeightedBCEDiceLoss()

        # 3. Evaluation Loop
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_samples = 0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, masks)
                
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size

                iou, dice = self.calculate_metrics(outputs, masks)
                total_iou += iou * batch_size
                total_dice += dice * batch_size
                num_samples += batch_size

        # 4. Compute Final Average Metrics
        avg_loss = total_loss / num_samples
        avg_iou = total_iou / num_samples
        avg_dice = total_dice / num_samples

        print("--- Segmentation Evaluation Report ---")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Mean IoU: {avg_iou:.4f}")
        print(f"Dice Coefficient: {avg_dice:.4f}")

        # 5. Save Metrics to JSON natively using standard Python open()
        scores = {
            "test_loss": float(avg_loss),
            "mean_iou": float(avg_iou),
            "dice_coefficient": float(avg_dice)
        }
        
        os.makedirs(Path(self.config.metric_file_name).parent, exist_ok=True)
        with open(self.config.metric_file_name, "w") as f:
            json.dump(scores, f, indent=4)
        
        print(f"Metrics successfully saved to {self.config.metric_file_name}")