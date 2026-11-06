import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
from scipy import ndimage
from inference.entity.config_entity import ModelInferenceConfig
from inference.logging import logger

class ModelInference:
    def __init__(self, config: ModelInferenceConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        logger.info(f"Loading U-Net model from {self.config.model_path}")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=4,
            classes=1
        ).to(self.device)

        if Path(self.config.model_path).exists():
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model weights not found at {self.config.model_path}")
        
        model.eval()
        return model

    def predict_mask(self, model, image_tensor):
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device).float()
        
        with torch.no_grad():
            output = model(image_tensor)
            pred_sigmoid = torch.sigmoid(output)
            pred_mask = (pred_sigmoid > 0.5).float()
            
        return pred_mask.squeeze(0).cpu()

    def detect_deforestation(self, mask_t1, mask_t2, min_pixels=50):
        """Compares two dates and filters out noise patches below half a hectare (~50 pixels)."""
        cleared_mask = (mask_t1 == 1) & (mask_t2 == 0)
        mask_np = cleared_mask.numpy().astype(int) if isinstance(cleared_mask, torch.Tensor) else np.squeeze(cleared_mask).astype(int)
        
        labeled_array, num_features = ndimage.label(mask_np)
        filtered_mask = np.zeros_like(mask_np)
        
        removed_count = 0
        retained_count = 0
        
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            if component_mask.sum() >= min_pixels:
                filtered_mask[component_mask] = 1
                retained_count += 1
            else:
                removed_count += 1
                
        logger.info(f"Noise Filtering: Removed {removed_count} small patches (< {min_pixels} px), Retained {retained_count} cleared areas.")
        return torch.tensor(filtered_mask, dtype=torch.float32)

    def run_inference_pipeline(self):
        model = self.load_model()
        
        logger.info(f"Loading test dataset arrays from {self.config.test_data_path}")
        dataset_dict = torch.load(self.config.test_data_path)
        X_full = dataset_dict["X_full"]
        
        if len(X_full) >= 2:
            img_t1 = X_full[0]
            img_t2 = X_full[1]
            
            mask_t1 = self.predict_mask(model, img_t1)
            mask_t2 = self.predict_mask(model, img_t2)
            
            cleared = self.detect_deforestation(mask_t1, mask_t2, min_pixels=50)
            
            os.makedirs(self.config.output_dir, exist_ok=True)
            output_file = Path(self.config.output_dir) / "change_detection_mask.pt"
            torch.save(cleared, output_file)
            logger.info(f"Change detection mask successfully saved to {output_file}")
        else:
            logger.warning("Not enough samples in test dataset for multi-date comparison.")