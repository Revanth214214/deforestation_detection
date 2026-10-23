import os
import torch
from deforestation_detection.logging import logger
from pathlib import Path
from deforestation_detection.entity.config_entity import DataValidationConfig
from deforestation_detection.utils.common import *

class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_tensor_data(self) -> bool:
        try:
            validation_status = True
            
            #file exist or not
            if not os.path.exists(self.config.data_file):
                validation_status = False
                logger.error(f"Data file not found at {self.config.data_file}")
            else:
                logger.info(f"Loading dataset from {self.config.data_file} for validation...")
                data = torch.load(self.config.data_file)
                
                #check values based on structure
                if isinstance(data, dict):
                    for key, tensor in data.items():
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            logger.error(f"NaN or Inf values detected in tensor key: {key}")
                            validation_status = False
                        else:
                            logger.info(f"Tensor key '{key}' validated successfully with shape: {tensor.shape}")
                            
                elif isinstance(data, (list, tuple)):
                    for idx, tensor in enumerate(data):
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            logger.error(f"NaN or Inf values detected at tensor index {idx}")
                            validation_status = False
                        else:
                            logger.info(f"Tensor at index {idx} validated successfully with shape: {tensor.shape}")
                            
                else:
                    #single tensor case
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        logger.error("NaN or Inf values detected in dataset tensor.")
                        validation_status = False
                    else:
                        logger.info(f"Dataset tensor validated successfully with shape: {data.shape}")

            #status file writing
            os.makedirs(Path(self.config.STATUS_FILE).parent, exist_ok=True)
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
                
            logger.info(f"Validation status recorded as: {validation_status} at {self.config.STATUS_FILE}")
            return validation_status

        except Exception as e:
            raise e
