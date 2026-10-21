import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from deforestation_detection.logging import logger
from deforestation_detection.utils.common import *
from deforestation_detection.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_files(self):
        """
        Downloads the dataset file from Hugging Face using huggingface_hub 
        with explicit anonymous access to handle Xet/LFS pointers.
        """
        if not os.path.exists(self.config.local_data_file):
            try:
                logger.info(f"Downloading dataset from Hugging Face into {self.config.local_data_file}")
                
                # Use hf_hub_download with token=False for public Xet-backed files
                downloaded_path = hf_hub_download(
                    repo_id="Revanth214/Deforestation",
                    filename="dataset_arrays.pt",
                    repo_type="dataset",
                    token=False
                )
                
                os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
                shutil.copy(downloaded_path, self.config.local_data_file)
                
                logger.info(f"Successfully downloaded file with size: {get_size(Path(self.config.local_data_file))}")
                
            except Exception as e:
                raise e
        else:
            logger.info(f"File already exists with size: {get_size(Path(self.config.local_data_file))}")