from deforestation_detection import *
from deforestation_detection.logging import logger
from deforestation_detection.utils.common import read_yaml, create_directories
from deforestation_detection.entity.config_entity import DataIngestionConfig
from deforestation_detection.entity.config_entity import DataValidationConfig


from deforestation_detection.constant import *
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,     # Access to constants
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath) # read all config and params yaml files
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
        )

        return data_ingestion_config
    def get_data_validation_config(self) -> DataValidationConfig:
            config = self.config.data_validation
    
            create_directories([config.root_dir])
    
            data_validation_config = DataValidationConfig(
                root_dir = config.root_dir,
                data_file= config.data_file,
                STATUS_FILE = config.STATUS_FILE
            )
    
            return data_validation_config