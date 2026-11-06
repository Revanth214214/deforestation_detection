from pathlib import Path
from inference.constant import *
from inference.utils.common import read_yaml, create_directories
from inference.entity.config_entity import (
    ModelInferenceConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_model_inference_config(self) -> ModelInferenceConfig:
        config = self.config.model_inference
        create_directories([config.root_dir])
        
        model_inference_config = ModelInferenceConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            test_data_path=Path(config.test_data_path),
            output_dir=Path(config.output_dir)
        )
        return model_inference_config