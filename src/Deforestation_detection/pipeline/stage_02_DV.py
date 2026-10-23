from deforestation_detection.logging import logger
from deforestation_detection.config.configuration import ConfigurationManager
from deforestation_detection.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config = data_validation_config)
        data_validation.validate_tensor_data()