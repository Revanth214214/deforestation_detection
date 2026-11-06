from inference.config.configuration import ConfigurationManager
from inference.components.model_inference import ModelInference
from inference.logging import logger

class PredictionPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            config_manager = ConfigurationManager()
            inference_config = config_manager.get_model_inference_config()
            model_inference = ModelInference(config=inference_config)
            model_inference.run_inference_pipeline()
            logger.info("Inference and change detection pipeline completed successfully!")
        except Exception as e:
            logger.error(f"Error occurred during inference pipeline execution: {e}")
            raise e