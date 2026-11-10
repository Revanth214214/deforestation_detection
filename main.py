from deforestation_detection.pipeline.stage_01_DI import DataIngestionTrainingPipeline
from deforestation_detection.pipeline.stage_02_DV import DataValidationTrainingPipeline
from deforestation_detection.pipeline.stage_03_MT import ModelTrainerPipeline
from deforestation_detection.pipeline.stage_04_ME import ModelEvaluationPipeline
from inference.pipeline.inference_pipeline import PredictionPipeline
from visualization.vis_pipeline import VisualizationPipeline
from deforestation_detection.logging import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    model_trainer = ModelTrainerPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Inference & Change Detection Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prediction_pipeline = PredictionPipeline()
    prediction_pipeline.run_pipeline()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x\n")
except Exception as e:
    logger.exception(f"Stage {STAGE_NAME} failed with error: {e}")
    raise e

STAGE_NAME = "Map Visualization & Hectare Calculation Stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        vis_pipeline = VisualizationPipeline()
        vis_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x=x\n")
    except Exception as e:
        logger.exception(f"Stage {STAGE_NAME} failed with error: {e}")
        raise e