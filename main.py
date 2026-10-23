from deforestation_detection.pipeline.stage_01_DI import DataIngestionTrainingPipeline
from deforestation_detection.pipeline.stage_02_DV import DataValidationTrainingPipeline
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
    data_ingestion = DataValidationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e