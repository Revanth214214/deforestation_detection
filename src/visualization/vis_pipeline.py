from visualization.map_visualizer import MapVisualizer
from inference.logging import logger

class VisualizationPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            mask_file = "artifacts/model_inference/outputs/change_detection_mask.pt"
            output_map = "artifacts/visualization/deforestation_map.html"
            
            visualizer = MapVisualizer(
                mask_path=mask_file,
                output_html_path=output_map
            )
            visualizer.generate_map()
            logger.info("Visualization pipeline completed successfully!")
        except Exception as e:
            raise e