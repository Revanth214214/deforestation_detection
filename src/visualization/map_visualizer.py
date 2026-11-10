import os
import torch
import numpy as np
import folium
from pathlib import Path
from inference.logging import logger

class MapVisualizer:
    def __init__(self, mask_path: str, output_html_path: str, center_lat: float = -3.4653, center_lon: float = -62.2159):
        self.mask_path = Path(mask_path)
        self.output_html_path = Path(output_html_path)
        self.center_lat = center_lat  # Default sample region (e.g., Brazilian Amazon)
        self.center_lon = center_lon
        
        # Approximate bounding box for the region view (based on center point ± delta)
        # Assuming standard Sentinel-2 tile extent (~0.2 degrees span)
        self.delta = 0.1
        self.bounds = [
            [self.center_lat - self.delta, self.center_lon - self.delta],
            [self.center_lat + self.delta, self.center_lon + self.delta]
        ]

    def calculate_cleared_area(self, mask_tensor: torch.Tensor) -> float:
        """Calculates total cleared area in hectares. 
        Each pixel is 10m x 10m = 100 m² = 0.01 hectares.
        """
        pixel_count = torch.sum(mask_tensor == 1).item()
        hectares = pixel_count * 0.01
        return hectares

    def generate_map(self):
        try:
            if not self.mask_path.exists():
                raise FileNotFoundError(f"Change detection mask not found at {self.mask_path}")

            logger.info(f"Loading change mask from {self.mask_path}")
            mask = torch.load(self.mask_path)
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = np.array(mask)

            # Handle batch/channel dimensions if present (e.g., [1, H, W] or [H, W])
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(0)

            # Calculate hectares
            total_hectares = self.calculate_cleared_area(mask)
            logger.info(f"Total Cleared Area Detected: {total_hectares:.2f} hectares ({torch.sum(mask == 1).item()} pixels)")

            # Initialize Folium Map
            logger.info("Initializing interactive Folium map...")
            m = folium.Map(
                location=[self.center_lat, self.center_lon],
                zoom_start=11,
                tiles='OpenStreetMap'
            )

            # Add Satellite imagery basemap layer (Esri World Imagery)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(m)

            # Create an RGBA overlay image for the change mask
            # Where mask == 1, color it Red with transparency; otherwise transparent (0)
            h, w = mask_np.shape
            rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Set cleared pixels (1) to Red with 60% opacity
            rgba_mask[mask_np == 1] = [255, 0, 0, 150]

            # Add the raster overlay if there are cleared pixels
            pixel_count = torch.sum(mask == 1).item()
            if pixel_count > 0:
                folium.raster_layers.ImageOverlay(
                    image=rgba_mask,
                    bounds=self.bounds,
                    name='Deforestation Mask',
                    opacity=0.7,
                    interactive=True,
                    cross_origin=False,
                    zIndex=2
                ).add_to(m)

                logger.info(f"Added Deforestation Overlay with {pixel_count} cleared pixels highlighted in red.")
            else:
                logger.info("No cleared pixels found above threshold; map generated without change overlay.")

            # Add summary marker/popup
            folium.Marker(
                location=[self.center_lat, self.center_lon],
                popup=folium.Popup(f"<b>Deforestation Alert Summary</b><br>Total Cleared Area: <b>{total_hectares:.2f} ha</b><br>Pixels Detected: {pixel_count}", max_width=300),
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)

            folium.LayerControl().add_to(m)

            # Save map output
            self.output_html_path.parent.mkdir(parents=True, exist_ok=True)
            m.save(str(self.output_html_path))
            logger.info(f"Interactive map successfully saved to {self.output_html_path}")

        except Exception as e:
            logger.error(f"Error generating map visualization: {e}")
            raise e