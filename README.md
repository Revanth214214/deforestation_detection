Roadmap & Architecture OverviewProject OverviewThis project aims to detect and quantify forest cover loss in the Brazilian Amazon using Sentinel-2 multi-spectral satellite imagery and deep learning. By comparing cloud-free median composites from two distinct years during the same dry season, a fine-tuned U-Net model classifies canopy cover and flags newly cleared land patches, presenting the results on an interactive web map.Technical SpecificationsFeatureSpecificationStudy RegionState of Mato Grosso / Rondônia, Brazilian AmazonTimeframe$T_1$: July–August 2022 vs. $T_2$: July–August 2024 (Dry Season)Data SourceSentinel-2 Level-2A (10m Resolution, 6 Bands: Blue, Green, Red, NIR, SWIR1, SWIR2)Labels / Ground TruthGoogle Dynamic World (trees probability class mask)Baseline ModelNDVI ($NDVI = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$) Thresholding ($\ge 0.6$)Primary ModelU-Net with pre-trained ResNet-34 Encoder (in_channels=6, classes=1)Loss FunctionCombined Dice Loss + Binary Cross-Entropy (BCE) LossPrimary Evaluation MetricsIntersection over Union (IoU) and F1-Score for the Forest classPost-ProcessingMorphological filtering via scipy.ndimage (filtering patches $< 0.5\text{ ha}$ / $50\text{ pixels}$)Deployment / OutputInteractive web map (folium / leafmap) with highlighted cleared areas in red + total hectares removedProject Execution FlowchartCode snippetflowchart TD
    subgraph Data Acquisition & Preprocessing
        A1[Google Earth Engine API] --> A2[Filter Sentinel-2 L2A Scenes\nJuly-Aug 2022 vs July-Aug 2024]
        A2 --> A3[Mask Clouds/Shadows via SCL Layer]
        A3 --> A4[Generate Median Composite & Export 6 Bands]
        A4 --> A5[Export Dynamic World Tree Mask Labels]
    end

    subgraph Data Splitting & Tiling
        A5 --> B1[Tile Rasters into 256x256 Patches]
        B1 --> B2[Spatial Block Split\nWest: Train | East: Test]
    end

    subgraph Modeling & Evaluation
        B2 --> C1[NDVI Baseline Thresholding]
        B2 --> C2[Train U-Net ResNet-34 Model\nin_channels=6, Dice + BCE Loss]
        C1 --> C3[Evaluate IoU & F1 on Test Block]
        C2 --> C3
    end

    subgraph Change Detection Pipeline
        C3 --> D1[Predict Forest Mask T1 - 2022]
        C3 --> D2[Predict Forest Mask T2 - 2024]
        D1 & D2 --> D3[Calculate Cleared Mask\nT1_Forest AND NOT T2_Forest]
        D3 --> D4[Filter Patches < 50 Pixels\nRemove noise under 0.5 ha]
    end

    subgraph Deployment & Map Output
        D4 --> E1[Interactive Folium / Leafmap Overlay]
        E1 --> E2[Display High-Contrast Red Cleared Areas]
        E1 --> E3[Calculate Total Hectares Deforested]
    end
Step-by-Step Implementation StrategyStep 1: Baseline EstablishmentCompute $NDVI$ on test tiles.Evaluate baseline forest detection using IoU and F1 score on the held-out geographic block.Step 2: Data Engineering & TilingDownload and tile multi-spectral bands into 256x256 GeoTIFFs.Implement geographic blocking to prevent neighboring pixel leakage between train and test sets.Step 3: U-Net Model TrainingTrain a 6-channel input U-Net architecture initialized with ImageNet weights on ResNet-34.Apply spatial augmentations (flips, 90-degree rotations).Evaluate against the NDVI baseline.Step 4: Change Detection & QuantificationGenerate predictions for 2022 and 2024.Apply logical operation cleared = forest_2022 & ~forest_2024.Suppress small patches ($< 0.5\text{ ha}$) using connected component analysis.Step 5: Visualization & Web InterfaceRender side-by-side satellite basemaps with vectorized red overlays for cleared areas.Add a summary card showing total deforested hectares.

LAYMAN TERMS:
Imagine an automated satellite security guard sitting in space. Instead of human eyes manually scanning millions of trees across the vast Amazon rainforest to see if trees are being cut down, our system uses satellite pictures taken two years apart, feeds them into an AI model, and automatically draws a bright red outline around every single spot where trees have disappeared.

1. The Core Problem We Are Solving
Deforestation happens across thousands of square kilometers. Checking this on the ground with human patrols is slow, dangerous, and expensive. Satellites take free pictures of the Earth every few days, but human analysts cannot manually look at millions of satellite images to spot changes.

Our Solution: Build an end-to-end computer vision pipeline that automatically downloads satellite images, compares two time periods, identifies tree loss, and plots it on an interactive map.

2. Step-by-Step Walkthrough (How It Works)
[Satellite Images (2022 & 2024)]
               │
               ▼
   [Clean & Prep the Data]
               │
               ▼
[Calculates Greenness Score (NDVI Baseline)]
               │
               ▼
    [AI Vision Model (U-Net)]  ◄── Learns tree canopy shapes & patterns
               │
               ▼
   [Compare 2022 vs. 2024 Masks]
               │
               ▼
  [Filter Out Small Noise (<0.5 ha)]
               │
               ▼
 [Interactive Web Map with Red Hotspots]
Phase 1: Gathering the Photos (Data Ingestion)
Where we get them: Google Earth Engine provides free satellite pictures from Europe's Sentinel-2 satellite.

The challenge: Clouds and shadows look like missing trees to a computer.

The fix: We pick photos from the exact same dry season month (e.g., August 2022 and August 2024) and blend multiple photos together to sweep away clouds, leaving us with a crystal-clear picture of the forest.

Phase 2: Teaching the AI (Model Training & Evaluation)
The Simple Rule (Our Benchmark / Baseline):

Before using heavy AI, we calculate a math score called NDVI (Normalized Difference Vegetation Index), which measures how much near-infrared light leaves bounce back. High score = vegetation; low score = bare soil or roads.

This gives us a basic benchmark score to beat.

The Smart AI Model (U-Net):

A simple score can get confused by green crops or grass. So, we train a specialized deep learning model called a U-Net (with a pre-trained ResNet-34 brain).

Instead of looking at just color, the U-Net looks at textures, shapes, and surrounding context to accurately identify actual forest canopy.

Fair Testing (No Cheating):

We do not mix neighboring photo tiles into the training and testing sets because side-by-side pictures look almost identical.

We split the region geographically: the model learns on the West side of our region and is tested strictly on the East side.

Phase 3: Finding the Missing Trees (Change Detection)
Once the model learns to identify forest canopy in both 2022 and 2024:

Subtraction: We check every pixel. If a pixel was Forest in 2022 AND NOT Forest in 2024, it gets marked as cleared.

Noise Cleaning: Small single-tree removals or minor glitches can create false alarms. We use a filter to wipe away any patch smaller than 50 pixels (about half a hectare / 0.5 ha).

Phase 4: What the Final User Sees (Deployment)
The final result is an interactive web map (like Google Maps):

Visuals: The user opens the web app and sees the satellite view. High-contrast red polygons show every location where deforestation occurred between 2022 and 2024.

Numbers: A clean scoreboard tells the user exactly how many total hectares of forest were lost.