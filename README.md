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
