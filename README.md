# Plantation Monitoring System

A system for analyzing and visualizing plantation data using satellite imagery and ground truth data. The project focuses on processing GeoJSON plantation data and creating comprehensive visualizations for analysis.

## Project Structure
```
plantation-monitoring/
├── data/
│   └── Plantations Data.geojson
├── src/
│   ├── data_preprocessing.py
│   ├── data_visualization.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── notebooks/
│   └── visualization_analysis.ipynb
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```


# Plantation Detection Model (pretraining)

### Model Architecture and Selection

#### Model Choice: DeepLabV3 with ResNet-50

The plantation detection system uses DeepLabV3 with a ResNet-50 backbone for semantic segmentation. This choice was made for several reasons:

1. **Architecture Benefits**
   - Strong feature extraction through ResNet-50 backbone
   - Atrous Spatial Pyramid Pooling (ASPP) for multi-scale processing
   - Effective handling of varying plantation sizes
   - Built-in handling of global context

2. **Model Configuration**
   ```python
   model = deeplabv3_resnet50(pretrained=False)
   model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=(1, 1))  # Binary classification
   ```
   - Modified for binary segmentation (plantation vs. non-plantation)
   - Leverages pretrained weights for feature extraction
   - Custom classification head for our specific task

3. **Input Processing**
   - Image size: 256x256 pixels
   - Three-channel input (RGB from Sentinel-2)
   - Normalization per batch
   - Optional data augmentation:
     - Random horizontal/vertical flips
     - Random rotation
     - Color jittering

#### Training Process

1. **Data Preparation**
   ```python
   # Image preprocessing
   image = (image - image.mean()) / image.std()  # Normalization
   image = torch.from_numpy(image).float()       # Convert to tensor
   ```

2. **Loss Function**
   - Binary Cross Entropy with Logits Loss
   - Handles class imbalance through weighting
   ```python
   criterion = nn.BCEWithLogitsLoss(
       pos_weight=torch.tensor([pos_weight])  # Adjusted for class balance
   )
   ```

3. **Optimization**
   - Adam optimizer with learning rate scheduling
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', patience=3
   )
   ```

4. **Training Schedule**
   - Number of epochs: 10 (default)
   - Batch size: 4 (adjustable based on GPU memory)
   - Learning rate: 0.001 with reduction on plateau
   - Checkpointing every 5 epochs

#### Model Performance Considerations

1. **Advantages**
   - Robust to varying plantation sizes
   - Good handling of spatial context
   - Efficient inference time
   - Memory-efficient training

2. **Limitations**
   - Requires good quality RGB imagery
   - May struggle with very small plantations
   - Sensitive to cloud cover in imagery

3. **Performance Metrics**
   - IoU (Intersection over Union)
   - Precision and Recall
   - F1 Score
   - Confusion Matrix analysis

#### Alternative Models Considered

1. **U-Net**
   - Pros: Lighter weight, good for medical imaging
   - Cons: Less context awareness, no pretrained weights

2. **Mask R-CNN**
   - Pros: Instance segmentation capability
   - Cons: Overkill for binary segmentation, slower inference

3. **FCN (Fully Convolutional Network)**
   - Pros: Simpler architecture
   - Cons: Less accurate on boundary details

DeepLabV3 was chosen as it provided the best balance of:
- Accuracy in plantation boundary detection
- Reasonable training time
- Good inference speed
- Memory efficiency
- Ability to handle varying plantation sizes


## Notebooks Documentation

### 1. Model Pretraining (`src/pretrain.py`)

The pretraining script handles the preparation of training data and model training for plantation detection.

#### Key Components:

1. **Data Preparation**
   ```prepare_training_data(geojson_path, output_dir, patch_size=256, max_samples=100)```
   - Downloads Sentinel-2 imagery for plantation areas
   - Creates binary masks from plantation polygons
   - Saves georeferenced image-mask pairs
   - Parameters:
     - `geojson_path`: Path to plantation polygons
     - `output_dir`: Directory to save processed data
     - `patch_size`: Size of image patches (default: 256x256)
     - `max_samples`: Maximum number of samples to process

2. **Dataset Class**
   ```python class PlantationDataset(Dataset)```
   - Handles data loading and preprocessing
   - Performs normalization and augmentation
   - Returns image-mask pairs for training

3. **Model Training**
   ```python train_model(data_dir, num_epochs=10, batch_size=4, learning_rate=0.001)```
   - Uses DeepLabV3 with ResNet-50 backbone
   - Binary segmentation for plantation detection
   - Saves model checkpoints during training

### 2. Inference and Analysis (`notebooks/plantation_analysis_inference.ipynb`)

The inference notebook provides comprehensive analysis and visualization tools.

#### Features of notebook:

1. **Dataset Analysis**
   - Ground truth statistics
   - Spatial distribution visualization
   - Interactive maps of plantation areas
   - Size distribution analysis

2. **Model Performance Analysis**
   ```showcase_best_predictions(model, data_dir, num_samples=15)```
   - Visualizes top predictions
   - Calculates performance metrics:
     - IoU (Intersection over Union)
     - Precision
     - Recall
     - F1 Score
   - Saves results to `outputs/best_predictions/`

3. **Coordinate-based Prediction**
   ```predict_on_coordinates(model, lon, lat, patch_size=256)```
   - Downloads recent Sentinel-2 imagery for given coordinates
   - Runs model prediction
   - Visualizes results with overlays
   - Calculates area statistics

4. **Error Analysis**
   - Confusion matrix visualization
   - IoU score distribution
   - False positive/negative analysis
   - Edge case examination


## Features

### 1. Data Processing (`data_preprocessing.py`)
- Reads and processes GeoJSON plantation data
- Handles coordinate transformations and spatial calculations
- Prepares data for visualization and analysis

### 2. Data Visualization (`data_visualization.py` & `visualization_analysis.ipynb`)
- Spatial distribution analysis
  - Interactive maps showing plantation locations
  - Area-based visualization
  - District-wise distribution
- Temporal analysis
  - Planting date patterns
  - Growth progression
  - Coppicing cycles
- Growth analysis
  - Quality distribution
  - Height analysis
  - Area correlations
- Species analysis
  - Distribution patterns
  - Growth characteristics
  - Area relationships

### 3. Machine Learning Pipeline
- Data preprocessing for model training
- Model training setup
- Inference pipeline

### Model Details
- Data Input Structure (8 channels total)
```
def forward(self, x):
    # x shape: [batch_size, 8, height, width]
    
    # Split input into different sources
    s2_input = x[:, :4]     # Sentinel-2: RGB + NIR bands
    s1_input = x[:, 4:6]    # Sentinel-1: VV + VH bands (radar)
    modis_input = x[:, 6:]  # MODIS: NDVI + EVI (vegetation indices)
```
- Separate Encoders for Each Source:
```
class MultiSourceUNet(nn.Module):
    def __init__(self, n_classes=1):
        # Specialized encoders for each data type
        self.s2_encoder = self._create_encoder(4)  # Optical data
        self.s1_encoder = self._create_encoder(2)  # Radar data
        self.modis_encoder = self._create_encoder(2)  # Time-series data
```
- Feature Extraction Process:
```
def _encode_single_source(self, x, encoder):
    features = []
    for layer in encoder:
        x = layer(x)
        features.append(x)  # Store features for skip connections
    return features
```
- Feature Fusion:
```
# Fusion layer combines features intelligently
self.fusion = nn.Sequential(
    nn.Conv2d(512 * 3, 512, 1),  # Combine features from all sources
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True)
)

# In forward pass
fused_features = self.fusion(torch.cat([
    s2_features[-1],  # High-level Sentinel-2 features
    s1_features[-1],  # High-level Sentinel-1 features
    modis_features[-1]  # High-level MODIS features
], dim=1))
```
### Advantages over Standard U-Net:
- Multi-Source Capability:
```
# Standard U-Net can only handle one type of input:
class StandardUNet:
    def __init__(self):
        self.encoder = single_encoder(in_channels=3)  # Only RGB

# Our MultiSourceUNet handles multiple sources:
class MultiSourceUNet:
    def __init__(self):
        self.s2_encoder = specialized_encoder(4)  # RGB + NIR
        self.s1_encoder = specialized_encoder(2)  # Radar
        self.modis_encoder = specialized_encoder(2)  # Time series
```
- Complementary Information:
```
# Each source provides unique information:
s2_features  # Spectral information (vegetation, water)
s1_features  # Radar backscatter (structure, moisture)
modis_features  # Temporal patterns (seasonal changes)
```
- Robustness to Missing Data:
```
def forward(self, x):
    # Can still work if some data is missing
    if has_s2_data:
        s2_features = self.s2_encoder(x[:, :4])
    else:
        s2_features = self.get_default_features()
    
    # Continue processing with available data
```
- Source-Specific Feature Learning:
```
def _create_encoder(self, in_channels):
    """Each encoder is optimized for its data type"""
    if in_channels == 4:  # Sentinel-2
        # Optimize for spectral data
        first_conv.weight.data[:, :3] = resnet.conv1.weight.data
        first_conv.weight.data[:, 3] = resnet.conv1.weight.data.mean(dim=1)
    elif in_channels == 2:  # Sentinel-1
        # Optimize for radar data
        # Different initialization for radar features
```
- Skip Connections with Rich Features:
```
# Decoding with rich feature combinations
dec4 = self.decoder4(torch.cat([
    dec5,  # High-level fused features
    s2_features[-2]  # Skip connection with optical features
], dim=1))
```

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/iabhi7/plantation-monitoring.git
cd plantation-monitoring
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Authenticate with Google Earth Engine:
```bash
earthengine authenticate
```

5. (Optional) Set up service account:
- Create a service-account.json file with your Google Earth Engine credentials
- Place it in the project root directory


## Usage

### 1. Data Visualization

Run the visualization script:
```bash
python src/data_visualization.py
```

This generates:
- `spatial_distribution.png`: Geographic distribution map
- `interactive_map.html`: Interactive web visualization
- `temporal_distribution.png`: Time-based analysis
- `growth_analysis.png`: Growth patterns
- `species_analysis.png`: Species distribution
- `dashboard.html`: Combined interactive dashboard

### 2. Jupyter Notebook Analysis

Start Jupyter and open `visualization_analysis.ipynb`:
```bash
jupyter notebook notebooks/visualization_analysis.ipynb
```

The notebook provides:
- Detailed data exploration
- Interactive visualizations
- Statistical analysis
- Custom analysis options

### 3. Data Preprocessing

Process the plantation data:
```bash
python src/data_preprocessing.py
```


### Basic Usage

Run the complete pipeline sequentially:
```bash
python run_pipeline.py
```

### Parallel Processing

For faster data processing, you can use parallel execution:

1. Run with default parallel settings (8 workers):
```bash
python run_pipeline.py --parallel
```

2. Run with custom number of workers:
```bash
python run_pipeline.py --parallel --workers 4
```

Note: The number of workers should be adjusted based on your system's capabilities. More workers may speed up processing but will also use more memory.

## Dependencies

Core requirements:
- geopandas
- matplotlib
- seaborn
- folium
- pandas
- numpy
- plotly
- contextily
- jupyter

Full list available in `requirements.txt`

## Testing

Run the test suite:
```bash
python test_setup.py
```

## Docker Setup

### Build Docker Image
```bash
docker-compose build
```

### Sequential Processing
```bash
# Run with sequential processing
docker-compose up plantation-monitor
```

### Parallel Processing
```bash
# Run with parallel processing (4 workers)
docker-compose up parallel-monitor

# Or specify custom workers
docker-compose run --rm parallel-monitor python run_pipeline.py --parallel --workers 8
```

### Memory Considerations
- Sequential service is limited to 8GB memory
- Parallel service is limited to 16GB memory
- Adjust memory limits in docker-compose.yml based on your system

### Service Account
To use a service account:
1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```
2. The file will be mounted automatically in the container





### Pipeline Components

The pipeline consists of several stages:
1. Data preprocessing (sequential or parallel)
2. Model training
3. Inference
4. Visualization

### Data Requirements

- Place your plantation data in `data/Plantations Data.geojson`

