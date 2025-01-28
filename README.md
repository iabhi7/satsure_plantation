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

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plantation-monitoring.git
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

1. Build and run using Docker Compose:
```bash
docker-compose up --build
```

### Commands

Common operations:
```bash
# Run visualization
docker-compose run --rm plantation-monitor python src/data_visualization.py

# Run Jupyter notebook
docker-compose run --rm -p 8888:8888 plantation-monitor jupyter notebook --ip 0.0.0.0 --allow-root
```



### Pipeline Components

The pipeline consists of several stages:
1. Data preprocessing (sequential or parallel)
2. Model training
3. Inference
4. Visualization

### Data Requirements

- Place your plantation data in `data/Plantations Data.geojson`
