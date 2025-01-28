import os
from datetime import datetime, timedelta
from src.data_preprocessing import MultiSourceDataCollector
from src.train import main as train_main
from src.inference import main as inference_main
from src.model import MultiSourceUNet
import json

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'data/processed',
        'outputs',
        'outputs/models',
        'outputs/visualizations',
        'outputs/checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_pipeline():
    # Setup directories
    setup_directories()
    
    # Load service account key if exists
    service_account_key = None
    if os.path.exists('service-account.json'):
        with open('service-account.json', 'r') as f:
            service_account_key = json.load(f)
    
    # Initialize data collector with parallel processing
    collector = MultiSourceDataCollector(
        geojson_path='data/Plantations Data.geojson',
        output_dir='data/processed'
    )
    
    # Set date range for data collection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last year's data
    
    print("Starting data preprocessing...")
    # Download and preprocess the data with 6-month composites
    collector.download_and_preprocess(
        start_date=start_date,
        end_date=end_date,
        composite_months=1
    )
    
    print("Data preprocessing completed!")
    
    # Train model
    print("Starting model training...")
    train_main()
    
    # Run inference
    print("Running inference...")
    inference_main()

if __name__ == '__main__':
    run_pipeline() 