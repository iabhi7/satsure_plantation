import os
from datetime import datetime, timedelta
from src.data_preprocessing import MultiSourceDataCollector
from src.data_preprocessing_parallel import ParallelDataCollector
from src.train import main as train_main
from src.inference import main as inference_main
from src.model import MultiSourceUNet
import json
import argparse

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

def run_pipeline(use_parallel=False, n_workers=8):
    """Run the complete pipeline
    
    Args:
        use_parallel (bool): Whether to use parallel processing
        n_workers (int): Number of workers for parallel processing
    """
    # Setup directories
    setup_directories()
    
    # Load service account key if exists
    service_account_key = None
    if os.path.exists('service-account.json'):
        with open('service-account.json', 'r') as f:
            service_account_key = json.load(f)
    
    # Initialize data collector based on parallel flag
    if use_parallel:
        print(f"Using parallel processing with {n_workers} workers")
        collector = ParallelDataCollector(
            geojson_path='data/Plantations Data.geojson',
            output_dir='data/processed',
            n_workers=n_workers
        )
    else:
        print("Using sequential processing")
        collector = MultiSourceDataCollector(
            geojson_path='data/Plantations Data.geojson',
            output_dir='data/processed'
        )
    
    # Set date range for data collection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("Starting data preprocessing...")
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

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run plantation detection pipeline')
    parser.add_argument('--parallel', action='store_true',
                      help='Use parallel processing for data collection')
    parser.add_argument('--workers', type=int, default=8,
                      help='Number of workers for parallel processing')
    
    args = parser.parse_args()
    
    # Run pipeline with specified arguments
    run_pipeline(use_parallel=args.parallel, n_workers=args.workers)

if __name__ == '__main__':
    main() 