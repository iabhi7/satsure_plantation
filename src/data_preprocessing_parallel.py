import geopandas as gpd
import rasterio
import numpy as np
import torch
import ee
import os
from datetime import datetime, timedelta
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import logging
from tqdm import tqdm
from functools import partial
import cv2
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_ee(service_account_key=None):
    """Initialize Earth Engine in worker process"""
    try:
        if service_account_key:
            credentials = ee.ServiceAccountCredentials(
                service_account_key['client_email'],
                key_data=service_account_key['private_key']
            )
            ee.Initialize(credentials)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        logger.error(f"Error initializing Earth Engine: {e}")
        return False

class ParallelDataCollector:
    def __init__(self, geojson_path, output_dir, n_workers=8):
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        
        try:
            self.gdf = gpd.read_file(geojson_path)
        except Exception as e:
            raise ValueError(f"Error reading GeoJSON: {e}")
        
        self.output_dir = output_dir
        self.n_workers = n_workers
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Earth Engine
        ee.Initialize()
        
    def _chunk_geometries(self, chunk_size=50):
        """Split geometries into chunks"""
        total_geoms = len(self.gdf)
        chunks = []
        
        for i in range(0, total_geoms, chunk_size):
            chunk_gdf = self.gdf.iloc[i:min(i + chunk_size, total_geoms)]
            chunks.append(chunk_gdf)
            
        return chunks

    def _process_chunk(self, args):
        """Process a single chunk of geometries"""
        chunk_idx, chunk_gdf, start_date, end_date = args
        
        try:
            # Initialize Earth Engine in worker process
            initialize_ee()
            
            logger.info(f"Processing chunk {chunk_idx}")
            
            # Get the geometry from chunk
            geometry = ee.Geometry.MultiPolygon(
                [feat['geometry']['coordinates'] for feat in chunk_gdf.__geo_interface__['features']]
            )
            
            # Process Sentinel-2
            s2_data, mask = self._process_sentinel2_chunk(geometry, start_date, end_date)
            if s2_data is None:
                return None
                
            # Process Sentinel-1 and MODIS with same dimensions
            target_size = s2_data.shape[1:]
            s1_data = self._process_sentinel1_chunk(geometry, start_date, end_date, target_size)
            modis_data = self._process_modis_chunk(geometry, start_date, end_date, target_size)
            
            if all(x is not None for x in [s2_data, mask, s1_data, modis_data]):
                # Combine data
                combined_data = np.concatenate([s2_data, s1_data, modis_data], axis=0)
                
                # Save chunk data
                chunk_filename = f"chunk_{chunk_idx}_{start_date.strftime('%Y%m')}"
                np.save(os.path.join(self.output_dir, f"{chunk_filename}_data.npy"), combined_data)
                np.save(os.path.join(self.output_dir, f"{chunk_filename}_mask.npy"), mask)
                
                return chunk_filename
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}")
            return None

    def _process_sentinel2_chunk(self, geometry, start_date, end_date):
        """Process Sentinel-2 data for a chunk"""
        # Implementation similar to original _process_sentinel2 but for a single chunk
        # ... (rest of the implementation)

    def _process_sentinel1_chunk(self, geometry, start_date, end_date, target_size):
        """Process Sentinel-1 data for a chunk"""
        # Implementation similar to original _process_sentinel1 but for a single chunk
        # ... (rest of the implementation)

    def _process_modis_chunk(self, geometry, start_date, end_date, target_size):
        """Process MODIS data for a chunk"""
        # Implementation similar to original _process_modis but for a single chunk
        # ... (rest of the implementation)

    def download_and_preprocess(self, start_date, end_date, composite_months=1):
        """Download and preprocess data in parallel"""
        logger.info("Starting parallel data preprocessing...")
        
        # Split geometries into chunks
        chunks = self._chunk_geometries()
        logger.info(f"Split data into {len(chunks)} chunks")
        
        # Prepare arguments for parallel processing
        args = [(i, chunk, start_date, end_date) for i, chunk in enumerate(chunks)]
        
        # Process chunks in parallel
        successful_chunks = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._process_chunk, arg) for arg in args]
            
            # Monitor progress with tqdm
            for future in tqdm(futures, total=len(chunks), desc="Processing chunks"):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    if result:
                        successful_chunks.append(result)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
        
        if not successful_chunks:
            raise Exception("No chunks were processed successfully")
            
        # Combine all chunk data
        logger.info("Combining chunk data...")
        self._combine_chunks(successful_chunks, start_date, end_date)
        
        logger.info("Data preprocessing completed!")

    def _combine_chunks(self, chunk_filenames, start_date, end_date):
        """Combine processed chunks into final dataset"""
        all_data = []
        all_masks = []
        
        for filename in chunk_filenames:
            data = np.load(os.path.join(self.output_dir, f"{filename}_data.npy"))
            mask = np.load(os.path.join(self.output_dir, f"{filename}_mask.npy"))
            all_data.append(data)
            all_masks.append(mask)
            
            # Clean up chunk files
            os.remove(os.path.join(self.output_dir, f"{filename}_data.npy"))
            os.remove(os.path.join(self.output_dir, f"{filename}_mask.npy"))
        
        # Combine all data
        combined_data = np.concatenate(all_data, axis=2)  # Concatenate along width
        combined_mask = np.concatenate(all_masks, axis=1)  # Concatenate along width
        
        # Save final combined data
        base_filename = f"{start_date.strftime('%Y%m')}_to_{end_date.strftime('%Y%m')}"
        np.save(os.path.join(self.output_dir, f"{base_filename}_data.npy"), combined_data)
        np.save(os.path.join(self.output_dir, f"{base_filename}_mask.npy"), combined_mask) 