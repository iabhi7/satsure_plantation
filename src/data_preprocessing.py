import geopandas as gpd
import rasterio
import numpy as np
import torch
from pydantic_settings import BaseSettings
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rioxarray
from shapely.geometry import box
import os
from datetime import datetime, timedelta
import ee
import geemap
from sentinelhub import SentinelHubRequest, DataCollection, MimeType, SHConfig
import boto3
import json
from shapely.geometry import shape, mapping
import rasterio.mask
from rasterio.warp import transform_bounds, transform_geom
from pyproj import CRS
from shapely.ops import transform
from functools import partial
import pyproj
import cv2

class S3DataLoader:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        
    def download_geojson(self, bucket, key, local_path):
        """Download GeoJSON file from S3"""
        try:
            self.s3_client.download_file(bucket, key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            return False

class MultiSourceDataCollector:
    def __init__(self, geojson_path, output_dir):
        self.geojson_path = geojson_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load local GeoJSON file
        self.gdf = gpd.read_file(geojson_path)
        
        # Project to a temporary UTM zone for centroid calculation
        temp_utm = self._get_utm_projection(
            self.gdf.geometry.iloc[0].centroid.x,
            self.gdf.geometry.iloc[0].centroid.y
        )
        self.gdf_projected = self.gdf.to_crs(temp_utm)
        
        # Initialize projections using the projected centroid
        self.utm_proj = self._get_utm_projection(
            self.gdf_projected.geometry.centroid.iloc[0].x,
            self.gdf_projected.geometry.centroid.iloc[0].y
        )
        self.project_to_utm = pyproj.Transformer.from_crs(
            'EPSG:4326',
            self.utm_proj,
            always_xy=True
        ).transform
        
        # Convert geometries to UTM for accurate area calculations
        self.gdf['geometry_utm'] = self.gdf['geometry'].apply(
            lambda geom: transform(self.project_to_utm, geom)
        )
        
        # Initialize Earth Engine and Sentinel Hub
        ee.Initialize()
        self.sh_config = SHConfig()
        self.sh_config.sh_client_id = 'YOUR-CLIENT-ID'
        self.sh_config.sh_client_secret = 'YOUR-CLIENT-SECRET'
    
    def _get_utm_projection(self, longitude, latitude):
        """Get the appropriate UTM projection for a given point"""
        # Normalize longitude to [-180, 180]
        longitude = longitude % 360
        if longitude > 180:
            longitude -= 360
            
        # Calculate UTM zone
        zone_number = int((longitude + 180) / 6) + 1
        
        # Ensure zone number is within valid range (1-60)
        zone_number = min(max(zone_number, 1), 60)
        
        # Northern or Southern hemisphere
        if latitude >= 0:
            epsg_code = f'EPSG:326{zone_number:02d}'  # Northern hemisphere
        else:
            epsg_code = f'EPSG:327{zone_number:02d}'  # Southern hemisphere
            
        return epsg_code
    
    def _create_mask(self, image_data, image_transform, image_crs):
        """Create binary mask from GeoJSON polygons"""
        height, width = image_data.shape[1], image_data.shape[2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Transform GeoJSON geometries to image CRS if needed
        image_crs = CRS.from_string(str(image_crs))
        geometries = []
        for geom in self.gdf.geometry:
            if image_crs != self.gdf.crs:
                transformed = transform_geom(
                    str(self.gdf.crs), 
                    str(image_crs), 
                    mapping(geom)
                )
                geometries.append(transformed)
            else:
                geometries.append(mapping(geom))
        
        # Rasterize geometries onto mask
        rasterio.features.rasterize(
            geometries,
            out=mask,
            transform=image_transform,
            default_value=1,
            dtype=np.uint8
        )
        
        return mask

    def _process_sentinel2(self, start_date, end_date):
        """Process Sentinel-2 data using Earth Engine for 6-month composite"""
        try:
            print(f"Processing Sentinel-2 composite for period: {start_date} to {end_date}")
            
            # Get the geometry from GeoJSON
            geometry = ee.Geometry.MultiPolygon(
                [feat['geometry']['coordinates'] for feat in self.gdf.__geo_interface__['features']]
            )
            
            # Get Sentinel-2 surface reflectance collection
            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
            
            # Check if we have any images
            image_count = s2_collection.size().getInfo()
            print(f"Found {image_count} Sentinel-2 images for the period")
            
            if image_count == 0:
                print(f"No Sentinel-2 images found for period")
                return None, None
            
            # Create composite
            composite = (s2_collection
                .select(['B2', 'B3', 'B4', 'B8'])
                .median())  # Using median for cloud-free composite
            
            # Get the image as a numpy array
            region = geometry.bounds()
            scale = 10  # 10m resolution
            
            # Get image data
            bands = ['B2', 'B3', 'B4', 'B8']
            image_data = composite.select(bands).reduceRegion(
                reducer=ee.Reducer.toList(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e9
            ).getInfo()
            
            if not image_data or not all(band in image_data for band in bands):
                print("No data found in the composite")
                return None, None
            
            # Convert to numpy arrays
            arrays = []
            for band in bands:
                array = np.array(image_data[band], dtype=np.float32)
                width = int(np.sqrt(len(array)))
                height = width
                array = array.reshape((height, width))
                arrays.append(array)
            
            # Stack bands
            stacked_array = np.stack(arrays)
            
            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Get the image transform
            bounds = region.getInfo()['coordinates'][0]
            transform = rasterio.transform.from_bounds(
                bounds[0][0], bounds[0][1],  # left, bottom
                bounds[2][0], bounds[2][1],  # right, top
                width, height
            )
            
            # Rasterize the geometries onto the mask
            shapes = [(geom, 1) for geom in self.gdf.geometry]
            rasterio.features.rasterize(
                shapes,
                out=mask,
                transform=transform
            )
            
            print(f"Successfully processed Sentinel-2 composite with shape: {stacked_array.shape}")
            return stacked_array, mask
            
        except Exception as e:
            print(f"Error in _process_sentinel2: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def fetch_sentinel1_data(self, start_date, end_date):
        """Fetch Sentinel-1 SAR data"""
        total_bounds = self.gdf.total_bounds
        bbox = box(*total_bounds)
        
        # Use Earth Engine to get Sentinel-1 data
        geometry = ee.Geometry.Rectangle(list(total_bounds))
        s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW')))
        
        return s1_collection
    
    def fetch_modis_data(self, start_date, end_date):
        """Fetch MODIS data"""
        total_bounds = self.gdf.total_bounds
        geometry = ee.Geometry.Rectangle(list(total_bounds))
        
        modis_collection = (ee.ImageCollection('MODIS/006/MOD13Q1')
            .filterBounds(geometry)
            .filterDate(start_date, end_date))
        
        return modis_collection
    
    def _resample_array(self, array, target_size):
        """Resample array to target size using bilinear interpolation"""
        if array.shape[1:] == target_size:
            return array
            
        # Reshape array for torch interpolation
        tensor = torch.from_numpy(array).float().unsqueeze(0)  # Add batch dimension
        
        # Use interpolate to resize
        resampled = torch.nn.functional.interpolate(
            tensor,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )
        
        return resampled.squeeze(0).numpy()  # Remove batch dimension

    def _process_sentinel1(self, start_date, end_date, target_size=None):
        """Process Sentinel-1 data using Earth Engine for 6-month composite"""
        try:
            print(f"Processing Sentinel-1 composite for period: {start_date} to {end_date}")
            
            # Get the geometry from GeoJSON
            geometry = ee.Geometry.MultiPolygon(
                [feat['geometry']['coordinates'] for feat in self.gdf.__geo_interface__['features']]
            )
            
            # Get Sentinel-1 GRD collection
            s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .filter(ee.Filter.eq('instrumentMode', 'IW')))
            
            # Check if we have any images
            image_count = s1_collection.size().getInfo()
            print(f"Found {image_count} Sentinel-1 images for the period")
            
            if image_count == 0:
                print(f"No Sentinel-1 images found for period")
                return None
            
            # Create composite
            composite = (s1_collection
                .select(['VV', 'VH'])
                .median())  # Using median for speckle reduction
            
            # Apply additional speckle filtering
            composite = composite.focal_mean(3)
            
            # Get the image as a numpy array
            region = geometry.bounds()
            scale = 10  # 10m resolution
            
            # Get image data
            bands = ['VV', 'VH']
            image_data = composite.select(bands).reduceRegion(
                reducer=ee.Reducer.toList(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e9
            ).getInfo()
            
            if not image_data or not all(band in image_data for band in bands):
                print("No data found in the Sentinel-1 composite")
                return None
            
            # Convert to numpy arrays
            arrays = []
            for band in bands:
                array = np.array(image_data[band], dtype=np.float32)
                width = int(np.sqrt(len(array)))
                height = width
                array = array.reshape((height, width))
                arrays.append(array)
            
            # Stack bands
            stacked_array = np.stack(arrays)
            
            # Resample if target size is provided
            if target_size is not None and stacked_array.shape[1:] != target_size:
                print(f"Resampling Sentinel-1 data from {stacked_array.shape[1:]} to {target_size}")
                stacked_array = self._resample_array(stacked_array, target_size)
                
            print(f"Final Sentinel-1 composite shape: {stacked_array.shape}")
            return stacked_array
            
        except Exception as e:
            print(f"Error in _process_sentinel1: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_modis(self, start_date, end_date, target_size=None):
        """Process MODIS data using Earth Engine for 6-month composite"""
        try:
            print(f"Processing MODIS composite for period: {start_date} to {end_date}")
            
            # Get the geometry from GeoJSON
            geometry = ee.Geometry.MultiPolygon(
                [feat['geometry']['coordinates'] for feat in self.gdf.__geo_interface__['features']]
            )
            
            # Get MODIS collection (MOD13Q1: 16-day vegetation indices)
            modis_collection = (ee.ImageCollection('MODIS/006/MOD13Q1')
                .filterBounds(geometry)
                .filterDate(start_date, end_date))
            
            # Check if we have any images
            image_count = modis_collection.size().getInfo()
            print(f"Found {image_count} MODIS images for the period")
            
            if image_count == 0:
                print(f"No MODIS images found for period")
                return None
            
            # Create composite
            composite = (modis_collection
                .select(['NDVI', 'EVI'])
                .median())
            
            # Get the image as a numpy array
            region = geometry.bounds()
            scale = 250  # MODIS resolution
            
            # Get image data
            bands = ['NDVI', 'EVI']
            image_data = composite.select(bands).reduceRegion(
                reducer=ee.Reducer.toList(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e9
            ).getInfo()
            
            if not image_data or not all(band in image_data for band in bands):
                print("No data found in the MODIS composite")
                return None
            
            # Convert to numpy arrays
            arrays = []
            for band in bands:
                array = np.array(image_data[band], dtype=np.float32)
                width = int(np.sqrt(len(array)))
                height = width
                array = array.reshape((height, width))
                # Scale NDVI and EVI to 0-1 range
                array = array * 0.0001  # MODIS scaling factor
                arrays.append(array)
            
            # Stack bands
            stacked_array = np.stack(arrays)
            
            # Resample if target size is provided
            if target_size is not None and stacked_array.shape[1:] != target_size:
                print(f"Resampling MODIS data from {stacked_array.shape[1:]} to {target_size}")
                stacked_array = self._resample_array(stacked_array, target_size)
                
            print(f"Final MODIS composite shape: {stacked_array.shape}")
            return stacked_array
            
        except Exception as e:
            print(f"Error in _process_modis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def download_and_preprocess(self, start_date, end_date, composite_months=6):
        """Download and preprocess multi-source imagery using 6-month composites"""
        print(f"Starting data preprocessing from {start_date} to {end_date}")
        
        # Generate dates for 6-month intervals
        current_date = start_date
        while current_date < end_date:
            try:
                period_end = min(current_date + timedelta(days=composite_months*30), end_date)
                print(f"\nProcessing period: {current_date} to {period_end}")
                
                # Process Sentinel-2 first to get the target size
                s2_data, mask = self._process_sentinel2(current_date, period_end)
                if s2_data is None:
                    print("No Sentinel-2 data available for this period")
                    current_date = period_end
                    continue
                
                target_size = s2_data.shape[1:]
                print(f"Using target size from Sentinel-2: {target_size}")
                
                # Process other data sources with resampling
                s1_data = self._process_sentinel1(current_date, period_end, target_size)
                modis_data = self._process_modis(current_date, period_end, target_size)
                
                if all(x is not None for x in [s2_data, mask, s1_data, modis_data]):
                    # Verify shapes before concatenating
                    print(f"Shapes before concatenation:")
                    print(f"Sentinel-2: {s2_data.shape}")
                    print(f"Sentinel-1: {s1_data.shape}")
                    print(f"MODIS: {modis_data.shape}")
                    
                    # Combine all data sources
                    combined_data = np.concatenate([s2_data, s1_data, modis_data], axis=0)
                    
                    # Save the preprocessed data and mask
                    base_filename = f"{current_date.strftime('%Y%m')}_to_{period_end.strftime('%Y%m')}"
                    data_path = os.path.join(self.output_dir, f"{base_filename}_data.npy")
                    mask_path = os.path.join(self.output_dir, f"{base_filename}_mask.npy")
                    
                    print(f"Saving combined data to {data_path}")
                    np.save(data_path, combined_data)
                    print(f"Saving mask to {mask_path}")
                    np.save(mask_path, mask)
                    
                    print(f"Successfully saved composite for period {base_filename}")
                    print(f"Combined data shape: {combined_data.shape}")
                else:
                    print(f"Missing data for period {current_date} to {period_end}")
                    
            except Exception as e:
                print(f"Error processing period {current_date} to {period_end}: {e}")
                import traceback
                traceback.print_exc()
                
            current_date = period_end

class MultiSourceDataset(Dataset):
    def __init__(self, image_dir, patch_size=256, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transform = transform
        
        # Verify directory exists
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory {image_dir} does not exist")
        
        # Get all matching data and mask pairs
        self.data_files = []
        for f in os.listdir(image_dir):
            if f.endswith('_data.npy'):
                mask_file = f.replace('_data.npy', '_mask.npy')
                if os.path.exists(os.path.join(image_dir, mask_file)):
                    self.data_files.append(f)
        
        if not self.data_files:
            raise ValueError(f"No valid data/mask pairs found in {image_dir}")
            
        self.data_files.sort()
        
    def __len__(self):
        return len(self.data_files)
        
    def __getitem__(self, idx):
        try:
            # Load data and corresponding mask
            base_filename = self.data_files[idx].replace('_data.npy', '')
            data_path = os.path.join(self.image_dir, f"{base_filename}_data.npy")
            mask_path = os.path.join(self.image_dir, f"{base_filename}_mask.npy")
            
            if not os.path.exists(data_path) or not os.path.exists(mask_path):
                raise FileNotFoundError(f"Data or mask file not found for {base_filename}")
            
            data = np.load(data_path)
            mask = np.load(mask_path)
            
            # Verify data dimensions
            if len(data.shape) != 3 or len(mask.shape) != 2:
                raise ValueError(f"Invalid data dimensions for {base_filename}")
            
            # Random crop if patch size is specified and smaller than image dimensions
            if self.patch_size:
                h, w = data.shape[1], data.shape[2]
                
                if h < self.patch_size or w < self.patch_size:
                    # If image is smaller than patch size, resize it
                    new_h = max(h, self.patch_size)
                    new_w = max(w, self.patch_size)
                    
                    # Resize data
                    data_resized = np.zeros((data.shape[0], new_h, new_w), dtype=data.dtype)
                    for c in range(data.shape[0]):
                        data_resized[c] = cv2.resize(data[c], (new_w, new_h))
                    data = data_resized
                    
                    # Resize mask
                    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    
                    h, w = new_h, new_w
                
                # Now perform random crop
                i = np.random.randint(0, h - self.patch_size + 1)
                j = np.random.randint(0, w - self.patch_size + 1)
                
                data = data[:, i:i+self.patch_size, j:j+self.patch_size]
                mask = mask[i:i+self.patch_size, j:j+self.patch_size]
            
            # Convert to torch tensors
            data = torch.from_numpy(data).float()
            mask = torch.from_numpy(mask).float()
            
            if self.transform:
                data = self.transform(data)
            
            return data, mask
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Return a zero tensor of the expected shape as fallback
            return torch.zeros((8, self.patch_size, self.patch_size)), torch.zeros((self.patch_size, self.patch_size))

def get_data_loaders(image_dir, batch_size=8):
    # Define transformations for each data source
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5,    # S2 bands (RGB + NIR)
                  -10.0, -17.0,                 # S1 bands (adjusted VV, VH values)
                  0.3, 0.3],                    # MODIS bands (adjusted for typical NDVI/EVI ranges)
            std=[0.229, 0.224, 0.225, 0.2,     # S2 bands
                 4.0, 5.0,                      # S1 bands (adjusted for typical variance)
                 0.15, 0.15]                    # MODIS bands
        )
    ])
    
    # Create dataset
    dataset = MultiSourceDataset(image_dir, transform=transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=4)
    
    return train_loader, val_loader 

def test_data_pipeline():
    """Test the complete data pipeline with a small example"""
    import tempfile
    import shutil
    from datetime import datetime
    print("Starting data pipeline test...")
    
    # Initialize Earth Engine with authentication
    try:
        ee.Initialize()
    except Exception as e:
        print("Earth Engine authentication failed. Please authenticate using:")
        print("1. Run 'earthengine authenticate' in your terminal")
        print("2. Follow the instructions to get your authentication token")
        print(f"Error details: {e}")
        return
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, 'processed_data')
    
    try:
        # Create a simple GeoJSON for testing
        test_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.084, 37.422],
                        [-122.084, 37.424],
                        [-122.082, 37.424],
                        [-122.082, 37.422],
                        [-122.084, 37.422]
                    ]]
                },
                "properties": {}
            }]
        }
        
        # Save test GeoJSON
        geojson_path = os.path.join(temp_dir, 'test.geojson')
        with open(geojson_path, 'w') as f:
            json.dump(test_geojson, f)
        
        print("Testing data collection...")
        # Initialize data collector
        collector = MultiSourceDataCollector(geojson_path, output_dir)
        
        # Set date range for testing (use a full year for testing composites)
        start_date = datetime(2022, 1, 1)  # Changed to 2022 for better data availability
        end_date = datetime(2022, 12, 31)
        
        print("Downloading and preprocessing data...")
        # Download and preprocess the data with 6-month composites
        collector.download_and_preprocess(
            start_date=start_date,
            end_date=end_date,
            composite_months=6
        )
        
        # Verify that files were created
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            print(f"No files were created in {output_dir}")
            return
            
        print(f"Created files: {os.listdir(output_dir)}")
        
        # Create a simple dataset and test the DataLoader
        print("Testing DataLoader...")
        train_loader, val_loader = get_data_loaders(output_dir, batch_size=2)
        
        # Try to get one batch
        for batch_data, batch_masks in train_loader:
            print(f"Successfully loaded batch with shape: {batch_data.shape}")
            break
        
        print("Data pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_data_pipeline() 