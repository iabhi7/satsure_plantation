import os
import ee
import json
import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from datetime import datetime, timedelta
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
import random
from rasterio.features import rasterize
from shapely.affinity import translate
from shapely.geometry import box
import torchvision.transforms as transforms
from PIL import Image

class PlantationDataset(Dataset):
    def __init__(self, image_dir, transform=None, target_size=(256, 256)):
        """
        Args:
            image_dir (str): Directory with all the images and masks
            transform (callable, optional): Optional transform to be applied
            target_size (tuple): Target size for resizing (height, width)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.images = [f for f in os.listdir(os.path.join(image_dir, 'images')) 
                      if f.endswith('.tif')]

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, 'images', img_name)
        mask_path = os.path.join(self.image_dir, 'masks', 
                                img_name.replace('image', 'mask'))
        
        # Read image and mask
        with rasterio.open(img_path) as src:
            image = src.read()  # (3, H, W)
            image = image.astype(np.float32)
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # (H, W)
            mask = mask.astype(np.uint8)
        
        # Normalize image before converting
        image = (image - image.mean()) / image.std()
        
        # Convert to PIL Images for resizing
        # Ensure proper shape and scaling for PIL Image
        image = (image * 255).astype(np.uint8)  # Scale to 0-255 range
        image = np.transpose(image, (1, 2, 0))  # Change to (H, W, C)
        image = Image.fromarray(image, mode='RGB')
        mask = Image.fromarray(mask, mode='L')  # 'L' mode for single channel
        
        # Resize
        image = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)(image)
        mask = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST)(mask)
        
        # Convert back to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Convert image back to (C, H, W) format and normalize
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

    def __len__(self):
        return len(self.images)

def prepare_training_data(geojson_path, output_dir, patch_size=256, max_samples=100):
    """
    Prepare training data from geojson and Sentinel-2 imagery
    """
    # Initialize Earth Engine
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    
    # Read plantation data
    gdf = gpd.read_file(geojson_path)
    input_crs = gdf.crs  # Store the input CRS
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # Calculate date range for imagery
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Last 3 months
    
    # Convert to string format that GEE expects (YYYY-MM-DD)
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # Sample points (limit the number of samples due to compute constraints)
    sample_size = min(max_samples, len(gdf))
    sampled_polygons = gdf.sample(sample_size)
    
    for idx, row in tqdm(sampled_polygons.iterrows(), total=sample_size):
        try:
            polygon = row.geometry
            centroid = polygon.centroid
            
            # Create ROI around polygon centroid
            roi = ee.Geometry.Point([centroid.x, centroid.y]).buffer(patch_size)
            
            # Get Sentinel-2 image
            s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(roi) \
                .filterDate(start_date_str, end_date_str) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .first()
            
            if s2 is None:
                print(f"No suitable image found for sample {idx}")
                continue
                
            # Select RGB bands
            rgb = s2.select(['B4', 'B3', 'B2'])
            
            # Download image
            url = rgb.getDownloadURL({
                'scale': 10,
                'region': roi,
                'format': 'GEO_TIFF'
            })
            response = requests.get(url)
            
            # Save image with georeferencing
            img_path = f"{output_dir}/images/image_{idx}.tif"
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            # Read the downloaded image to get its transform and CRS
            with rasterio.open(img_path) as src:
                image_transform = src.transform
                image_crs = src.crs
                image_shape = src.shape
                image_bounds = src.bounds
                image_data = src.read().astype(np.float32)
            
            # Save the converted image
            with rasterio.open(
                img_path,
                'w',
                driver='GTiff',
                height=image_shape[0],
                width=image_shape[1],
                count=3,
                dtype=np.float32,
                crs=image_crs,
                transform=image_transform
            ) as dst:
                dst.write(image_data)
            
            # Create a GeoDataFrame with the single polygon
            polygon_gdf = gpd.GeoDataFrame(
                geometry=[polygon], 
                crs=input_crs
            )
            
            # Reproject polygon to match image CRS
            polygon_gdf = polygon_gdf.to_crs(image_crs)
            
            # Clip polygon to image bounds
            image_box = box(*image_bounds)
            polygon_gdf['geometry'] = polygon_gdf.geometry.intersection(image_box)
            
            # Create mask
            mask_array = rasterio.features.rasterize(
                shapes=[(geom, 1) for geom in polygon_gdf.geometry],
                out_shape=image_shape,
                transform=image_transform,
                dtype=np.uint8
            )
            
            # Save mask with same georeferencing as image
            mask_path = f"{output_dir}/masks/mask_{idx}.tif"
            with rasterio.open(
                mask_path,
                'w',
                driver='GTiff',
                height=image_shape[0],
                width=image_shape[1],
                count=1,
                dtype=np.uint8,
                crs=image_crs,
                transform=image_transform
            ) as dst:
                dst.write(mask_array, 1)
            
            # Verify alignment (uncomment for debugging)
            # verify_alignment(img_path, mask_path)
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue

def train_model(data_dir, num_epochs=10, batch_size=4, learning_rate=0.001, target_size=(256, 256)):
    """
    Fine-tune the model on plantation data
    """
    # Create dataset and dataloader
    dataset = PlantationDataset(data_dir, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Load pre-trained model
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=(1, 1))  # Binary classification
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'outputs/checkpoints/model_epoch_{epoch+1}.pth')
    
    return model

def verify_alignment(image_path, mask_path):
    """Verify that image and mask are properly aligned"""
    with rasterio.open(image_path) as img_src:
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            print(f"Image CRS: {img_src.crs}")
            print(f"Mask CRS: {mask_src.crs}")
            
            # Check transforms
            print(f"Image transform: {img_src.transform}")
            print(f"Mask transform: {mask_src.transform}")
            
            # Visualize overlay
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            
            # Plot image
            show(img_src.read([1,2,3]), transform=img_src.transform, ax=ax1)
            ax1.set_title('Satellite Image')
            
            # Plot mask
            show(mask_src.read(1), transform=mask_src.transform, ax=ax2)
            ax2.set_title('Mask')
            
            # Plot overlay
            show(img_src.read([1,2,3]), transform=img_src.transform, ax=ax3)
            show(mask_src.read(1), transform=mask_src.transform, ax=ax3, alpha=0.5)
            ax3.set_title('Overlay')
            
            plt.tight_layout()
            plt.show()

def verify_data_preparation(output_dir, num_samples=5):
    """
    Verify the prepared data by checking a few random samples
    """
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    
    if len(image_files) == 0:
        print("No images found in the output directory!")
        return
    
    # Randomly select some samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for image_file in samples:
        mask_file = image_file.replace('image', 'mask')
        
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        print(f"\nVerifying {image_file}:")
        verify_alignment(image_path, mask_path)

def main():
    # Set parameters
    geojson_path = 'data/Plantations Data.geojson'
    output_dir = 'data/processed'
    max_samples = 50  # Limit samples due to compute constraints
    
    # Prepare data
    # print("Preparing training data...")
    # prepare_training_data(geojson_path, output_dir, max_samples=max_samples)
    
    # Verify data preparation
    print("\nVerifying data preparation...")
    verify_data_preparation(output_dir, num_samples=3)
    
    # Train model
    print("\nTraining model...")
    model = train_model(output_dir)
    
    # Save final model
    torch.save(model.state_dict(), 'outputs/models/plantation_model_final.pth')

if __name__ == '__main__':
    main() 