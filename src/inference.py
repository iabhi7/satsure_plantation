import torch
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from src.model import MultiSourceUNet
from rasterio.features import shapes
from shapely.geometry import shape
import folium
from folium import plugins
import json
from src.data_preprocessing import get_data_loaders

class PlantationPredictor:
    def __init__(self, model_path, device):
        self.device = device
        self.model = MultiSourceUNet(n_channels=4, n_classes=1)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        
    def predict_tile(self, image):
        """Predict plantation areas for a single tile"""
        with torch.no_grad():
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            prediction = self.model(image_tensor)
            prediction = prediction.squeeze().cpu().numpy()
        return prediction
    
    def predict_large_image(self, image, tile_size=1024, overlap=100):
        """Predict plantation areas for a large image using sliding window"""
        height, width = image.shape[1], image.shape[2]
        prediction = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        
        for y in range(0, height - overlap, tile_size - overlap):
            for x in range(0, width - overlap, tile_size - overlap):
                y2 = min(y + tile_size, height)
                x2 = min(x + tile_size, width)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)
                
                tile = image[:, y1:y2, x1:x2]
                pred_tile = self.predict_tile(tile)
                
                prediction[y1:y2, x1:x2] += pred_tile
                counts[y1:y2, x1:x2] += 1
                
        prediction = prediction / counts
        return prediction
    
    def prediction_to_polygons(self, prediction, transform, threshold=0.5):
        """Convert prediction mask to polygons"""
        mask = prediction > threshold
        mask = mask.astype(np.uint8)
        
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(mask, transform=transform))
            if v == 1
        )
        
        geoms = list(results)
        polygons = [shape(geom['geometry']) for geom in geoms]
        return polygons
    
    def visualize_results(self, image, prediction, output_path):
        """Visualize prediction results"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display RGB image
        rgb = image[0:3].transpose(1, 2, 0)
        ax1.imshow(rgb)
        ax1.set_title('RGB Image')
        
        # Display prediction
        ax2.imshow(prediction, cmap='jet')
        ax2.set_title('Prediction')
        
        # Display overlay
        overlay = np.zeros_like(rgb)
        overlay[:, :, 1] = prediction * 255  # Green channel
        ax3.imshow(rgb)
        ax3.imshow(overlay, alpha=0.5)
        ax3.set_title('Overlay')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def create_interactive_map(self, polygons, bounds, output_path):
        """Create interactive map with predictions"""
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add prediction polygons
        for polygon in polygons:
            folium.GeoJson(
                polygon.__geo_interface__,
                style_function=lambda x: {
                    'fillColor': '#00ff00',
                    'color': '#00ff00',
                    'weight': 1,
                    'fillOpacity': 0.5
                }
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(output_path)

def main():
    # Initialize predictor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = PlantationPredictor('best_model.pth', device)
    
    # Load test image
    with rasterio.open('path/to/test/image.tif') as src:
        image = src.read()
        transform = src.transform
        bounds = src.bounds
    
    # Make prediction
    prediction = predictor.predict_large_image(image)
    
    # Convert to polygons
    polygons = predictor.prediction_to_polygons(prediction, transform)
    
    # Save results
    predictor.visualize_results(image, prediction, 'prediction_visualization.png')
    predictor.create_interactive_map(polygons, bounds, 'interactive_map.html')
    
    # Save polygons as GeoJSON
    gdf = gpd.GeoDataFrame({'geometry': polygons})
    gdf.to_file('predicted_plantations.geojson', driver='GeoJSON')

if __name__ == '__main__':
    main() 