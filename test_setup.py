import sys
import os
import unittest

class TestSetup(unittest.TestCase):
    def test_dependencies(self):
        """Test if all required packages are installed"""
        required_packages = [
            'geopandas',
            'matplotlib',
            'seaborn',
            'folium',
            'pandas',
            'numpy',
            'plotly',
            'contextily',
            'rasterio',
            'torch',
            'earthengine-api',
            'sentinelhub'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                self.fail(f"Required package '{package}' is not installed")
    
    def test_data_files(self):
        """Test if required data files exist"""
        required_files = [
            'data/Plantations Data.geojson'
        ]
        
        for file in required_files:
            self.assertTrue(os.path.exists(file), f"Required file '{file}' not found")
    
    def test_credentials(self):
        """Test if credentials are properly set"""
        required_env_vars = [
            'SENTINEL_HUB_CLIENT_ID',
            'SENTINEL_HUB_CLIENT_SECRET'
        ]
        
        for var in required_env_vars:
            self.assertIn(var, os.environ, f"Environment variable '{var}' not set")

if __name__ == '__main__':
    unittest.main() 