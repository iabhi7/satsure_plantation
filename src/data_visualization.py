import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import box
import contextily as ctx
import os

class PlantationDataVisualizer:
    def __init__(self, geojson_path):
        """Initialize the visualizer with the GeoJSON data"""
        self.gdf = gpd.read_file(geojson_path)
        self.output_dir = os.path.join('outputs', 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Convert date columns
        self.gdf['PLANTING DATE'] = pd.to_datetime(self.gdf['PLANTING DATE'])
        self.gdf['COPPICING DATE'] = pd.to_datetime(self.gdf['COPPICING DATE'])
        
        # Handle NaN values
        self.gdf = self.gdf.fillna({
            'PLOT AREA': 0,
            'AVG HEIGHT': 0,
            'PLTN GROWTH': 'Unknown',
            'Row to Row SPACING': 0,
            'Plant to Plant SPACING': 0
        })
    
    def plot_spatial_distribution(self, save_path=None):
        """Plot spatial distribution of plantations"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot base map
        self.gdf.plot(ax=ax, alpha=0.5, column='PLOT AREA', 
                     legend=True, legend_kwds={'label': 'Plot Area (ha)'})
        
        # Add contextual basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        
        ax.set_title('Spatial Distribution of Plantations')
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_interactive_map(self, save_path=None):
        """Create interactive folium map"""
        # Calculate center point
        center_lat = self.gdf.geometry.centroid.y.mean()
        center_lon = self.gdf.geometry.centroid.x.mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add plantations to map
        for idx, row in self.gdf.iterrows():
            popup_html = f"""
                <b>Farmer:</b> {row['FARMER NAME']}<br>
                <b>Area:</b> {row['PLOT AREA']} ha<br>
                <b>Growth:</b> {row['PLTN GROWTH']}<br>
                <b>Planting Date:</b> {row['PLANTING DATE'].strftime('%Y-%m-%d')}<br>
                <b>Height:</b> {row['AVG HEIGHT']} m
            """
            
            folium.GeoJson(
                row.geometry.__geo_interface__,
                popup=folium.Popup(popup_html, max_width=300),
                style_function=lambda x: {
                    'fillColor': '#00ff00',
                    'color': '#000000',
                    'weight': 1,
                    'fillOpacity': 0.5
                }
            ).add_to(m)
        
        if save_path:
            m.save(os.path.join(self.output_dir, save_path))
    
    def plot_temporal_distribution(self, save_path=None):
        """Plot temporal distribution of plantations"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Planting timeline
        self.gdf['PLANTING DATE'].dt.year.value_counts().sort_index().plot(
            kind='bar', ax=ax1, color='green'
        )
        ax1.set_title('Plantation Timeline by Year')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Plantations')
        
        # Growth progression
        sns.boxplot(
            data=self.gdf,
            x=self.gdf['PLANTING DATE'].dt.year,
            y='AVG HEIGHT',
            ax=ax2
        )
        ax2.set_title('Height Distribution by Planting Year')
        ax2.set_xlabel('Planting Year')
        ax2.set_ylabel('Average Height (m)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_growth_analysis(self, save_path=None):
        """Plot growth analysis"""
        fig = plt.figure(figsize=(20, 10))
        
        # Growth quality distribution
        ax1 = plt.subplot(221)
        growth_counts = self.gdf['PLTN GROWTH'].value_counts()
        growth_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax1)
        ax1.set_title('Growth Quality Distribution')
        
        # Height distribution
        ax2 = plt.subplot(222)
        sns.histplot(data=self.gdf, x='AVG HEIGHT', bins=30, ax=ax2)
        ax2.set_title('Height Distribution')
        
        # Area vs Height
        ax3 = plt.subplot(223)
        sns.scatterplot(data=self.gdf, x='PLOT AREA', y='AVG HEIGHT', 
                       hue='PLTN GROWTH', ax=ax3)
        ax3.set_title('Area vs Height')
        
        # Growth by District
        ax4 = plt.subplot(224)
        district_growth = pd.crosstab(self.gdf['DISTRICT'], self.gdf['PLTN GROWTH'])
        district_growth.plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('Growth Quality by District')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_dashboard(self, save_path=None):
        """Create interactive dashboard using plotly"""
        # Create subplots
        fig = go.Figure()
        
        # Add temporal analysis
        temporal_data = self.gdf['PLANTING DATE'].dt.year.value_counts().sort_index()
        fig.add_trace(go.Bar(
            x=temporal_data.index,
            y=temporal_data.values,
            name='Plantations by Year'
        ))
        
        # Add growth distribution
        growth_data = self.gdf['PLTN GROWTH'].value_counts()
        fig.add_trace(go.Pie(
            labels=growth_data.index,
            values=growth_data.values,
            name='Growth Distribution'
        ))
        
        # Update layout
        fig.update_layout(
            title='Plantation Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(os.path.join(self.output_dir, save_path))
    
    def generate_summary_statistics(self):
        """Generate summary statistics of the plantation data"""
        summary = {
            'Total Plantations': len(self.gdf),
            'Total Area': self.gdf['PLOT AREA'].sum(),
            'Average Plot Size': self.gdf['PLOT AREA'].mean(),
            'Number of Districts': self.gdf['DISTRICT'].nunique(),
            'Number of Farmers': self.gdf['FARMER NAME'].nunique(),
            'Most Common Species': self.gdf['TREE SPECIES'].mode().iloc[0],
            'Average Height': self.gdf['AVG HEIGHT'].mean(),
            'Growth Quality Distribution': self.gdf['PLTN GROWTH'].value_counts().to_dict()
        }
        return pd.Series(summary)

def main():
    print("test")
    # Initialize visualizer
    visualizer = PlantationDataVisualizer('data/Plantations Data.geojson')
    
    # Generate all visualizations
    print("Generating visualizations...")
    visualizer.plot_spatial_distribution('spatial_distribution.png')
    visualizer.create_interactive_map('interactive_map.html')
    visualizer.plot_temporal_distribution('temporal_distribution.png')
    visualizer.plot_growth_analysis('growth_analysis.png')
    visualizer.create_dashboard('dashboard.html')
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(visualizer.generate_summary_statistics())

if __name__ == "__main__":
    main() 