from setuptools import setup, find_packages

setup(
    name="plantation-monitoring",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "pandas==1.5.3",
        "geopandas==0.12.2",
        "torch==2.0.1",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "folium==0.14.0",
        "plotly==5.14.1",
        "contextily==1.3.0",
        "rasterio==1.3.7",
        "earthengine-api==0.1.341",
        "wandb",
        "tqdm",
        "tensorboard"
    ]
) 