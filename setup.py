# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="pinn_ocean",
    version="1.0.0",
    author="Lei Di",
    author_email="3066443513@qq.com",
    description="Coupled PINN and Swin-Transformer Architecture for 3-D Ocean Thermohaline Reconstruction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ldray857/Pinn-Ocean",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "xarray>=2023.1.0",
        "netCDF4>=1.6.0",
        "scipy>=1.10.0",
        "einops>=0.6.0",
        "copernicusmarine>=1.0.0",
    ],
)
