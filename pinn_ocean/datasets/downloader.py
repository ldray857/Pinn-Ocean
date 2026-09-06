# -*- coding: utf-8 -*-
"""
Downloader module for Pinn-Ocean
Encapsulates CMEMS data subsetting API calls for programmatic usage.
"""

import os
from typing import Dict, Any, Optional, List


def download_pacific_dataset(
    dataset_id: str,
    variables: List[str],
    output_filename: str,
    output_dir: str = "data",
    min_lon: float = 145.0,
    max_lon: float = 165.0,
    min_lat: float = 30.0,
    max_lat: float = 40.0,
    start_time: str = "2013-01-01",
    end_time: str = "2021-12-31",
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    dry_run: bool = False
) -> bool:
    """
    Subsets and downloads a specific CMEMS NetCDF dataset for the Open Pacific region.
    
    Args:
        dataset_id: CMEMS dataset identifier
        variables: List of physical variables to extract
        output_filename: Name of target .nc file
        output_dir: Local destination folder
        min_lon, max_lon: Longitude bounding box (degrees east)
        min_lat, max_lat: Latitude bounding box (degrees north)
        start_time, end_time: Temporal slice (YYYY-MM-DD)
        min_depth, max_depth: Depth range in meters (for 3-D products)
        dry_run: If True, prints parameters without downloading
        
    Returns:
        bool: True if download succeeded or file already exists, False otherwise.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(target_path):
        print(f"[Info] File already exists: {target_path}")
        return True
        
    kwargs = {
        "dataset_id": dataset_id,
        "variables": variables,
        "minimum_longitude": min_lon,
        "maximum_longitude": max_lon,
        "minimum_latitude": min_lat,
        "maximum_latitude": max_lat,
        "start_datetime": start_time,
        "end_datetime": end_time,
        "output_directory": output_dir,
        "output_filename": output_filename,
        "overwrite": False
    }
    
    if min_depth is not None and max_depth is not None:
        kwargs["minimum_depth"] = min_depth
        kwargs["maximum_depth"] = max_depth

    if dry_run:
        print(f"[Dry Run] Prepared download kwargs for {dataset_id}: {kwargs}")
        return True
        
    try:
        import copernicusmarine
        copernicusmarine.subset(**kwargs)
        return True
    except ImportError:
        raise ImportError(
            "The 'copernicusmarine' library is required to download data from CMEMS. "
            "Please install it via 'pip install copernicusmarine'."
        )
    except Exception as e:
        print(f"[Error] Failed downloading {dataset_id}: {e}")
        return False
