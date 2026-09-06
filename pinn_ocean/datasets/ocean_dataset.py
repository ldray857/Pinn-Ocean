# -*- coding: utf-8 -*-
"""
Dataset module for Pinn-Ocean
Loads multi-source satellite observations and 3-D ocean reanalysis data from NetCDF files.
"""

import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset


class OceanContinuousDataset(Dataset):
    """
    Pacific Ocean Thermohaline Dataset Loader:
    Inputs (8 channels):
        0: SST (Sea Surface Temperature)
        1: SLA (Sea Level Anomaly)
        2: SSS (Sea Surface Salinity)
        3: Wind U (Zonal wind component)
        4: Wind V (Meridional wind component)
        5: Longitude (Normalized coordinate)
        6: Latitude (Normalized coordinate)
        7: Month (Cyclic time encoding)
    Labels (2 channels, 3-D):
        0: Potential Temperature (0-1000m)
        1: Practical Salinity (0-1000m)
    """
    def __init__(self, sla_path, gt_path, mode='train', train_ratio=0.8, val_ratio=0.15):
        super().__init__()
        self.mode = mode
        
        if not (os.path.exists(sla_path) and os.path.exists(gt_path)):
            raise FileNotFoundError(
                f"Data files not found at {sla_path} or {gt_path}. Please check data path configuration."
            )

        # 1. Load NetCDF datasets
        self.sla_ds_full = xr.open_dataset(sla_path)
        self.gt_ds_full = xr.open_dataset(gt_path)

        total_months = len(self.gt_ds_full.time)
        n_train = int(total_months * train_ratio)
        n_val = int(total_months * val_ratio)

        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, total_months)

        # 2. Compute normalization stats strictly from training partition
        train_gt = self.gt_ds_full.isel(time=train_idx)
        train_sla = self.sla_ds_full.isel(time=train_idx)

        train_sla_aligned = train_sla.interp(
            longitude=train_gt.longitude,
            latitude=train_gt.latitude,
            method="linear"
        )

        temp_train = np.nan_to_num(train_gt.thetao.values, nan=0.0)
        sal_train = np.nan_to_num(train_gt.so.values, nan=0.0)
        sla_train = np.nan_to_num(train_sla_aligned.sla.values, nan=0.0)

        self.stats = {
            'mean_sst': float(temp_train[:, 0, :, :].mean()),
            'std_sst': float(temp_train[:, 0, :, :].std() + 1e-6),
            'mean_sla': float(sla_train.mean()),
            'std_sla': float(sla_train.std() + 1e-6),
            'mean_sss': float(sal_train[:, 0, :, :].mean()),
            'std_sss': float(sal_train[:, 0, :, :].std() + 1e-6),
            'mean_t': float(temp_train.mean()),
            'std_t': float(temp_train.std() + 1e-6),
            'mean_s': float(sal_train.mean()),
            'std_s': float(sal_train.std() + 1e-6),
        }

        # 3. Extract subset for specified mode
        current_idx = train_idx if mode == 'train' else (val_idx if mode == 'val' else test_idx)
        self.gt_ds = self.gt_ds_full.isel(time=current_idx)
        self.sla_ds = self.sla_ds_full.isel(time=current_idx)

        self.sla_ds_aligned = self.sla_ds.interp(
            longitude=self.gt_ds.longitude,
            latitude=self.gt_ds.latitude,
            method="linear"
        )

        self.sla_raw = np.nan_to_num(self.sla_ds_aligned.sla.values, nan=0.0)
        temp_all = np.nan_to_num(self.gt_ds.thetao.values, nan=0.0)
        sal_all = np.nan_to_num(self.gt_ds.so.values, nan=0.0)

        self.sst_raw = temp_all[:, 0, :, :]
        self.sss_raw = sal_all[:, 0, :, :]
        self.depths = self.gt_ds.depth.values
        self.times = self.gt_ds.time.values

        # 4. Normalized Spatial Coordinates
        lon_vals = self.gt_ds.longitude.values
        lat_vals = self.gt_ds.latitude.values
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        self.lon_norm = (lon_grid - lon_vals.min()) / (lon_vals.max() - lon_vals.min() + 1e-6)
        self.lat_norm = (lat_grid - lat_vals.min()) / (lat_vals.max() - lat_vals.min() + 1e-6)

        # 5. Normalized Labels
        self.temp_norm = (temp_all - self.stats['mean_t']) / self.stats['std_t']
        self.sal_norm = (sal_all - self.stats['mean_s']) / self.stats['std_s']

        # 6. Normalized Cyclic Month Encoding
        self.months_norm = np.array([
            float(t.astype('datetime64[M]').astype(int) % 12 + 1) / 12.0
            for t in self.times
        ])

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        sst = (self.sst_raw[idx] - self.stats['mean_sst']) / self.stats['std_sst']
        sla = (self.sla_raw[idx] - self.stats['mean_sla']) / self.stats['std_sla']
        sss = (self.sss_raw[idx] - self.stats['mean_sss']) / self.stats['std_sss']

        # Placeholder for real wind observations if not yet extracted
        wind_u = np.zeros_like(sla)
        wind_v = np.zeros_like(sla)
        lon = self.lon_norm
        lat = self.lat_norm
        month = np.full_like(sla, self.months_norm[idx])

        # Stack into 8-channel 2D input
        x_8ch = np.stack([sst, sla, sss, wind_u, wind_v, lon, lat, month], axis=0)
        # Stack into 2-channel 3D output: [Temp, Sal]
        y_3d = np.stack([self.temp_norm[idx], self.sal_norm[idx]], axis=0)

        return torch.tensor(x_8ch, dtype=torch.float32), torch.tensor(y_3d, dtype=torch.float32)

    def get_depth_tensor(self):
        return torch.tensor(self.depths, dtype=torch.float32)
