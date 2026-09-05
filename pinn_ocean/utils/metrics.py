# -*- coding: utf-8 -*-
"""
Evaluation metrics for 3-D Ocean Thermohaline Reconstruction
Includes RMSE, MAE, R2 score, and oceanographic Mixed Layer Depth (MLD).
"""

import numpy as np
import torch


def calc_rmse(preds, targets):
    """Root Mean Squared Error"""
    if isinstance(preds, torch.Tensor):
        return torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    return np.sqrt(np.mean((preds - targets) ** 2))


def calc_mae(preds, targets):
    """Mean Absolute Error"""
    if isinstance(preds, torch.Tensor):
        return torch.mean(torch.abs(preds - targets)).item()
    return np.mean(np.abs(preds - targets))


def calc_r2(preds, targets):
    """Coefficient of Determination (R^2 Score)"""
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def calc_mld(temp_profile, depths, delta_t=0.5):
    """
    Compute Mixed Layer Depth (MLD) defined as the depth where
    temperature decreases by delta_t (typically 0.5 deg C) from surface.
    
    Args:
        temp_profile: 1D array of temperature profile from surface down to depth
        depths: 1D array of corresponding depths
        delta_t: threshold difference from surface temperature (default: 0.5 deg C)
        
    Returns:
        mld: Estimated depth of mixed layer in meters
    """
    if isinstance(temp_profile, torch.Tensor):
        temp_profile = temp_profile.detach().cpu().numpy()
    if isinstance(depths, torch.Tensor):
        depths = depths.detach().cpu().numpy()
        
    t_surf = temp_profile[0]
    threshold = t_surf - delta_t
    
    for i in range(1, len(temp_profile)):
        if temp_profile[i] <= threshold:
            # Linear interpolation
            t0, t1 = temp_profile[i - 1], temp_profile[i]
            z0, z1 = depths[i - 1], depths[i]
            if t1 == t0:
                return z0
            mld = z0 + (threshold - t0) * (z1 - z0) / (t1 - t0)
            return float(mld)
            
    return float(depths[-1])
