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


def calc_layer_metrics(preds, targets, depths):
    """
    Computes layer-by-layer RMSE, MAE, and R^2 across standard vertical depth coordinates.

    Args:
        preds: (..., D, H, W) or (..., D) predicted field
        targets: matching ground truth array
        depths: (D,) 1D array of depths

    Returns:
        dict with keys: 'depths', 'rmse', 'mae', 'r2' (all length D lists)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(depths, torch.Tensor):
        depths = depths.detach().cpu().numpy()

    D = len(depths)
    # Ensure depth axis is accessible: find axis matching D
    shape = preds.shape
    if len(shape) == 4:  # (T, D, H, W)
        d_axis = 1
    elif len(shape) == 3:  # (D, H, W)
        d_axis = 0
    else:
        d_axis = -1

    layer_rmse = []
    layer_mae = []
    layer_r2 = []

    for d in range(D):
        if d_axis == 1:
            p_layer = preds[:, d, :, :].flatten()
            t_layer = targets[:, d, :, :].flatten()
        elif d_axis == 0:
            p_layer = preds[d, :, :].flatten()
            t_layer = targets[d, :, :].flatten()
        else:
            p_layer = preds[..., d].flatten()
            t_layer = targets[..., d].flatten()

        layer_rmse.append(float(calc_rmse(p_layer, t_layer)))
        layer_mae.append(float(calc_mae(p_layer, t_layer)))
        layer_r2.append(float(calc_r2(p_layer, t_layer)))

    return {
        "depths": [float(z) for z in depths],
        "rmse": layer_rmse,
        "mae": layer_mae,
        "r2": layer_r2
    }


def calc_regime_metrics(preds, targets, depths):
    """
    Computes aggregated performance across three key oceanographic regimes:
    1. Mixed Layer (混合层): 0 <= z <= 100m
    2. Thermocline / Halocline (主跃层): 100m < z <= 400m
    3. Intermediate & Deep Layer (中深层): 400m < z <= 1000m

    Returns:
        dict mapping regime names to {'rmse', 'mae', 'r2', 'depth_range'}
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(depths, torch.Tensor):
        depths = depths.detach().cpu().numpy()

    # Determine depth axis
    if preds.ndim == 4:
        d_axis = 1
    elif preds.ndim == 3:
        d_axis = 0
    else:
        d_axis = -1

    regimes = {
        "mixed_layer": (0.0, 100.0, "0-100m (混合层)"),
        "thermocline": (100.0, 400.0, "100-400m (主跃层)"),
        "deep_layer": (400.0, 1005.0, "400-1000m (中深层)")
    }

    results = {}
    for key, (z_min, z_max, label) in regimes.items():
        mask = (depths >= z_min) & (depths <= z_max)
        if not np.any(mask):
            continue

        if d_axis == 1:
            p_sub = preds[:, mask, :, :].flatten()
            t_sub = targets[:, mask, :, :].flatten()
        elif d_axis == 0:
            p_sub = preds[mask, :, :].flatten()
            t_sub = targets[mask, :, :].flatten()
        else:
            p_sub = preds[..., mask].flatten()
            t_sub = targets[..., mask].flatten()

        results[key] = {
            "label": label,
            "depth_range": f"{int(z_min)}-{int(z_max)}m",
            "rmse": float(calc_rmse(p_sub, t_sub)),
            "mae": float(calc_mae(p_sub, t_sub)),
            "r2": float(calc_r2(p_sub, t_sub))
        }

    return results


def calc_density_inversion_rate(temp, sal, depths, tol=1e-5):
    """
    Calculates Density Inversion Rate (DIR) based on UNESCO/TEOS-10 equation of state.
    Evaluates whether water column satisfies gravitational stability (d_rho / dz >= 0).

    Args:
        temp: (..., D, H, W) temperature in deg C
        sal: (..., D, H, W) salinity in PSU
        depths: (D,) depth in meters
        tol: numerical tolerance for floating-point inversion (default: 1e-5 kg/m^3/m)

    Returns:
        dict with:
            'inversion_rate_percent': percentage of unstable gradient points (%)
            'total_inversions': number of violating layer points
            'total_evaluated': total number of vertical gradient points evaluated
    """
    from .teos10 import approx_seawater_density

    if not isinstance(temp, torch.Tensor):
        temp_t = torch.from_numpy(np.asarray(temp, dtype=np.float32))
    else:
        temp_t = temp.float()

    if not isinstance(sal, torch.Tensor):
        sal_t = torch.from_numpy(np.asarray(sal, dtype=np.float32))
    else:
        sal_t = sal.float()

    if not isinstance(depths, torch.Tensor):
        depths_t = torch.from_numpy(np.asarray(depths, dtype=np.float32))
    else:
        depths_t = depths.float()

    D = len(depths_t)
    orig_shape = temp_t.shape

    # Find depth dimension
    if len(orig_shape) == 4:
        # (T, D, H, W) -> permute to (T, H, W, D)
        temp_flat = temp_t.permute(0, 2, 3, 1).reshape(-1, D)
        sal_flat = sal_t.permute(0, 2, 3, 1).reshape(-1, D)
    elif len(orig_shape) == 3:
        # (D, H, W) -> permute to (H, W, D)
        temp_flat = temp_t.permute(1, 2, 0).reshape(-1, D)
        sal_flat = sal_t.permute(1, 2, 0).reshape(-1, D)
    else:
        temp_flat = temp_t.reshape(-1, D)
        sal_flat = sal_t.reshape(-1, D)

    z_broadcast = depths_t.view(1, D).expand(temp_flat.shape[0], -1)

    with torch.no_grad():
        rho = approx_seawater_density(sal_flat, temp_flat, z_broadcast)  # (N, D)
        # Vertical gradient: d_rho / dz
        dz = depths_t[1:] - depths_t[:-1]  # (D-1,)
        d_rho = rho[:, 1:] - rho[:, :-1]   # (N, D-1)
        d_rho_dz = d_rho / dz.view(1, -1)

        # Inversion occurs when density decreases with depth (lighter water below heavier water)
        # Stable: d_rho_dz >= 0; Inversion: d_rho_dz < -tol
        inversions = (d_rho_dz < -tol)
        total_inversions = int(inversions.sum().item())
        total_evaluated = int(inversions.numel())
        rate_pct = (total_inversions / max(total_evaluated, 1)) * 100.0

    return {
        "inversion_rate_percent": float(rate_pct),
        "total_inversions": total_inversions,
        "total_evaluated": total_evaluated
    }


def calc_temp_monotonicity_violation(temp, depths, start_depth=100.0, tol=0.01):
    """
    Computes violation rate of temperature vertical monotonicity (dT/dz <= 0)
    in the open ocean below the mixed layer (default: z >= 100m).

    Returns:
        dict with:
            'violation_rate_percent': percentage of non-physical warming with depth (%)
            'violations': count
            'total': total evaluated
    """
    if isinstance(temp, torch.Tensor):
        temp = temp.detach().cpu().numpy()
    if isinstance(depths, torch.Tensor):
        depths = depths.detach().cpu().numpy()

    D = len(depths)
    if temp.ndim == 4:
        temp_flat = np.moveaxis(temp, 1, -1).reshape(-1, D)
    elif temp.ndim == 3:
        temp_flat = np.moveaxis(temp, 0, -1).reshape(-1, D)
    else:
        temp_flat = temp.reshape(-1, D)

    # Filter depth range
    sub_mask = depths[:-1] >= start_depth
    if not np.any(sub_mask):
        return {"violation_rate_percent": 0.0, "violations": 0, "total": 0}

    dz = depths[1:] - depths[:-1]
    dt = temp_flat[:, 1:] - temp_flat[:, :-1]
    dt_dz = dt[:, sub_mask] / dz[sub_mask]

    # Warming with depth by more than tol deg C/m
    viol = dt_dz > tol
    violations = int(np.sum(viol))
    total = int(viol.size)
    rate_pct = (violations / max(total, 1)) * 100.0

    return {
        "violation_rate_percent": float(rate_pct),
        "violations": violations,
        "total": total
    }


def calc_domain_mld_metrics(pred_t, true_t, depths, delta_t=0.5):
    """
    Computes Mixed Layer Depth (MLD) across all spatial grid columns in the domain,
    and returns comprehensive MLD error statistics.

    Returns:
        dict with:
            'mld_rmse': RMSE of MLD in meters
            'mld_mae': MAE of MLD in meters
            'mld_r2': R^2 score of MLD
            'pred_mld': 2D/3D array of predicted MLD
            'true_mld': 2D/3D array of ground truth MLD
    """
    if isinstance(pred_t, torch.Tensor):
        pred_t = pred_t.detach().cpu().numpy()
    if isinstance(true_t, torch.Tensor):
        true_t = true_t.detach().cpu().numpy()
    if isinstance(depths, torch.Tensor):
        depths = depths.detach().cpu().numpy()

    if pred_t.ndim == 4:  # (T, D, H, W)
        T, D, H, W = pred_t.shape
        pred_mld = np.zeros((T, H, W), dtype=np.float32)
        true_mld = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    pred_mld[t, h, w] = calc_mld(pred_t[t, :, h, w], depths, delta_t=delta_t)
                    true_mld[t, h, w] = calc_mld(true_t[t, :, h, w], depths, delta_t=delta_t)
    elif pred_t.ndim == 3:  # (D, H, W)
        D, H, W = pred_t.shape
        pred_mld = np.zeros((H, W), dtype=np.float32)
        true_mld = np.zeros((H, W), dtype=np.float32)
        for h in range(H):
            for w in range(W):
                pred_mld[h, w] = calc_mld(pred_t[:, h, w], depths, delta_t=delta_t)
                true_mld[h, w] = calc_mld(true_t[:, h, w], depths, delta_t=delta_t)
    else:
        raise ValueError(f"Unsupported pred_t dimension: {pred_t.ndim}")

    p_flat = pred_mld.flatten()
    t_flat = true_mld.flatten()

    return {
        "mld_rmse": float(calc_rmse(p_flat, t_flat)),
        "mld_mae": float(calc_mae(p_flat, t_flat)),
        "mld_r2": float(calc_r2(p_flat, t_flat)),
        "pred_mld": pred_mld,
        "true_mld": true_mld
    }

