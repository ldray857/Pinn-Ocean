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
    d_axis = -1
    for idx, s in enumerate(preds.shape):
        if s == D:
            d_axis = idx
            break

    layer_rmse = []
    layer_mae = []
    layer_r2 = []

    for d in range(D):
        sl = [slice(None)] * preds.ndim
        sl[d_axis] = d
        p_layer = preds[tuple(sl)].flatten()
        t_layer = targets[tuple(sl)].flatten()

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

    D = len(depths)
    d_axis = -1
    for idx, s in enumerate(preds.shape):
        if s == D:
            d_axis = idx
            break

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

        sl = [slice(None)] * preds.ndim
        sl[d_axis] = mask
        p_sub = preds[tuple(sl)].flatten()
        t_sub = targets[tuple(sl)].flatten()

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
    # Auto-detect depth axis
    d_axis = -1
    for idx, s in enumerate(temp_t.shape):
        if s == D:
            d_axis = idx
            break

    temp_flat = temp_t.transpose(d_axis, -1).contiguous().reshape(-1, D)
    sal_flat = sal_t.transpose(d_axis, -1).contiguous().reshape(-1, D)

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
    d_axis = -1
    for idx, s in enumerate(temp.shape):
        if s == D:
            d_axis = idx
            break
    temp_flat = np.moveaxis(temp, d_axis, -1).reshape(-1, D)

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

    D = len(depths)
    d_axis = -1
    for idx, s in enumerate(pred_t.shape):
        if s == D:
            d_axis = idx
            break

    if pred_t.ndim == 4:  # (T, D, H, W)
        if d_axis != 1:
            pred_t = np.moveaxis(pred_t, d_axis, 1)
            true_t = np.moveaxis(true_t, d_axis, 1)
        T, D, H, W = pred_t.shape
        pred_mld = np.zeros((T, H, W), dtype=np.float32)
        true_mld = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    pred_mld[t, h, w] = calc_mld(pred_t[t, :, h, w], depths, delta_t=delta_t)
                    true_mld[t, h, w] = calc_mld(true_t[t, :, h, w], depths, delta_t=delta_t)
    elif pred_t.ndim == 3:  # (D, H, W)
        if d_axis != 0:
            pred_t = np.moveaxis(pred_t, d_axis, 0)
            true_t = np.moveaxis(true_t, d_axis, 0)
        D, H, W = pred_t.shape
        pred_mld = np.zeros((H, W), dtype=np.float32)
        true_mld = np.zeros((H, W), dtype=np.float32)
        for h in range(H):
            for w in range(W):
                pred_mld[h, w] = calc_mld(pred_t[:, h, w], depths, delta_t=delta_t)
                true_mld[h, w] = calc_mld(true_t[:, h, w], depths, delta_t=delta_t)
    elif pred_t.ndim == 2:  # (N, D)
        if d_axis != -1 and d_axis != 1:
            pred_t = np.moveaxis(pred_t, d_axis, -1)
            true_t = np.moveaxis(true_t, d_axis, -1)
        N, D = pred_t.shape
        pred_mld = np.zeros(N, dtype=np.float32)
        true_mld = np.zeros(N, dtype=np.float32)
        for i in range(N):
            pred_mld[i] = calc_mld(pred_t[i, :], depths, delta_t=delta_t)
            true_mld[i] = calc_mld(true_t[i, :], depths, delta_t=delta_t)
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


def calc_buoyancy_frequency_metrics(temp, sal, depths, true_temp=None, true_sal=None, n2_tol=-1e-7):
    """
    Computes Brunt-Väisälä buoyancy frequency squared (N^2, s^-2) and pycnocline stratification
    metrics using TEOS-10 local midpoint pressure formulation.

    Args:
        temp: (..., D, H, W) or (N, D) temperature (°C), torch.Tensor or np.ndarray
        sal: (..., D, H, W) or (N, D) salinity (PSU), matching temp
        depths: (D,) depth array (meters)
        true_temp: optional ground truth temperature for comparative error analysis
        true_sal: optional ground truth salinity
        n2_tol: convective instability threshold (default: -1e-7 s^-2)

    Returns:
        dict with CIR, mean N^2, pycnocline depth error metrics, etc.
    """
    from .teos10 import calc_buoyancy_frequency_n2

    if isinstance(temp, torch.Tensor):
        temp_np = temp.detach().cpu().numpy()
    else:
        temp_np = np.asarray(temp, dtype=np.float32)

    if isinstance(sal, torch.Tensor):
        sal_np = sal.detach().cpu().numpy()
    else:
        sal_np = np.asarray(sal, dtype=np.float32)

    if isinstance(depths, torch.Tensor):
        depths_np = depths.detach().cpu().numpy()
    else:
        depths_np = np.asarray(depths, dtype=np.float32)

    D = len(depths_np)
    # Find depth axis
    d_axis = -1
    for idx, s in enumerate(temp_np.shape):
        if s == D:
            d_axis = idx
            break

    n2_pred = calc_buoyancy_frequency_n2(sal_np, temp_np, depths_np, depth_axis=d_axis)
    z_mid = 0.5 * (depths_np[:-1] + depths_np[1:])

    pred_unstable = n2_pred < n2_tol
    total_eval = int(pred_unstable.size)
    pred_inversions = int(np.sum(pred_unstable))
    cir_pred_pct = float(pred_inversions / max(total_eval, 1) * 100.0)
    mean_n2_pred = float(np.mean(n2_pred))

    # Pycnocline depth (arg max N^2 along depth axis)
    pyc_idx_pred = np.argmax(n2_pred, axis=d_axis)
    pyc_depth_pred = z_mid[pyc_idx_pred]
    pyc_peak_pred = np.max(n2_pred, axis=d_axis)

    metrics = {
        "cir_pred_percent": cir_pred_pct,
        "pred_inversions": pred_inversions,
        "total_evaluated": total_eval,
        "mean_n2_pred": mean_n2_pred,
        "pred_n2": n2_pred,
        "pred_pycnocline_depth": pyc_depth_pred,
        "pred_pycnocline_peak": pyc_peak_pred
    }

    if true_temp is not None and true_sal is not None:
        if isinstance(true_temp, torch.Tensor):
            true_t_np = true_temp.detach().cpu().numpy()
        else:
            true_t_np = np.asarray(true_temp, dtype=np.float32)
        if isinstance(true_sal, torch.Tensor):
            true_s_np = true_sal.detach().cpu().numpy()
        else:
            true_s_np = np.asarray(true_sal, dtype=np.float32)

        n2_true = calc_buoyancy_frequency_n2(true_s_np, true_t_np, depths_np, depth_axis=d_axis)
        true_unstable = n2_true < n2_tol
        true_inversions = int(np.sum(true_unstable))
        cir_true_pct = float(true_inversions / max(total_eval, 1) * 100.0)
        mean_n2_true = float(np.mean(n2_true))

        pyc_idx_true = np.argmax(n2_true, axis=d_axis)
        pyc_depth_true = z_mid[pyc_idx_true]
        pyc_peak_true = np.max(n2_true, axis=d_axis)

        p_depth_flat = pyc_depth_pred.flatten()
        t_depth_flat = pyc_depth_true.flatten()
        p_peak_flat = pyc_peak_pred.flatten()
        t_peak_flat = pyc_peak_true.flatten()

        metrics.update({
            "cir_true_percent": cir_true_pct,
            "true_inversions": true_inversions,
            "mean_n2_true": mean_n2_true,
            "pycnocline_depth_rmse": float(calc_rmse(p_depth_flat, t_depth_flat)),
            "pycnocline_depth_mae": float(calc_mae(p_depth_flat, t_depth_flat)),
            "pycnocline_peak_rmse": float(calc_rmse(p_peak_flat, t_peak_flat)),
            "true_n2": n2_true,
            "true_pycnocline_depth": pyc_depth_true,
            "true_pycnocline_peak": pyc_peak_true
        })

    return metrics


def calc_multilevel_density_inversion_rate(temp, sal, depths, split_depth=500.0, tol=1e-5):
    """
    Calculates Multi-Level Potential Density Inversion Rate.
    Uses surface-referenced sigma_0 (p_ref=0 dbar) for upper layers (z <= split_depth)
    and intermediate-referenced sigma_1 (p_ref=1000 dbar) for deep layers (z > split_depth).
    This eliminates thermobaric fictitious inversions caused by single surface reference pressure.

    Args:
        temp: (..., D, H, W) or (N, D) temperature (°C)
        sal: (..., D, H, W) or (N, D) salinity (PSU)
        depths: (D,) depth array (meters)
        split_depth: transition depth in meters (default: 500m)
        tol: numerical tolerance (default: 1e-5 kg/m^3/m)

    Returns:
        dict with:
            'overall_inversion_rate_percent': multi-level potential density inversion rate (%)
            'sigma0_inversion_rate_percent': upper layer inversion rate (%)
            'sigma1_inversion_rate_percent': deep layer inversion rate (%)
            'total_inversions': count
            'total_evaluated': total vertical gradient points
    """
    from .teos10 import calc_potential_density_sigma

    if isinstance(temp, torch.Tensor):
        temp_np = temp.detach().cpu().numpy()
    else:
        temp_np = np.asarray(temp, dtype=np.float32)
    if isinstance(sal, torch.Tensor):
        sal_np = sal.detach().cpu().numpy()
    else:
        sal_np = np.asarray(sal, dtype=np.float32)
    if isinstance(depths, torch.Tensor):
        depths_np = depths.detach().cpu().numpy()
    else:
        depths_np = np.asarray(depths, dtype=np.float32)

    D = len(depths_np)
    d_axis = -1
    for idx, s in enumerate(temp_np.shape):
        if s == D:
            d_axis = idx
            break
    temp_flat = np.moveaxis(temp_np, d_axis, -1).reshape(-1, D)
    sal_flat = np.moveaxis(sal_np, d_axis, -1).reshape(-1, D)

    # Calculate sigma_0 and sigma_1 across all profiles
    sigma_0 = calc_potential_density_sigma(sal_flat, temp_flat, p_ref=0.0)
    sigma_1 = calc_potential_density_sigma(sal_flat, temp_flat, p_ref=1000.0)

    dz = depths_np[1:] - depths_np[:-1]

    # Gradient for sigma_0 in upper region: z < split_depth
    dz_upper_mask = (depths_np[:-1] < split_depth)
    d_sigma0 = sigma_0[:, 1:] - sigma_0[:, :-1]
    d_sigma0_dz = d_sigma0[:, dz_upper_mask] / dz[dz_upper_mask]
    inv_s0 = (d_sigma0_dz < -tol)
    s0_inv_cnt = int(np.sum(inv_s0))
    s0_total = int(inv_s0.size)

    # Gradient for sigma_1 in lower region: z >= split_depth
    dz_lower_mask = (depths_np[:-1] >= split_depth)
    if np.any(dz_lower_mask):
        d_sigma1 = sigma_1[:, 1:] - sigma_1[:, :-1]
        d_sigma1_dz = d_sigma1[:, dz_lower_mask] / dz[dz_lower_mask]
        inv_s1 = (d_sigma1_dz < -tol)
        s1_inv_cnt = int(np.sum(inv_s1))
        s1_total = int(inv_s1.size)
    else:
        s1_inv_cnt = 0
        s1_total = 0

    total_inversions = s0_inv_cnt + s1_inv_cnt
    total_evaluated = s0_total + s1_total
    overall_rate = float(total_inversions / max(total_evaluated, 1) * 100.0)
    s0_rate = float(s0_inv_cnt / max(s0_total, 1) * 100.0)
    s1_rate = float(s1_inv_cnt / max(s1_total, 1) * 100.0) if s1_total > 0 else 0.0

    return {
        "overall_inversion_rate_percent": overall_rate,
        "sigma0_inversion_rate_percent": s0_rate,
        "sigma1_inversion_rate_percent": s1_rate,
        "total_inversions": total_inversions,
        "total_evaluated": total_evaluated
    }


def calc_psnr(preds, targets, data_range=None):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher is better.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    mse = float(np.mean((preds - targets) ** 2))
    if mse == 0.0:
        return 100.0

    if data_range is None:
        data_range = float(np.max(targets) - np.min(targets))
        if data_range < 1e-4:
            data_range = 1.0

    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def calc_gradient_fidelity(preds, targets):
    """
    Computes spatial gradient magnitude correlation and gradient RMSE.
    Evaluates preservation of frontal shear and mesoscale eddy edges.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    gy_t, gx_t = np.gradient(targets, axis=(-2, -1))
    gy_p, gx_p = np.gradient(preds, axis=(-2, -1))

    mag_t = np.sqrt(gx_t**2 + gy_t**2)
    mag_p = np.sqrt(gx_p**2 + gy_p**2)

    grad_rmse = float(np.sqrt(np.mean((mag_p - mag_t) ** 2)))
    corr = np.corrcoef(mag_t.ravel(), mag_p.ravel())[0, 1]
    grad_corr = float(corr) if not np.isnan(corr) else 1.0

    return {
        "grad_rmse": grad_rmse,
        "grad_correlation": grad_corr
    }


def calc_model_superiority_index(rmse_t, rmse_s, r2_t, r2_s, cir_percent, tmv_percent, mld_mae):
    """
    Computes a normalized composite Superiority Score (0-100) combining
    statistical accuracy, physical compliance, and boundary fidelity.
    
    Higher score indicates greater superiority.
    """
    # Normalized components (bounded [0, 1])
    # Temperature accuracy score: 1.0 at RMSE=0, drops at higher RMSE
    s_t = max(0.0, 1.0 - (rmse_t / 3.0))
    # Salinity accuracy score
    s_s = max(0.0, 1.0 - (rmse_s / 0.3))
    # Correlation score
    s_r2 = max(0.0, (max(0.0, r2_t) + max(0.0, r2_s)) / 2.0)
    # Convective stability score: 1.0 at 0% instability, 0 at >=15%
    s_stab = max(0.0, 1.0 - (cir_percent / 15.0))
    # Deep thermal monotonicity score: 1.0 at 0% violation, 0 at >=5%
    s_mono = max(0.0, 1.0 - (tmv_percent / 5.0))
    # MLD boundary score: 1.0 at 0m MAE, 0 at >=50m
    s_mld = max(0.0, 1.0 - (mld_mae / 50.0))

    composite_score = (
        0.20 * s_t +
        0.20 * s_s +
        0.15 * s_r2 +
        0.20 * s_stab +
        0.15 * s_mono +
        0.10 * s_mld
    ) * 100.0

    return {
        "composite_score": float(composite_score),
        "score_t": float(s_t * 100.0),
        "score_s": float(s_s * 100.0),
        "score_r2": float(s_r2 * 100.0),
        "score_stratification": float(s_stab * 100.0),
        "score_monotonicity": float(s_mono * 100.0),
        "score_mld": float(s_mld * 100.0)
    }

