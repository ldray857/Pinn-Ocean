# -*- coding: utf-8 -*-
"""
Differentiable Seawater Equation of State (TEOS-10 / UNESCO approximation)
Provides analytical, differentiable computation of seawater in-situ density
given salinity, potential temperature, and pressure/depth.
"""

import torch


def approx_seawater_density(sal, temp, depth):
    """
    Differentiable approximation of seawater density (kg/m^3).
    Based on standard oceanographic polynomial equation of state (UNESCO 1980 / TEOS-10 polynomial).
    
    Args:
        sal: Salinity in PSU / g/kg, torch.Tensor
        temp: Temperature in deg C, torch.Tensor
        depth: Depth in meters (positive downwards), torch.Tensor
        
    Returns:
        rho: Seawater in-situ density in kg/m^3, torch.Tensor
    """
    # Pressure in dbar (approx 1 dbar per meter depth)
    p = depth * 1.019716e-1
    
    # Pure water density at atmospheric pressure (standard UNESCO formula)
    rhow = (
        999.842594
        + 6.793952e-2 * temp
        - 9.095290e-3 * (temp ** 2)
        + 1.001685e-4 * (temp ** 3)
        - 1.120083e-6 * (temp ** 4)
        + 6.536332e-9 * (temp ** 5)
    )
    
    # Atmospheric pressure density terms for salinity
    a = (
        8.24493e-1
        - 4.0899e-3 * temp
        + 7.6438e-5 * (temp ** 2)
        - 8.2467e-7 * (temp ** 3)
        + 5.3875e-9 * (temp ** 4)
    )
    b = -5.72466e-3 + 1.0227e-4 * temp - 1.6546e-6 * (temp ** 2)
    c = 4.8314e-4
    
    rho_0 = rhow + a * sal + b * (torch.clamp(sal, min=0.0) ** 1.5) + c * (sal ** 2)
    
    # Secant bulk modulus K(S, T, p) terms
    kw = 19652.21 + 148.4206 * temp - 2.327105 * (temp ** 2) + 1.360477e-2 * (temp ** 3) - 5.155288e-5 * (temp ** 4)
    k_sal = (54.6746 - 0.603459 * temp + 1.09987e-2 * (temp ** 2) - 6.1670e-5 * (temp ** 3)) * sal
    k_sal2 = (7.944e-2 + 1.6483e-2 * temp - 5.3009e-4 * (temp ** 2)) * (torch.clamp(sal, min=0.0) ** 1.5)
    k0 = kw + k_sal + k_sal2
    
    # Pressure dependence terms
    a1 = 3.239908 + 1.43713e-3 * temp + 1.16092e-4 * (temp ** 2) - 5.77905e-7 * (temp ** 3)
    b1 = (2.2838e-3 - 1.0981e-5 * temp - 1.6078e-6 * (temp ** 2)) * sal
    c1 = 1.91075e-4 * (torch.clamp(sal, min=0.0) ** 1.5)
    k_p = (a1 + b1 + c1) * p
    
    k = k0 + k_p + (8.50935e-5 - 6.12293e-6 * temp + 5.2787e-8 * (temp ** 2)) * (p ** 2)
    
    # In-situ density under pressure
    rho = rho_0 / (1.0 - p / torch.clamp(k, min=1e4))
    return rho


def calc_potential_density_sigma(sal, temp, p_ref=0.0):
    """
    Computes potential density anomaly (kg/m^3) referenced to pressure p_ref:
    sigma = rho(S, T, p_ref) - 1000.0
    
    Args:
        sal: Salinity in PSU / g/kg, torch.Tensor or numpy.ndarray
        temp: Temperature in deg C, matching type
        p_ref: Reference pressure in dbar (e.g. 0.0 for sigma_0, 1000.0 for sigma_1)
        
    Returns:
        sigma: Potential density anomaly in kg/m^3
    """
    is_numpy = not isinstance(sal, torch.Tensor)
    if is_numpy:
        sal_t = torch.from_numpy(sal).float()
        temp_t = torch.from_numpy(temp).float()
    else:
        sal_t = sal
        temp_t = temp

    # Convert p_ref in dbar to equivalent depth in meters
    z_ref = torch.tensor(p_ref / 1.019716e-1, dtype=sal_t.dtype, device=sal_t.device)
    rho_ref = approx_seawater_density(sal_t, temp_t, z_ref)
    sigma = rho_ref - 1000.0

    if is_numpy:
        return sigma.detach().cpu().numpy()
    return sigma


def calc_buoyancy_frequency_n2(sal, temp, depth, g=9.80665, depth_axis=-1):
    """
    Computes the exact Brunt-Väisälä buoyancy frequency squared N^2 (s^-2)
    using the international TEOS-10 local midpoint pressure method.
    
    N^2 = g * (rho_lower - rho_upper) / (rho_mid * dz)
    where rho_upper and rho_lower are both evaluated at the shared local midpoint pressure.
    This strictly eliminates fictitious density inversions caused by thermobaricity / reference pressure biases.
    
    Args:
        sal: Salinity in PSU / g/kg, torch.Tensor or numpy.ndarray
        temp: Temperature in deg C, matching type
        depth: 1D depth array/tensor in meters
        g: Gravitational acceleration (default: 9.80665 m/s^2)
        depth_axis: Axis along which depth is oriented (default: -1 or auto-detected)
        
    Returns:
        N2: Buoyancy frequency squared in s^-2, shape matches inputs with depth dimension (D - 1)
    """
    is_numpy = not isinstance(sal, torch.Tensor)
    if is_numpy:
        sal_t = torch.from_numpy(sal).float()
        temp_t = torch.from_numpy(temp).float()
    else:
        sal_t = sal
        temp_t = temp

    if not isinstance(depth, torch.Tensor):
        depth_t = torch.tensor(depth, dtype=sal_t.dtype, device=sal_t.device)
    else:
        depth_t = depth.to(device=sal_t.device, dtype=sal_t.dtype)

    D = len(depth_t)
    # Determine depth axis
    if depth_axis is not None and depth_axis != -1:
        axis = depth_axis
    elif sal_t.shape[-1] == D:
        axis = sal_t.dim() - 1
    else:
        # Auto-detect which axis matches depth dimension D
        axis = sal_t.dim() - 1
        for d_idx, s_val in enumerate(sal_t.shape):
            if s_val == D:
                axis = d_idx
                break

    # Extract upper and lower slices along depth axis
    upper_slice = [slice(None)] * sal_t.dim()
    upper_slice[axis] = slice(0, D - 1)
    lower_slice = [slice(None)] * sal_t.dim()
    lower_slice[axis] = slice(1, D)

    temp_upper = temp_t[tuple(upper_slice)]
    temp_lower = temp_t[tuple(lower_slice)]
    sal_upper = sal_t[tuple(upper_slice)]
    sal_lower = sal_t[tuple(lower_slice)]

    # Compute midpoint depth and dz
    z_mid = 0.5 * (depth_t[:-1] + depth_t[1:])
    dz = depth_t[1:] - depth_t[:-1]

    # Reshape z_mid and dz to broadcast across other dimensions
    view_shape = [1] * sal_t.dim()
    view_shape[axis] = D - 1
    z_mid_view = z_mid.view(view_shape)
    dz_view = dz.view(view_shape)

    # Evaluate density of both parcels at shared midpoint pressure
    rho_upper = approx_seawater_density(sal_upper, temp_upper, z_mid_view)
    rho_lower = approx_seawater_density(sal_lower, temp_lower, z_mid_view)
    rho_mid = 0.5 * (rho_upper + rho_lower)

    # N^2 = g * (rho_lower - rho_upper) / (rho_mid * dz)
    # When lower water is denser under identical pressure, rho_lower > rho_upper => N^2 > 0 (stable)
    n2 = g * (rho_lower - rho_upper) / (torch.clamp(rho_mid, min=900.0) * torch.clamp(dz_view, min=0.1))

    if is_numpy:
        return n2.detach().cpu().numpy()
    return n2

