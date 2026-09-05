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
