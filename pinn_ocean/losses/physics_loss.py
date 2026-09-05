# -*- coding: utf-8 -*-
"""
Physics Loss Module for Pinn-Ocean
Implements Autograd-based vertical temperature gradient monotonicity constraint
and TEOS-10 / Seawater stratification stability (anti-density-inversion) constraint.
"""

import torch
import torch.nn as nn
from ..utils.teos10 import approx_seawater_density


class OceanPhysicsLoss(nn.Module):
    """
    Ocean Physics Loss Function embedding domain physical laws:
    1. Thermal monotonicity: L_phy_T = (1/N) * sum( ReLU( d T_hat / d z ) )
    2. Stratification stability: L_phy_rho = (1/N) * sum( ReLU( - d rho_hat / d z ) )
    """
    def __init__(self, temp_grad_threshold=0.005, enable_density=True):
        super().__init__()
        self.temp_grad_threshold = temp_grad_threshold
        self.enable_density = enable_density

    def forward(self, preds, z_norm, stats=None, z_raw=None):
        """
        Args:
            preds: Predicted thermohaline field (B, 2, D, S) or (B, 2, D, H, W)
                   channel 0: normalized temperature
                   channel 1: normalized salinity
            z_norm: Continuous depth tensor (D,), must have requires_grad=True
            stats: Optional dict containing normalization statistics ('mean_t', 'std_t', 'mean_s', 'std_s')
            z_raw: Optional raw depth tensor in meters (D,)
            
        Returns:
            loss_physics: Total physics penalty loss
            loss_dict: Dictionary recording individual components (L_phy_T, L_phy_rho)
        """
        # Squeeze spatial dimensions to compute depth-dependent gradient
        # shape: (B, 2, D, S) -> (D,)
        if preds.dim() == 5:
            # (B, 2, D, H, W) -> flatten H, W to S
            B, C, D, H, W = preds.shape
            preds_flat = preds.view(B, C, D, -1)
        else:
            preds_flat = preds

        temp_profile = preds_flat[:, 0, :, :].mean(dim=(0, 2))  # (D,)
        sal_profile = preds_flat[:, 1, :, :].mean(dim=(0, 2))   # (D,)

        # 1. Autograd analytical vertical temperature gradient: d T / d z
        grad_t = torch.autograd.grad(
            outputs=temp_profile.sum(),
            inputs=z_norm,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Penalize non-physical temperature increase with depth
        loss_phy_t = torch.mean(torch.relu(grad_t + self.temp_grad_threshold))

        # 2. Seawater Stratification Stability (Anti-Density Inversion)
        loss_phy_rho = torch.tensor(0.0, device=preds.device)
        if self.enable_density and stats is not None and z_raw is not None:
            # Un-normalize to physical units for accurate density equation
            temp_phys = temp_profile * stats['std_t'] + stats['mean_t']
            sal_phys = sal_profile * stats['std_s'] + stats['mean_s']
            
            # Compute differentiable in-situ seawater density
            rho = approx_seawater_density(sal_phys, temp_phys, z_raw)  # (D,)
            
            # Autograd derivative of density with respect to depth: d rho / d z
            grad_rho = torch.autograd.grad(
                outputs=rho.sum(),
                inputs=z_norm,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Penalize density decrease with depth (density inversion)
            loss_phy_rho = torch.mean(torch.relu(-grad_rho))

        total_physics_loss = loss_phy_t + loss_phy_rho
        loss_dict = {
            "loss_phy_total": total_physics_loss,
            "loss_phy_temp": loss_phy_t,
            "loss_phy_density": loss_phy_rho
        }

        return total_physics_loss, loss_dict
