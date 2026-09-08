# -*- coding: utf-8 -*-
"""
Physics Loss Module for Pinn-Ocean
Implements multi-objective ocean physics constraints:
1. Active Dynamic Height (SLA) Physical Coupling (TEOS-10 steric integration)
2. Sea Surface Dirichlet Boundary Condition Anchor (SST / SSS)
3. Autograd Continuous Vertical Profile Derivative Supervision (Thermocline & Halocline Curvature)
4. Continuous Seawater Stratification Stability (Anti-Inversion / Buoyancy Frequency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.teos10 import approx_seawater_density


class OceanPhysicsLoss(nn.Module):
    """
    Active Ocean Physics Loss Engine embedding fundamental geophysical laws:
    1. Dynamic Height Anomaly (SLA): L_sla = MSE(eta_steric, SLA_obs)
    2. Sea Surface Dirichlet Boundary Condition: L_surf = MSE(T_0, SST) + MSE(S_0, SSS) (in unified coordinates)
    3. Vertical Gradient Form: L_grad = MSE(dT/dz, dT_gt/dz) + 2 * MSE(dS/dz, dS_gt/dz)
    4. Mixed Layer Isothermal Regularization: L_mld = mean(relu(|dT/dz| - 0.02)) for z <= 30m
    5. Stratification Stability: L_stab = mean(softplus(- d(rho)/dz * scale))
    """
    def __init__(self, w_sla=5.0, w_surf=2.0, w_grad=1.0, w_mld=2.0, w_stab=0.2,
                 temp_grad_threshold=0.005, enable_density=True):
        super().__init__()
        self.w_sla = w_sla
        self.w_surf = w_surf
        self.w_grad = w_grad
        self.w_mld = w_mld
        self.w_stab = w_stab
        self.temp_grad_threshold = temp_grad_threshold
        self.enable_density = enable_density

    def forward(self, preds, z_pts, stats=None, z_raw=None,
                surface_obs=None, y_target=None):
        """
        Args:
            preds: Predicted thermohaline field (B, 2, D, S) or (B, 2, D, H, W)
                   channel 0: normalized temperature
                   channel 1: normalized salinity
            z_pts: Continuous depth tensor (B, S, D, 1) or (D,), requires_grad=True
            stats: Dict containing normalization statistics ('mean_t', 'std_t', 'mean_s', 'std_s', 'mean_sla', 'std_sla')
            z_raw: 1D Tensor of raw depths in meters (D,)
            surface_obs: Optional dict with 'sst', 'sss', 'sla' surface ground truth tensors of shape (B, S)
            y_target: Optional ground truth 3D tensor (B, 2, D, S)
            
        Returns:
            loss_physics: Combined active physics penalty loss
            loss_dict: Dictionary recording individual components
        """
        device = preds.device
        if preds.dim() == 5:
            B, C, D, H, W = preds.shape
            preds_flat = preds.view(B, C, D, -1)
        else:
            preds_flat = preds
            B, C, D, S = preds.shape

        temp_pts = preds_flat[:, 0, :, :].permute(0, 2, 1)  # (B, S, D)
        sal_pts = preds_flat[:, 1, :, :].permute(0, 2, 1)   # (B, S, D)

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        # 1. Surface Boundary Condition Anchor (Dirichlet BC at z=0)
        # Unified into the exact normalized coordinate frame of 3D thermohaline predictions
        loss_surf = torch.tensor(0.0, device=device)
        if surface_obs is not None and 'sst' in surface_obs and 'sss' in surface_obs:
            t_surf_pred = temp_pts[:, :, 0]  # (B, S) at surface layer (z=0.5m)
            s_surf_pred = sal_pts[:, :, 0]  # (B, S) at surface layer (z=0.5m)
            if stats is not None:
                mean_sst = stats.get('mean_sst', stats['mean_t'])
                std_sst = stats.get('std_sst', stats['std_t'])
                mean_sss = stats.get('mean_sss', stats['mean_s'])
                std_sss = stats.get('std_sss', stats['std_s'])

                sst_phys = surface_obs['sst'] * std_sst + mean_sst
                sss_phys = surface_obs['sss'] * std_sss + mean_sss

                sst_target_norm = (sst_phys - stats['mean_t']) / stats['std_t']
                sss_target_norm = (sss_phys - stats['mean_s']) / stats['std_s']
            else:
                sst_target_norm = surface_obs['sst']
                sss_target_norm = surface_obs['sss']

            loss_surf = F.mse_loss(t_surf_pred, sst_target_norm) + F.mse_loss(s_surf_pred, sss_target_norm)
            total_loss = total_loss + self.w_surf * loss_surf
        loss_dict['loss_surf'] = loss_surf

        # 2. Dynamic Height (SLA) Physical Steric Integration Constraint
        loss_sla = torch.tensor(0.0, device=device)
        if (surface_obs is not None and 'sla' in surface_obs and
                stats is not None and z_raw is not None and D > 1):
            t_phys = temp_pts * stats['std_t'] + stats['mean_t']
            s_phys = sal_pts * stats['std_s'] + stats['mean_s']
            depth_phys = z_raw.view(1, 1, D).to(device)

            rho = approx_seawater_density(s_phys, t_phys, depth_phys)
            rho_ref = rho.mean(dim=1, keepdim=True)  # (B, 1, D)
            rho_prime = rho - rho_ref

            dz = (z_raw[1:] - z_raw[:-1]).view(1, 1, D - 1).to(device)
            rho_mid = 0.5 * (rho_prime[:, :, :-1] + rho_prime[:, :, 1:])
            dyn_height = - torch.sum(rho_mid * dz, dim=-1) / 1025.0  # (B, S) in meters

            sla_phys = surface_obs['sla'] * stats['std_sla'] + stats['mean_sla']
            loss_sla = F.mse_loss(dyn_height, sla_phys)
            total_loss = total_loss + self.w_sla * loss_sla
        loss_dict['loss_sla'] = loss_sla

        # 3. Vertical Profile Gradient & Curvature Supervision
        loss_grad = torch.tensor(0.0, device=device)
        loss_mld = torch.tensor(0.0, device=device)
        loss_stab = torch.tensor(0.0, device=device)
        if y_target is not None and z_raw is not None and D > 1:
            t_gt = y_target[:, 0, :, :].permute(0, 2, 1)  # (B, S, D)
            s_gt = y_target[:, 1, :, :].permute(0, 2, 1)  # (B, S, D)

            dz_100 = ((z_raw[1:] - z_raw[:-1]) / 100.0).view(1, 1, D - 1).to(device)

            pred_dt = (temp_pts[:, :, 1:] - temp_pts[:, :, :-1]) / dz_100
            pred_ds = (sal_pts[:, :, 1:] - sal_pts[:, :, :-1]) / dz_100
            gt_dt = (t_gt[:, :, 1:] - t_gt[:, :, :-1]) / dz_100
            gt_ds = (s_gt[:, :, 1:] - s_gt[:, :, :-1]) / dz_100

            loss_grad_t = F.mse_loss(pred_dt, gt_dt)
            loss_grad_s = F.mse_loss(pred_ds, gt_ds)
            loss_grad = loss_grad_t + 2.0 * loss_grad_s
            total_loss = total_loss + self.w_grad * loss_grad

            # 4. Mixed Layer Isothermal Regularization (MLD Flatness in upper 30m)
            # Ocean mixed layer is well-mixed: penalizes boundary overshoot or curvature
            if stats is not None:
                mld_mask = z_raw[:-1] <= 30.0
                if mld_mask.any():
                    dz_mld = (z_raw[1:] - z_raw[:-1])[mld_mask].view(1, 1, -1).to(device)
                    dt_phys_mld = torch.abs(temp_pts[:, :, 1:][:, :, mld_mask] - temp_pts[:, :, :-1][:, :, mld_mask]) * stats['std_t'] / dz_mld
                    loss_mld = torch.mean(F.relu(dt_phys_mld - 0.02))
                    total_loss = total_loss + self.w_mld * loss_mld

            # 5. Smooth Seawater Stratification Stability (Anti-Inversion)
            if self.enable_density and stats is not None:
                d_rho_dz = - 0.25 * (pred_dt * stats['std_t']) + 0.75 * (pred_ds * stats['std_s'])
                loss_stab = torch.mean(F.softplus(- d_rho_dz * 10.0))
                total_loss = total_loss + self.w_stab * loss_stab

        loss_dict['loss_grad'] = loss_grad
        loss_dict['loss_mld'] = loss_mld
        loss_dict['loss_stab'] = loss_stab
        loss_dict['loss_phy_total'] = total_loss

        return total_loss, loss_dict
