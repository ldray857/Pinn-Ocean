# -*- coding: utf-8 -*-
"""
Self-Contained Verification & Pipeline Test Suite for Pinn-Ocean
Validates the complete deep learning & physics pipeline without external NetCDF dependencies:
1. Hardware / CUDA environment check
2. TEOS-10 differentiable seawater equation of state & Autograd derivative
3. Swin-Ocean-PINN model initialization & parameter accounting
4. Forward pass in full-grid and random-sampling modes
5. Physics loss (thermal monotonicity + stratification stability)
6. Backward pass and optimizer parameter updates
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim

from configs.default_config import ModelConfig, PhysicsConfig
from pinn_ocean.models.swin_ocean_pinn import SwinOceanPINN
from pinn_ocean.losses.physics_loss import OceanPhysicsLoss
from pinn_ocean.losses.adaptive_loss import AdaptiveMultiObjectiveLoss
from pinn_ocean.utils.teos10 import approx_seawater_density
from pinn_ocean.utils.metrics import calc_rmse, calc_r2, calc_mld


def run_unit_tests():
    print("==================================================================")
    print("           Pinn-Ocean Unit Testing & Pipeline Verification        ")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] Hardware Environment Check: Device = {device}")

    # 1. Test Differentiable Seawater Equation of State
    print("\n[2/6] Testing TEOS-10 / UNESCO Differentiable Seawater Density...")
    sample_sal = torch.tensor([34.5, 35.0, 35.5], requires_grad=True, device=device)
    sample_temp = torch.tensor([25.0, 15.0, 5.0], requires_grad=True, device=device)
    sample_depth = torch.tensor([10.0, 200.0, 800.0], requires_grad=True, device=device)

    rho = approx_seawater_density(sample_sal, sample_temp, sample_depth)
    print(f"      Computed Densities (kg/m^3): {rho.detach().cpu().numpy()}")
    assert rho.shape == (3,), "Density tensor shape mismatch"
    assert torch.all(rho > 1000.0) and torch.all(rho < 1050.0), "Density values out of physical seawater range"

    # Check differentiability
    rho.sum().backward()
    assert sample_depth.grad is not None, "Autograd gradient through density failed"
    print("      --> Density equation differentiability test passed.")

    # 2. Test Swin-Ocean-PINN Model Initialization
    print("\n[3/6] Initializing Swin-Ocean-PINN Architecture...")
    model = SwinOceanPINN(
        in_channels=8,
        embed_dim=64,
        window_size=4,
        physics_hidden_dim=128,
        out_dim=2
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total Trainable Parameters: {total_params:,}")

    # 3. Simulate Inputs
    print("\n[4/6] Simulating 8-Channel Sea Surface Input and Continuous Depth...")
    B, H, W = 2, 40, 40
    D = 25
    x_8ch = torch.randn(B, 8, H, W, device=device)
    z_raw = torch.linspace(0.5, 1000.0, D, device=device)
    z_norm = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-6)
    z_norm.requires_grad_(True)

    print(f"      Input x shape: {tuple(x_8ch.shape)}")
    print(f"      Depth z shape: {tuple(z_norm.shape)}")

    # 4. Test Forward Pass
    print("\n[5/6] Testing Model Forward Pass (Full Grid and Random Sampling)...")
    # Mode A: Full grid
    preds_full = model(x_8ch, z_norm, sample_idx=None)
    print(f"      Full-grid Prediction Shape: {tuple(preds_full.shape)} (Expected: {B}, 2, {D}, {H}, {W})")
    assert preds_full.shape == (B, 2, D, H, W), "Full-grid prediction shape mismatch"

    # Mode B: Random sampling (e.g. 500 points)
    sample_idx = torch.randperm(H * W)[:500].to(device)
    preds_sampled = model(x_8ch, z_norm, sample_idx=sample_idx)
    print(f"      Sampled Prediction Shape: {tuple(preds_sampled.shape)} (Expected: {B}, 2, {D}, 500)")
    assert preds_sampled.shape == (B, 2, D, 500), "Sampled prediction shape mismatch"

    # 5. Test Physics Loss & Autograd Backpropagation
    print("\n[6/6] Testing Physics Loss Calculation and Backpropagation...")
    phy_loss_fn = OceanPhysicsLoss(temp_grad_threshold=0.005, enable_density=True).to(device)
    adaptive_loss_fn = AdaptiveMultiObjectiveLoss().to(device)
    mse_loss_fn = nn.MSELoss()

    synthetic_target = torch.randn(B, 2, D, 500, device=device)
    synthetic_stats = {
        'mean_t': 15.0, 'std_t': 8.0,
        'mean_s': 34.5, 'std_s': 0.5
    }

    optimizer = optim.AdamW(list(model.parameters()) + list(adaptive_loss_fn.parameters()), lr=1e-3)
    optimizer.zero_grad()

    loss_data = mse_loss_fn(preds_sampled, synthetic_target)
    loss_phy, loss_dict = phy_loss_fn(preds_sampled, z_norm, stats=synthetic_stats, z_raw=z_raw)
    total_loss, w1, w2 = adaptive_loss_fn(loss_data, loss_phy)

    print(f"      Data Loss (MSE): {loss_data.item():.5f}")
    print(f"      Physics Loss: {loss_phy.item():.5f} (Temp: {loss_dict['loss_phy_temp'].item():.5f}, Density: {loss_dict['loss_phy_density'].item():.5f})")
    print(f"      Adaptive Total Loss: {total_loss.item():.5f}")

    total_loss.backward()
    optimizer.step()

    print("      --> Backward pass and parameter update executed successfully.")
    print("\n==================================================================")
    print(" [PASSED] All Pinn-Ocean core components verified successfully!   ")
    print("==================================================================")


if __name__ == "__main__":
    run_unit_tests()
