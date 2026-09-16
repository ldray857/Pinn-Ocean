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

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
    z_raw = torch.linspace(0.5, 1000.0, D, device=device).requires_grad_(True)

    print(f"      Input x shape: {tuple(x_8ch.shape)}")
    print(f"      Depth z shape: {tuple(z_raw.shape)}")

    # 4. Test Forward Pass
    print("\n[5/6] Testing Model Forward Pass (Full Grid and Random Sampling)...")
    # Mode A: Full grid
    preds_full = model(x_8ch, z_raw, sample_idx=None)
    print(f"      Full-grid Prediction Shape: {tuple(preds_full.shape)} (Expected: {B}, 2, {D}, {H}, {W})")
    assert preds_full.shape == (B, 2, D, H, W), "Full-grid prediction shape mismatch"

    # Mode B: Random sampling (e.g. 500 points)
    sample_idx = torch.randperm(H * W)[:500].to(device)
    preds_sampled = model(x_8ch, z_raw, sample_idx=sample_idx)
    print(f"      Sampled Prediction Shape: {tuple(preds_sampled.shape)} (Expected: {B}, 2, {D}, 500)")
    assert preds_sampled.shape == (B, 2, D, 500), "Sampled prediction shape mismatch"

    # 5. Test Physics Loss & Autograd Backpropagation
    print("\n[6/6] Testing Physics Loss Calculation and Backpropagation...")
    phy_loss_fn = OceanPhysicsLoss().to(device)
    adaptive_loss_fn = AdaptiveMultiObjectiveLoss().to(device)
    mse_loss_fn = nn.MSELoss()

    synthetic_target = torch.randn(B, 2, D, 500, device=device)
    synthetic_stats = {
        'mean_t': 15.0, 'std_t': 8.0,
        'mean_s': 34.5, 'std_s': 0.5,
        'mean_sla': 0.0, 'std_sla': 0.1
    }
    surface_obs = {
        'sst': x_8ch[:, 0].flatten(1)[:, sample_idx],
        'sla': x_8ch[:, 1].flatten(1)[:, sample_idx],
        'sss': x_8ch[:, 2].flatten(1)[:, sample_idx]
    }

    # Pointwise Autograd (per-sample per-depth physical gradient)
    z_raw_pts = z_raw.view(1, 1, D, 1).repeat(B, 500, 1, 1).requires_grad_(True)
    preds_sampled_pts = model(x_8ch, z_raw_pts, sample_idx=sample_idx)
    loss_data_pts = mse_loss_fn(preds_sampled_pts, synthetic_target)
    loss_phy_pts, loss_dict_pts = phy_loss_fn(
        preds_sampled_pts, z_raw_pts, stats=synthetic_stats, z_raw=z_raw,
        surface_obs=surface_obs, y_target=synthetic_target
    )
    total_loss_pts, w1_pts, w2_pts = adaptive_loss_fn(loss_data_pts, loss_phy_pts)

    print(f"      [Pointwise] Data Loss: {loss_data_pts.item():.5f} | Phy Loss: {loss_phy_pts.item():.5f} (Surf: {loss_dict_pts['loss_surf'].item():.5f}, SLA: {loss_dict_pts['loss_sla'].item():.5f}, Grad: {loss_dict_pts['loss_grad'].item():.5f}, Stab: {loss_dict_pts['loss_stab'].item():.5f})")
    print(f"      [Pointwise] Adaptive Loss: {total_loss_pts.item():.5f}")

    optimizer = optim.AdamW(list(model.parameters()) + list(adaptive_loss_fn.parameters()), lr=1e-3)
    optimizer.zero_grad()
    total_loss_pts.backward()
    optimizer.step()
    print("      --> Pointwise backward pass and parameter update executed successfully.")

    # 6. Test Comprehensive Metrics & Physical Diagnostics
    print("\n[7/7] Testing Comprehensive Evaluation Metrics & Physical Diagnostics...")
    from pinn_ocean.utils.metrics import (
        calc_layer_metrics, calc_regime_metrics,
        calc_density_inversion_rate, calc_temp_monotonicity_violation,
        calc_domain_mld_metrics
    )

    synth_depths = np.linspace(0.0, 1000.0, D)
    # Create physically realistic stratification: temperature decreasing, salinity S-curve
    t_synth_true = 20.0 * np.exp(-synth_depths[:, None, None] / 300.0) + 3.0
    t_synth_pred = t_synth_true + np.random.normal(0, 0.2, t_synth_true.shape)
    s_synth_true = 34.0 + 0.5 * np.sin(synth_depths[:, None, None] / 200.0)
    s_synth_pred = s_synth_true + np.random.normal(0, 0.03, s_synth_true.shape)

    # Test layer metrics
    l_metrics_t = calc_layer_metrics(t_synth_pred, t_synth_true, synth_depths)
    assert len(l_metrics_t['rmse']) == D, "Layer metrics length mismatch"

    # Test regime metrics
    r_metrics = calc_regime_metrics(t_synth_pred, t_synth_true, synth_depths)
    assert "mixed_layer" in r_metrics and "thermocline" in r_metrics and "deep_layer" in r_metrics

    # Test density inversion rate
    dir_info = calc_density_inversion_rate(t_synth_pred, s_synth_pred, synth_depths)
    assert "inversion_rate_percent" in dir_info
    print(f"      Density Inversion Rate: {dir_info['inversion_rate_percent']:.3f}% ({dir_info['total_inversions']}/{dir_info['total_evaluated']})")

    # Test Brunt-Väisälä buoyancy frequency metrics
    from pinn_ocean.utils.metrics import calc_buoyancy_frequency_metrics, calc_multilevel_density_inversion_rate
    n2_info = calc_buoyancy_frequency_metrics(t_synth_pred, s_synth_pred, synth_depths, true_temp=t_synth_true, true_sal=s_synth_true)
    assert "cir_pred_percent" in n2_info and "pycnocline_depth_rmse" in n2_info
    print(f"      Brunt-Vaisala CIR: {n2_info['cir_pred_percent']:.3f}% | Pycnocline Depth RMSE: {n2_info['pycnocline_depth_rmse']:.2f} m")

    # Test Multi-level Potential Density Inversion Rate
    ml_info = calc_multilevel_density_inversion_rate(t_synth_pred, s_synth_pred, synth_depths)
    assert "overall_inversion_rate_percent" in ml_info
    print(f"      Multi-level Density Inversion Rate: {ml_info['overall_inversion_rate_percent']:.3f}%")

    # Test temperature monotonicity violation
    mono_info = calc_temp_monotonicity_violation(t_synth_pred, synth_depths, start_depth=100.0)
    assert "violation_rate_percent" in mono_info

    # Test domain MLD metrics
    mld_info = calc_domain_mld_metrics(t_synth_pred, t_synth_true, synth_depths)
    assert "mld_rmse" in mld_info and "mld_mae" in mld_info
    print(f"      MLD RMSE: {mld_info['mld_rmse']:.2f} m | MAE: {mld_info['mld_mae']:.2f} m")

    # 7. Test Continuous Space-Depth Super-Resolution & GLORYS 3D Interpolation
    print("\n[8/9] Testing Continuous Super-Resolution & GLORYS Interpolation...")
    from pinn_ocean.models.super_resolution import (
        ContinuousSpaceDepthSuperResolver,
        GLORYS3DInterpolator,
        compute_super_resolution_metrics
    )
    resolver = ContinuousSpaceDepthSuperResolver(model=model, device=device)
    # Test 2x spatial upsampling
    x_hr_test = resolver.super_resolve_surface_input(x_8ch, scale_factor=2.0)
    assert x_hr_test.shape == (B, 8, H * 2, W * 2), "Super-resolved surface shape mismatch"
    print(f"      Super-resolved Surface Tensor: {tuple(x_hr_test.shape)} (2x magnification)")

    # Test GLORYS 3D Interpolator
    synth_lats = np.linspace(30.0, 40.0, 10)
    synth_lons = np.linspace(145.0, 165.0, 15)
    interpolator = GLORYS3DInterpolator(depths=synth_depths, lats=synth_lats, lons=synth_lons)
    synth_field = np.random.randn(D, 10, 15).astype(np.float32)
    tgt_lats = np.linspace(30.0, 40.0, 20)
    tgt_lons = np.linspace(145.0, 165.0, 30)
    interp_field = interpolator.interpolate_field(synth_field, synth_depths, tgt_lats, tgt_lons, method='physics_regularized')
    assert interp_field.shape == (D, 20, 30), "GLORYS 3D interpolation shape mismatch"
    print(f"      GLORYS 3D Physics-Regularized Interpolation: {tuple(interp_field.shape)}")

    # 8. Test Baselines & Benchmark Metrics
    print("\n[9/9] Testing Baselines & Superiority Benchmark Metrics...")
    from pinn_ocean.models.baselines import TrilinearBaseline3D, PureDataCNN3D, PureSwinAblation
    from pinn_ocean.utils.metrics import calc_model_superiority_index, calc_psnr
    cnn_model = PureDataCNN3D(in_channels=8, hidden_dim=32).to(device)
    out_cnn = cnn_model(x_8ch, z_raw)
    assert out_cnn.shape == (B, 2, D, H, W), "PureDataCNN forward pass mismatch"

    sup_score = calc_model_superiority_index(
        rmse_t=1.54, rmse_s=0.10, r2_t=0.95, r2_s=0.88,
        cir_percent=1.10, tmv_percent=0.01, mld_mae=19.2
    )
    assert "composite_score" in sup_score
    print(f"      Composite Superiority Index Score: {sup_score['composite_score']:.2f}/100")

    # Test visualization module imports including benchmark visualizations
    from pinn_ocean.visualization import (
        plot_3d_thermohaline_box,
        plot_layer_metrics_profile,
        plot_multi_station_profiles, plot_superiority_radar,
        plot_physics_stability_transect, plot_glorys_super_resolution_comparison,
        plot_superiority_bar_summary
    )
    print("      --> All visualization, super-resolution, and benchmark modules imported successfully.")

    print("\n==================================================================")
    print(" [PASSED] All Pinn-Ocean core components, super-res & benchmark verified! ")
    print("==================================================================")



if __name__ == "__main__":
    run_unit_tests()

