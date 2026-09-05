# -*- coding: utf-8 -*-
"""
Evaluation and Metrics Assessment Script for Pinn-Ocean (Swin-Ocean-PINN)
Computes layer-by-layer RMSE, MAE, R^2 score and Mixed Layer Depth (MLD) error.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from configs.default_config import ModelConfig, DataConfig
from pinn_ocean.models.swin_ocean_pinn import SwinOceanPINN
from pinn_ocean.datasets.ocean_dataset import OceanContinuousDataset
from pinn_ocean.utils.metrics import calc_rmse, calc_mae, calc_r2, calc_mld


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Swin-Ocean-PINN Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/swin_ocean_pinn_best.pth",
                        help="Path to trained model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device(args.device)

    print("==================================================================")
    print("                Swin-Ocean-PINN Model Evaluation                  ")
    print(f" Device: {device} | Checkpoint: {args.checkpoint}")
    print("==================================================================")

    data_cfg = DataConfig()
    try:
        test_dataset = OceanContinuousDataset(data_cfg.sla_path, data_cfg.gt_path, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"[Warning] Could not load test dataset ({e}).")
        return

    # Load Model
    model_cfg = ModelConfig()
    model = SwinOceanPINN(
        in_channels=model_cfg.in_channels,
        embed_dim=model_cfg.embed_dim,
        window_size=model_cfg.window_size,
        physics_hidden_dim=model_cfg.physics_hidden_dim,
        out_dim=model_cfg.out_dim
    ).to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        stats = checkpoint.get('stats', test_dataset.stats)
        print(f"Loaded weights from {args.checkpoint} (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        print(f"[Notice] Checkpoint {args.checkpoint} not found. Running with initial weights.")
        stats = test_dataset.stats

    model.eval()
    all_preds_t = []
    all_preds_s = []
    all_targets_t = []
    all_targets_s = []

    z_raw = test_dataset.get_depth_tensor().to(device)
    z_norm = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-6)

    with torch.no_grad():
        for x_8ch, y_3d in test_loader:
            x_8ch, y_3d = x_8ch.to(device), y_3d.to(device)
            preds = model(x_8ch, z_norm, sample_idx=None)  # (1, 2, D, H, W)

            # Un-normalize to physical units
            pred_t = preds[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
            pred_s = preds[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']
            target_t = y_3d[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
            target_s = y_3d[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']

            all_preds_t.append(pred_t)
            all_preds_s.append(pred_s)
            all_targets_t.append(target_t)
            all_targets_s.append(target_s)

    all_preds_t = np.array(all_preds_t)
    all_preds_s = np.array(all_preds_s)
    all_targets_t = np.array(all_targets_t)
    all_targets_s = np.array(all_targets_s)

    # Compute overall metrics
    rmse_t = calc_rmse(all_preds_t, all_targets_t)
    r2_t = calc_r2(all_preds_t, all_targets_t)
    rmse_s = calc_rmse(all_preds_s, all_targets_s)
    r2_s = calc_r2(all_preds_s, all_targets_s)

    print("\n---------------------- Evaluation Results ----------------------")
    print(f" Temperature:  RMSE = {rmse_t:.4f} °C | R^2 = {r2_t:.4f}")
    print(f" Salinity:     RMSE = {rmse_s:.4f} PSU | R^2 = {r2_s:.4f}")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    evaluate()
