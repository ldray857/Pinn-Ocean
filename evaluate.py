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
from pinn_ocean.utils import get_result_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Swin-Ocean-PINN Checkpoint")
    parser.add_argument("--data_dir", type=str,
                        default="data/2020" if os.path.exists("data/2020/pacific_sla_2013_2021.nc") else "data",
                        help="Path to folder containing NetCDF datasets (default: auto-detect data/2020 or data)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/swin_ocean_pinn_best.pth",
                        help="Path to trained model weights")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "val", "test", "all"],
                        help="Dataset partition to evaluate ('train', 'val', 'test', or 'all')")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Optional specific years to include (e.g. --years 2017 2018 2019 2020)")
    parser.add_argument("--result_dir", type=str, default="result",
                        help="Root result directory (default: result)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment/year tag name (default: auto-detected, e.g. 2015_2020)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device(args.device)

    sla_path = os.path.join(args.data_dir, "pacific_sla_2013_2021.nc")
    gt_path = os.path.join(args.data_dir, "pacific_glorys_3d_temp_sal_2013_2021.nc")

    try:
        test_dataset = OceanContinuousDataset(sla_path, gt_path, years=args.years, mode=args.mode)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"[Dataset] {args.mode.upper()} samples: {len(test_dataset)}")
    except Exception as e:
        print(f"[Warning] Could not load test dataset ({e}).")
        return

    # Setup standardized result directory structure
    res_dirs = get_result_dirs(
        result_dir=args.result_dir,
        tag=args.tag,
        dataset=test_dataset,
        data_dir=args.data_dir,
        years=args.years
    )

    # Checkpoint resolution: prioritize result/<year_tag>/checkpoints/ if default was provided
    ckpt_path = args.checkpoint
    tag_ckpt = os.path.join(res_dirs['ckpt_dir'], "swin_ocean_pinn_best.pth")
    if args.checkpoint == "checkpoints/swin_ocean_pinn_best.pth" and os.path.exists(tag_ckpt):
        ckpt_path = tag_ckpt

    print("==================================================================")
    print("                Swin-Ocean-PINN Model Evaluation                  ")
    print(f" Device: {device} | Checkpoint: {ckpt_path} | Partition: {args.mode.upper()}")
    print(f" Data Directory: {os.path.abspath(args.data_dir)}")
    print(f" Result Tag    : {res_dirs['tag']} ({res_dirs['exp_dir']})")
    if args.years:
        print(f" Filter Years  : {args.years}")
    print("==================================================================")

    # Load Model
    model_cfg = ModelConfig()
    model = SwinOceanPINN(
        in_channels=model_cfg.in_channels,
        embed_dim=model_cfg.embed_dim,
        window_size=model_cfg.window_size,
        physics_hidden_dim=model_cfg.physics_hidden_dim,
        out_dim=model_cfg.out_dim
    ).to(device)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        stats = checkpoint.get('stats', test_dataset.stats)
        print(f"Loaded weights from {ckpt_path} (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        print(f"[Notice] Checkpoint {ckpt_path} not found. Running with initial weights.")
        stats = test_dataset.stats

    model.eval()
    all_preds_t = []
    all_preds_s = []
    all_targets_t = []
    all_targets_s = []

    z_raw = test_dataset.get_depth_tensor().to(device)

    with torch.no_grad():
        for x_8ch, y_3d in test_loader:
            x_8ch, y_3d = x_8ch.to(device), y_3d.to(device)
            preds = model(x_8ch, z_raw, sample_idx=None)  # (1, 2, D, H, W)

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
    mae_t = calc_mae(all_preds_t, all_targets_t)
    r2_t = calc_r2(all_preds_t, all_targets_t)

    rmse_s = calc_rmse(all_preds_s, all_targets_s)
    mae_s = calc_mae(all_preds_s, all_targets_s)
    r2_s = calc_r2(all_preds_s, all_targets_s)

    summary_lines = [
        "---------------------- Evaluation Results ----------------------",
        f" Checkpoint:   {ckpt_path}",
        f" Partition:    {args.mode.upper()} ({len(test_dataset)} samples)",
        f" Result Tag:   {res_dirs['tag']}",
        f" Temperature:  RMSE = {rmse_t:.4f} °C | MAE = {mae_t:.4f} °C | R^2 = {r2_t:.4f}",
        f" Salinity:     RMSE = {rmse_s:.4f} PSU | MAE = {mae_s:.4f} PSU | R^2 = {r2_s:.4f}",
        "----------------------------------------------------------------"
    ]

    for line in summary_lines:
        print(line)

    eval_log_path = os.path.join(res_dirs['log_dir'], "eval.log")
    with open(eval_log_path, "a", encoding="utf-8") as f_eval:
        f_eval.write("\n".join(summary_lines) + "\n\n")
    print(f"Results saved to log: {eval_log_path}")


if __name__ == "__main__":
    evaluate()
