# -*- coding: utf-8 -*-
"""
Evaluation and Metrics Assessment Script for Pinn-Ocean (Swin-Ocean-PINN)
Computes layer-by-layer RMSE, MAE, R^2 score and Mixed Layer Depth (MLD) error.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from configs.default_config import ModelConfig, DataConfig
from pinn_ocean.models.swin_ocean_pinn import SwinOceanPINN
from pinn_ocean.datasets.ocean_dataset import OceanContinuousDataset
from pinn_ocean.utils.metrics import (
    calc_rmse, calc_mae, calc_r2, calc_mld,
    calc_layer_metrics, calc_regime_metrics,
    calc_density_inversion_rate, calc_temp_monotonicity_violation,
    calc_domain_mld_metrics
)
from pinn_ocean.utils import get_result_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Swin-Ocean-PINN Checkpoint with Comprehensive Metrics")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to folder containing NetCDF datasets (default: data)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model weights (default: result/<year_tag>/checkpoints/swin_ocean_pinn_best.pth)")
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

    sla_path = os.path.join(args.data_dir, "pacific_sla.nc")
    gt_path = os.path.join(args.data_dir, "pacific_glorys_3d_temp_sal.nc")

    try:
        test_dataset = OceanContinuousDataset(sla_path, gt_path, years=args.years, mode=args.mode)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"[Dataset] {args.mode.upper()} samples: {len(test_dataset)}")
    except Exception as e:
        print(f"[Warning] Could not load test dataset ({e}).", file=sys.stderr)
        return

    # Setup standardized result directory structure
    res_dirs = get_result_dirs(
        result_dir=args.result_dir,
        tag=args.tag,
        dataset=test_dataset,
        data_dir=args.data_dir,
        years=args.years
    )

    # Checkpoint resolution: prioritize result/<year_tag>/checkpoints/
    ckpt_path = args.checkpoint
    tag_ckpt = os.path.join(res_dirs['ckpt_dir'], "swin_ocean_pinn_best.pth")
    if ckpt_path is None:
        ckpt_path = tag_ckpt if os.path.exists(tag_ckpt) else "checkpoints/swin_ocean_pinn_best.pth"

    print("==================================================================")
    print("        Swin-Ocean-PINN Comprehensive Scientific Evaluation       ")
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
    depths = test_dataset.depths

    with torch.no_grad():
        for x_8ch, y_3d in test_loader:
            x_8ch, y_3d = x_8ch.to(device), y_3d.to(device)
            preds = model(x_8ch, z_raw, sample_idx=None)  # (1, 2, D, H, W)

            # Un-normalize to physical units (°C and PSU)
            pred_t = preds[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
            pred_s = preds[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']
            target_t = y_3d[0, 0].cpu().numpy() * stats['std_t'] + stats['mean_t']
            target_s = y_3d[0, 1].cpu().numpy() * stats['std_s'] + stats['mean_s']

            all_preds_t.append(pred_t)
            all_preds_s.append(pred_s)
            all_targets_t.append(target_t)
            all_targets_s.append(target_s)

    all_preds_t = np.array(all_preds_t)  # (T, D, H, W)
    all_preds_s = np.array(all_preds_s)
    all_targets_t = np.array(all_targets_t)
    all_targets_s = np.array(all_targets_s)

    # 1. Compute overall global metrics
    rmse_t = calc_rmse(all_preds_t, all_targets_t)
    mae_t = calc_mae(all_preds_t, all_targets_t)
    r2_t = calc_r2(all_preds_t, all_targets_t)

    rmse_s = calc_rmse(all_preds_s, all_targets_s)
    mae_s = calc_mae(all_preds_s, all_targets_s)
    r2_s = calc_r2(all_preds_s, all_targets_s)

    # 2. Compute oceanographic regime metrics
    regimes_t = calc_regime_metrics(all_preds_t, all_targets_t, depths)
    regimes_s = calc_regime_metrics(all_preds_s, all_targets_s, depths)

    # 3. Compute layer-by-layer metrics across all 35 depths
    layer_metrics_t = calc_layer_metrics(all_preds_t, all_targets_t, depths)
    layer_metrics_s = calc_layer_metrics(all_preds_s, all_targets_s, depths)

    # 4. Compute physical consistency & stability metrics
    print("\nEvaluating oceanographic physics consistency (TEOS-10 & Monotonicity)...")
    dir_pred = calc_density_inversion_rate(all_preds_t, all_preds_s, depths)
    dir_true = calc_density_inversion_rate(all_targets_t, all_targets_s, depths)
    mono_pred = calc_temp_monotonicity_violation(all_preds_t, depths, start_depth=100.0)
    mono_true = calc_temp_monotonicity_violation(all_targets_t, depths, start_depth=100.0)
    mld_metrics = calc_domain_mld_metrics(all_preds_t, all_targets_t, depths)

    summary_lines = [
        "======================== Evaluation Report ========================",
        f" Checkpoint : {ckpt_path}",
        f" Partition  : {args.mode.upper()} ({len(test_dataset)} time steps, {all_preds_t.size:,} total points)",
        f" Result Tag : {res_dirs['tag']}",
        "------------------------------------------------------------------",
        " 1. 全局总体统计指标 (Global Statistical Metrics):",
        f"    [温度 Temperature] RMSE: {rmse_t:.4f} °C  | MAE: {mae_t:.4f} °C  | R^2: {r2_t:.4f}",
        f"    [盐度 Salinity   ] RMSE: {rmse_s:.4f} PSU | MAE: {mae_s:.4f} PSU | R^2: {r2_s:.4f}",
        "------------------------------------------------------------------",
        " 2. 垂直水动力学分层指标 (Regime-wise Metrics):"
    ]

    for reg_key, t_info in regimes_t.items():
        s_info = regimes_s.get(reg_key, {})
        summary_lines.append(
            f"    - {t_info['label']:<18} | "
            f"T-RMSE: {t_info['rmse']:.4f} °C, T-R2: {t_info['r2']:.4f} | "
            f"S-RMSE: {s_info.get('rmse', 0.0):.4f} PSU, S-R2: {s_info.get('r2', 0.0):.4f}"
        )

    summary_lines.extend([
        "------------------------------------------------------------------",
        " 3. 物理一致性与层化稳定性指标 (PINN Physical Consistency):",
        f"    - 密度倒置率 (Density Inversion Rate, DIR):",
        f"        Swin-Ocean-PINN : {dir_pred['inversion_rate_percent']:.3f}% ({dir_pred['total_inversions']}/{dir_pred['total_evaluated']})",
        f"        GLORYS12V1 真值  : {dir_true['inversion_rate_percent']:.3f}% ({dir_true['total_inversions']}/{dir_true['total_evaluated']})",
        f"    - 位温单调性违规率 (dT/dz > 0 在 z>=100m 深水区):",
        f"        Swin-Ocean-PINN : {mono_pred['violation_rate_percent']:.3f}% ({mono_pred['violations']}/{mono_pred['total']})",
        f"        GLORYS12V1 真值  : {mono_true['violation_rate_percent']:.3f}% ({mono_true['violations']}/{mono_true['total']})",
        f"    - 混合层深度误差 (MLD Error, ΔT=0.5°C 阈值):",
        f"        MLD-RMSE: {mld_metrics['mld_rmse']:.2f} m | MLD-MAE: {mld_metrics['mld_mae']:.2f} m | MLD-R^2: {mld_metrics['mld_r2']:.4f}",
        "=================================================================="
    ])


    for line in summary_lines:
        print(line)

    # Save to text log
    eval_log_path = os.path.join(res_dirs['log_dir'], "eval.log")
    with open(eval_log_path, "a", encoding="utf-8") as f_eval:
        f_eval.write("\n".join(summary_lines) + "\n\n")
    print(f"\n[Log Saved] Full summary appended to: {eval_log_path}")

    # Export structured detailed metrics to JSON
    json_metrics = {
        "tag": res_dirs['tag'],
        "checkpoint": ckpt_path,
        "mode": args.mode,
        "sample_count": len(test_dataset),
        "global_metrics": {
            "temp": {"rmse": rmse_t, "mae": mae_t, "r2": r2_t},
            "sal": {"rmse": rmse_s, "mae": mae_s, "r2": r2_s}
        },
        "regime_metrics": {
            "temp": regimes_t,
            "sal": regimes_s
        },
        "layer_metrics": {
            "depths": [float(z) for z in depths],
            "temp": layer_metrics_t,
            "sal": layer_metrics_s
        },
        "physics_metrics": {
            "density_inversion": {
                "pred_rate_percent": dir_pred['inversion_rate_percent'],
                "true_rate_percent": dir_true['inversion_rate_percent'],
                "pred_inversions": dir_pred['total_inversions'],
                "true_inversions": dir_true['total_inversions']
            },
            "temp_monotonicity_violation": {
                "pred_rate_percent": mono_pred['violation_rate_percent'],
                "true_rate_percent": mono_true['violation_rate_percent']
            },
            "mld": {
                "rmse_m": float(mld_metrics['mld_rmse']),
                "mae_m": float(mld_metrics['mld_mae']),
                "r2": float(mld_metrics['mld_r2'])
            }
        }
    }

    def _json_serial(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, '__float__'):
            return float(obj)
        if hasattr(obj, '__int__'):
            return int(obj)
        return str(obj)

    json_path = os.path.join(res_dirs['log_dir'], "metrics_detailed.json")
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(json_metrics, f_json, indent=2, ensure_ascii=False, default=_json_serial)
    print(f"[JSON Saved] Detailed layer metrics exported to: {json_path}")



if __name__ == "__main__":
    evaluate()

