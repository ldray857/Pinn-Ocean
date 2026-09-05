# -*- coding: utf-8 -*-
"""
Main Training Pipeline for Pinn-Ocean (Swin-Ocean-PINN)
Executes physics-informed neural network training on ocean thermohaline fields.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.default_config import ModelConfig, PhysicsConfig, TrainConfig, DataConfig
from pinn_ocean.models.swin_ocean_pinn import SwinOceanPINN
from pinn_ocean.losses.physics_loss import OceanPhysicsLoss
from pinn_ocean.losses.adaptive_loss import AdaptiveMultiObjectiveLoss
from pinn_ocean.datasets.ocean_dataset import OceanContinuousDataset
from pinn_ocean.utils.metrics import calc_rmse, calc_r2


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin-Ocean-PINN Model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--sampling_points", type=int, default=800, help="Number of spatial sampling points")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("==================================================================")
    print("                Swin-Ocean-PINN Training Pipeline                 ")
    print(f" Computing Device: {device} | Total Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("==================================================================")

    # 1. Dataset & DataLoader
    data_cfg = DataConfig()
    try:
        train_dataset = OceanContinuousDataset(data_cfg.sla_path, data_cfg.gt_path, mode='train')
        val_dataset = OceanContinuousDataset(data_cfg.sla_path, data_cfg.gt_path, mode='val')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"[Dataset] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    except Exception as e:
        print(f"[Warning] Real dataset could not be loaded ({e}).")
        print("Please check NetCDF file paths or run demo_test.py for synthetic verification.")
        return

    # 2. Model, Losses, and Optimizers
    model_cfg = ModelConfig()
    model = SwinOceanPINN(
        in_channels=model_cfg.in_channels,
        embed_dim=model_cfg.embed_dim,
        window_size=model_cfg.window_size,
        physics_hidden_dim=model_cfg.physics_hidden_dim,
        out_dim=model_cfg.out_dim
    ).to(device)

    phy_cfg = PhysicsConfig()
    phy_loss_fn = OceanPhysicsLoss(
        temp_grad_threshold=phy_cfg.temp_grad_threshold,
        enable_density=phy_cfg.enable_density_loss
    ).to(device)
    
    adaptive_loss_fn = AdaptiveMultiObjectiveLoss(
        init_w1=phy_cfg.init_log_var_data,
        init_w2=phy_cfg.init_log_var_phy
    ).to(device)

    mse_loss_fn = nn.MSELoss()

    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr, 'weight_decay': 1e-4},
        {'params': adaptive_loss_fn.parameters(), 'lr': 1e-3}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    best_val_loss = float('inf')

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_data_loss_sum = 0.0
        train_phy_loss_sum = 0.0

        for x_8ch, y_3d in train_loader:
            x_8ch, y_3d = x_8ch.to(device), y_3d.to(device)

            # Prepare continuous vertical depth coordinate with gradient tracking
            z_raw = train_dataset.get_depth_tensor().to(device)
            z_norm = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-6)
            z_norm.requires_grad_(True)

            # Random spatial sampling to maintain efficient VRAM footprint
            total_points = y_3d.shape[3] * y_3d.shape[4]
            sample_idx = torch.randperm(total_points)[:args.sampling_points].to(device)

            optimizer.zero_grad()

            # Forward pass on sampled points
            preds = model(x_8ch, z_norm, sample_idx=sample_idx)
            y_target = y_3d.view(y_3d.shape[0], 2, y_3d.shape[2], -1)[:, :, :, sample_idx]

            # 1. Data-driven fidelity loss
            loss_data = mse_loss_fn(preds, y_target)

            # 2. Physics-informed constraint loss
            loss_phy, loss_dict = phy_loss_fn(
                preds, z_norm, stats=train_dataset.stats, z_raw=z_raw
            )

            # 3. Joint adaptive multi-objective loss
            total_loss, w1, w2 = adaptive_loss_fn(loss_data, loss_phy)

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_data_loss_sum += loss_data.item()
            train_phy_loss_sum += loss_phy.item()

        # Validation step
        model.eval()
        val_mse_sum = 0.0
        with torch.no_grad():
            for x_8ch, y_3d in val_loader:
                x_8ch, y_3d = x_8ch.to(device), y_3d.to(device)
                z_raw = val_dataset.get_depth_tensor().to(device)
                z_norm = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-6)

                preds = model(x_8ch, z_norm, sample_idx=None)
                val_mse_sum += mse_loss_fn(preds, y_3d).item()

        avg_train_mse = train_data_loss_sum / max(1, len(train_loader))
        avg_train_phy = train_phy_loss_sum / max(1, len(train_loader))
        avg_val_mse = val_mse_sum / max(1, len(val_loader))

        scheduler.step(avg_val_mse)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:03d}/{args.epochs}] | "
                f"Train MSE: {avg_train_mse:.5f} | "
                f"Phy Loss: {avg_train_phy:.5f} | "
                f"Val MSE: {avg_val_mse:.5f} | "
                f"Dual Weights (w1/w2): {w1:.2f}/{w2:.2f}"
            )

            if avg_val_mse < best_val_loss:
                best_val_loss = avg_val_mse
                save_path = os.path.join(args.output_dir, "swin_ocean_pinn_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'adaptive_loss_state_dict': adaptive_loss_fn.state_dict(),
                    'val_loss': best_val_loss,
                    'stats': train_dataset.stats
                }, save_path)
                print(f"--> [Checkpoint] Updated optimal model saved to {save_path}")

    print("\n[Complete] Training finished successfully.")


if __name__ == "__main__":
    main()
