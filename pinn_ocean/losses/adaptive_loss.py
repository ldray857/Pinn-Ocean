# -*- coding: utf-8 -*-
"""
Adaptive Multi-Objective Loss Module for Pinn-Ocean
Balances data-driven MSE loss and physics-informed constraint loss
using learnable homoscedastic uncertainty / dual optimization weights.
"""

import torch
import torch.nn as nn


class AdaptiveMultiObjectiveLoss(nn.Module):
    """
    Adaptive Multi-Objective Joint Optimization:
    L_total = exp(-w1) * L_data + w1 + exp(-w2) * L_phy + w2
    
    where w1, w2 are learnable homoscedastic log-variance / dual variables balancing data fidelity
    and thermodynamic regularization without manual weight tuning.
    """
    def __init__(self, init_w1=0.0, init_w2=0.0):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(init_w1, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(init_w2, dtype=torch.float32))

    def forward(self, loss_data, loss_phy):
        """
        Args:
            loss_data: MSE loss between predicted and ground truth fields
            loss_phy: Combined physics constraint loss (temperature + density)
            
        Returns:
            total_loss: Joint loss scalar for backpropagation
            w1_val: Current value of w1
            w2_val: Current value of w2
        """
        w1_clamped = torch.clamp(self.w1, min=-10.0, max=10.0)
        w2_clamped = torch.clamp(self.w2, min=-10.0, max=10.0)
        total_loss = (
            torch.exp(-w1_clamped) * loss_data + w1_clamped +
            torch.exp(-w2_clamped) * loss_phy + w2_clamped
        )
        return total_loss, w1_clamped.item(), w2_clamped.item()
