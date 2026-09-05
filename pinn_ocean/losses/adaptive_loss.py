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
    L_total = exp(-w1) * L_data + w1 + exp(w2) * L_phy + w2
    
    where w1, w2 are learnable dual variables balancing data fidelity
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
        total_loss = (
            torch.exp(-self.w1) * loss_data + self.w1 +
            torch.exp(self.w2) * loss_phy + self.w2
        )
        return total_loss, self.w1.item(), self.w2.item()
