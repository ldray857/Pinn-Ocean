# -*- coding: utf-8 -*-
"""
I/O and Result Directory Organization Module for Pinn-Ocean
Standardizes directory layout:
<result_dir>/<year_tag>/
    ├── pic/          (Visualization figures)
    ├── log/          (Training and evaluation logs)
    ├── con/          (Reconstructed 3D NetCDF fields)
    └── checkpoints/  (Trained weights .pth)
"""

import os
from typing import Optional, List, Dict, Any


def get_result_dirs(
    result_dir: str = "result",
    tag: Optional[str] = None,
    dataset: Optional[Any] = None,
    data_dir: Optional[str] = None,
    years: Optional[List[int]] = None
) -> Dict[str, str]:
    """
    Resolves and creates standard output directory structure:
    result/<year_tag>/
        ├── pic/          (figures)
        ├── log/          (logs)
        ├── con/          (reconstructions)
        └── checkpoints/  (weights)
    """
    if tag:
        year_tag = tag
    elif years:
        sorted_y = sorted(list(set(int(y) for y in years)))
        year_tag = f"{sorted_y[0]}_{sorted_y[-1]}" if len(sorted_y) > 1 else str(sorted_y[0])
    elif dataset is not None:
        full_times = None
        if hasattr(dataset, "gt_ds_full") and hasattr(dataset.gt_ds_full, "time"):
            full_times = dataset.gt_ds_full.time.values
        elif hasattr(dataset, "times") and len(dataset.times) > 0:
            full_times = dataset.times

        if full_times is not None and len(full_times) > 0:
            d_years = sorted(list(set(int(str(t)[:4]) for t in full_times if str(t)[:4].isdigit())))
            if d_years:
                year_tag = f"{d_years[0]}_{d_years[-1]}" if len(d_years) > 1 else str(d_years[0])
            else:
                year_tag = "experiment"
        else:
            year_tag = "experiment"
    elif data_dir:
        base = os.path.basename(os.path.abspath(data_dir))
        if any(c.isdigit() for c in base):
            year_tag = base
        else:
            year_tag = "experiment"
    else:
        year_tag = "experiment"

    exp_dir = os.path.join(result_dir, year_tag)
    pic_dir = os.path.join(exp_dir, "pic")
    log_dir = os.path.join(exp_dir, "log")
    con_dir = os.path.join(exp_dir, "con")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    os.makedirs(pic_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(con_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    return {
        "root": result_dir,
        "exp_dir": exp_dir,
        "pic_dir": pic_dir,
        "log_dir": log_dir,
        "con_dir": con_dir,
        "ckpt_dir": ckpt_dir,
        "tag": year_tag
    }
