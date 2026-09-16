# -*- coding: utf-8 -*-
"""
Visualization Subpackage for Pinn-Ocean
Modular high-resolution scientific plotting routines for oceanographic evaluations.
"""

from .profiles import plot_vertical_profiles, plot_multi_station_profiles
from .ts_diagram import plot_ts_diagram
from .scatter_density import plot_scatter_density
from .mld import plot_mld_validation
from .volumetric_3d import plot_3d_thermohaline_box, plot_3d_isotherm_surface
from .sections import plot_vertical_section, plot_layer_metrics_profile
from .horizontal_layers import plot_depth_layers_grid

__all__ = [
    "plot_vertical_profiles",
    "plot_multi_station_profiles",
    "plot_ts_diagram",
    "plot_scatter_density",
    "plot_mld_validation",
    "plot_3d_thermohaline_box",
    "plot_3d_isotherm_surface",
    "plot_vertical_section",
    "plot_layer_metrics_profile",
    "plot_depth_layers_grid"
]

