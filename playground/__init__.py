"""Playground page: split into sidebar, visualization, metrics, description, guide, export.

Entry point is pages/1_Playground.py, which imports and calls the render functions.
"""

from .state import init_session_state
from .sidebar import render_sidebar
from .run import run_simulation_loop
from .visualization import render_visualization_tab
from .metrics import render_metrics_tab
from .description import render_description_tab
from .export import handle_export

__all__ = [
    "init_session_state",
    "render_sidebar",
    "run_simulation_loop",
    "render_visualization_tab",
    "render_metrics_tab",
    "render_description_tab",
    "handle_export",
]
