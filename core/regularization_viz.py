"""Regularization visualizations for Glossary: L2, L1, Elastic net.

Builds Plotly figures with loss contours, constraint ball (circle/diamond),
regularization path θ*(λ), and current optimum. Used on the Glossary page
with a Streamlit slider for λ.
"""
import numpy as np
import plotly.graph_objects as go

DEFAULT_C = np.array([2.0, 2.0])
GRID_SIZE = 80
AXIS_LIM = 4.0


def _loss_quadratic(theta: np.ndarray, c: np.ndarray) -> float:
    """Quadratic loss L(θ) = 0.5||θ - c||²."""
    return 0.5 * np.sum((theta - c) ** 2)


def _optimal_l2(c: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form optimum for L(θ) + (λ/2)||θ||²: θ* = c/(1+λ)."""
    if lam <= 0:
        return c.copy()
    return c / (1.0 + lam)


def _optimal_l1(c: np.ndarray, lam: float) -> np.ndarray:
    """Soft-threshold optimum for L1 penalty: θ*_i = sign(c_i) max(|c_i|-λ, 0)."""
    if lam <= 0:
        return c.copy()
    return np.sign(c) * np.maximum(np.abs(c) - lam, 0.0)


def _optimal_elastic(c: np.ndarray, lam: float, l1_ratio: float = 0.5) -> np.ndarray:
    """Approximate optimum for elastic net via proximal gradient steps."""
    theta = c.copy()
    for _ in range(50):
        grad = (theta - c) + (1.0 - l1_ratio) * lam * theta
        theta = theta - 0.1 * grad
        theta = np.sign(theta) * np.maximum(np.abs(theta) - 0.1 * l1_ratio * lam, 0.0)
    return theta


def _make_contour_grid(c: np.ndarray, axis_lim: float = AXIS_LIM, n: int = GRID_SIZE):
    """Return (X, Y, Z) mesh for L(θ) = 0.5||θ - c||² over [-axis_lim, axis_lim]²."""
    t = np.linspace(-axis_lim, axis_lim, n)
    X, Y = np.meshgrid(t, t)
    Z = 0.5 * ((X - c[0]) ** 2 + (Y - c[1]) ** 2)
    return X, Y, Z


def _l2_ball_boundary(r: float, n: int = 100):
    """Return (x, y) arrays of points on the circle ‖θ‖₂ = r."""
    angles = np.linspace(0, 2 * np.pi, n)
    return r * np.cos(angles), r * np.sin(angles)


def _l1_ball_boundary(r: float, n: int = 40) -> tuple:
    """Return (x, y) arrays of points on the L1 ball boundary (diamond)."""
    if r <= 0:
        return np.array([0.0]), np.array([0.0])
    n2 = max(2, n // 4)
    x1 = np.linspace(r, 0, n2)
    y1 = np.linspace(0, r, n2)
    x2 = np.linspace(0, -r, n2)
    y2 = np.linspace(r, 0, n2)
    x3 = np.linspace(-r, 0, n2)
    y3 = np.linspace(0, -r, n2)
    x4 = np.linspace(0, r, n2)
    y4 = np.linspace(-r, 0, n2)
    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    return x, y


def _path_l2(c: np.ndarray, n: int = 50):
    """Return (x, y) of θ*(λ) for λ in [0, 3] (L2 path from c to origin)."""
    lams = np.linspace(0, 3, n)
    pts = np.array([_optimal_l2(c, lam) for lam in lams])
    return pts[:, 0], pts[:, 1]


def _path_l1(c: np.ndarray, n: int = 50):
    """Return (x, y) of θ*(λ) for λ in [0, 3] (L1 soft-threshold path)."""
    lams = np.linspace(0, 3, n)
    pts = np.array([_optimal_l1(c, lam) for lam in lams])
    return pts[:, 0], pts[:, 1]


def _path_elastic(c: np.ndarray, l1_ratio: float, n: int = 50):
    """Return (x, y) of θ*(λ) for λ in [0, 3] (Elastic net path)."""
    lams = np.linspace(0, 3, n)
    pts = np.array([_optimal_elastic(c, lam, l1_ratio) for lam in lams])
    return pts[:, 0], pts[:, 1]


def build_l2_figure(lam: float, c: np.ndarray = DEFAULT_C) -> go.Figure:
    """Build L2 (Ridge) Plotly figure: contours, path θ*(λ), L2 ball, θ*, c.

    Args:
        lam: Regularization strength (≥ 0).
        c: Center of unregularized loss (default [2, 2]).

    Returns:
        Plotly Figure with axes θ₁, θ₂.
    """
    X, Y, Z = _make_contour_grid(c)
    theta_star = _optimal_l2(c, lam)
    R = np.linalg.norm(theta_star)
    cx, cy = _l2_ball_boundary(R)
    path_x, path_y = _path_l2(c)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=X[0], y=Y[:, 0], z=Z,
            contours=dict(coloring="lines", showlabels=True),
            line=dict(width=2),
            name="Loss L(θ)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path_x, y=path_y, mode="lines",
            line=dict(color="rgba(200,80,80,0.7)", width=3),
            name="Путь θ*(λ), λ: 0→3",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="red", width=2.5, dash="dash"),
            name="L2 ball ‖θ‖₂ = R",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[theta_star[0]], y=[theta_star[1]],
            mode="markers+text",
            marker=dict(size=16, color="red", symbol="circle", line=dict(width=2, color="white")),
            text=["θ*"], textposition="top center",
            name="θ* (оптимум)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[c[0]], y=[c[1]],
            mode="markers",
            marker=dict(size=12, color="gray", symbol="x"),
            name="c (минимум без регуляризации)",
        )
    )
    fig.update_layout(
        title=dict(text=f"L2 (Ridge): λ = {lam:.2f}  →  θ* = c/(1+λ)"),
        xaxis=dict(title="θ₁", range=[-AXIS_LIM, AXIS_LIM], scaleanchor="y"),
        yaxis=dict(title="θ₂", range=[-AXIS_LIM, AXIS_LIM]),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=56, r=24, t=56, b=48),
        height=440,
    )
    return fig


def build_l1_figure(lam: float, c: np.ndarray = DEFAULT_C) -> go.Figure:
    """Build L1 (Lasso) Plotly figure: contours, path θ*(λ), L1 ball (diamond), θ*, c.

    Args:
        lam: Regularization strength (≥ 0).
        c: Center of unregularized loss.

    Returns:
        Plotly Figure with axes θ₁, θ₂.
    """
    X, Y, Z = _make_contour_grid(c)
    theta_star = _optimal_l1(c, lam)
    R = np.abs(theta_star[0]) + np.abs(theta_star[1])
    if R < 1e-6:
        R = 0.5
    dx, dy = _l1_ball_boundary(R)
    path_x, path_y = _path_l1(c)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=X[0], y=Y[:, 0], z=Z,
            contours=dict(coloring="lines", showlabels=True),
            line=dict(width=2),
            name="Loss L(θ)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path_x, y=path_y, mode="lines",
            line=dict(color="rgba(0,120,80,0.8)", width=3),
            name="Путь θ*(λ), λ: 0→3",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dx, y=dy, mode="lines",
            line=dict(color="darkgreen", width=2.5, dash="dash"),
            name="L1 ball ‖θ‖₁ = R (ромб)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[theta_star[0]], y=[theta_star[1]],
            mode="markers+text",
            marker=dict(size=16, color="darkgreen", symbol="circle", line=dict(width=2, color="white")),
            text=["θ*"], textposition="top center",
            name="θ* (оптимум)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[c[0]], y=[c[1]],
            mode="markers",
            marker=dict(size=12, color="gray", symbol="x"),
            name="c (минимум без регуляризации)",
        )
    )
    fig.update_layout(
        title=dict(text=f"L1 (Lasso): λ = {lam:.2f}  →  θ* = soft-threshold(c, λ)"),
        xaxis=dict(title="θ₁", range=[-AXIS_LIM, AXIS_LIM], scaleanchor="y"),
        yaxis=dict(title="θ₂", range=[-AXIS_LIM, AXIS_LIM]),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=56, r=24, t=56, b=48),
        height=440,
    )
    return fig


def build_elastic_figure(lam: float, l1_ratio: float = 0.5, c: np.ndarray = DEFAULT_C) -> go.Figure:
    """Build Elastic net Plotly figure: contours, path θ*(λ), L1/L2 balls, θ*, c.

    Args:
        lam: Regularization strength (≥ 0).
        l1_ratio: L1 weight in [0, 1]; rest is L2.
        c: Center of unregularized loss.

    Returns:
        Plotly Figure with axes θ₁, θ₂.
    """
    X, Y, Z = _make_contour_grid(c)
    theta_star = _optimal_elastic(c, lam, l1_ratio)
    R1 = np.abs(theta_star[0]) + np.abs(theta_star[1])
    R2 = np.linalg.norm(theta_star)
    if R1 < 1e-6:
        R1, R2 = 0.5, 0.5
    dx, dy = _l1_ball_boundary(R1)
    cx, cy = _l2_ball_boundary(R2)
    path_x, path_y = _path_elastic(c, l1_ratio)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=X[0], y=Y[:, 0], z=Z,
            contours=dict(coloring="lines", showlabels=True),
            line=dict(width=2),
            name="Loss L(θ)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path_x, y=path_y, mode="lines",
            line=dict(color="rgba(120,80,160,0.8)", width=3),
            name="Путь θ*(λ), λ: 0→3",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dx, y=dy, mode="lines",
            line=dict(color="purple", width=2, dash="dot"),
            name="L1 ‖θ‖₁ = R₁",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="orange", width=2, dash="dot"),
            name="L2 ‖θ‖₂ = R₂",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[theta_star[0]], y=[theta_star[1]],
            mode="markers+text",
            marker=dict(size=16, color="purple", symbol="circle", line=dict(width=2, color="white")),
            text=["θ*"], textposition="top center",
            name="θ* (оптимум)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[c[0]], y=[c[1]],
            mode="markers",
            marker=dict(size=12, color="gray", symbol="x"),
            name="c (минимум без регуляризации)",
        )
    )
    fig.update_layout(
        title=dict(text=f"Elastic net: λ = {lam:.2f}, l1_ratio = {l1_ratio:.2f}"),
        xaxis=dict(title="θ₁", range=[-AXIS_LIM, AXIS_LIM], scaleanchor="y"),
        yaxis=dict(title="θ₂", range=[-AXIS_LIM, AXIS_LIM]),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=56, r=24, t=56, b=48),
        height=440,
    )
    return fig
