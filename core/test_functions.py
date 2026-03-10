"""Test functions for optimizer comparison (func + grad). No Streamlit dependency."""
import numpy as np
from typing import Dict, Any, List, Tuple

TEST_FUNCTION_NAMES: List[str] = [
    "Quadratic", "Rastrigin", "Rosenbrock", "Himmelblau",
    "Ackley", "Griewank", "Schwefel", "Levy", "Beale",
]


def get_test_functions() -> Dict[str, Dict[str, Any]]:
    """Return dict of name -> {func, grad} for point evaluation.

    Returns:
        Dict mapping function name to dict with 'func' and 'grad' callables.
    """
    return {
        "Quadratic": {
            "func": lambda p: np.sum(p**2),
            "grad": lambda p: 2 * p
        },
        "Rastrigin": {
            "func": lambda p: 10 * 2 + (p[0]**2 - 10*np.cos(2*np.pi*p[0])) + (p[1]**2 - 10*np.cos(2*np.pi*p[1])),
            "grad": lambda p: np.array([2*p[0] + 20*np.pi*np.sin(2*np.pi*p[0]), 2*p[1] + 20*np.pi*np.sin(2*np.pi*p[1])])
        },
        "Rosenbrock": {
            "func": lambda p: (1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2,
            "grad": lambda p: np.array([-2*(1-p[0]) - 400*p[0]*(p[1]-p[0]**2), 200*(p[1]-p[0]**2)])
        },
        "Himmelblau": {
            "func": lambda p: (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2,
            "grad": lambda p: np.array([4*p[0]*(p[0]**2 + p[1] - 11) + 2*(p[0] + p[1]**2 - 7),
                                        2*(p[0]**2 + p[1] - 11) + 4*p[1]*(p[0] + p[1]**2 - 7)])
        },
        "Ackley": {
            "func": lambda p: -20 * np.exp(-0.2 * np.sqrt(0.5 * (p[0]**2 + p[1]**2))) - \
                              np.exp(0.5 * (np.cos(2*np.pi*p[0]) + np.cos(2*np.pi*p[1]))) + np.e + 20,
            "grad": lambda p: np.array([
                2 * np.exp(-0.2 * np.sqrt(0.5 * (p[0]**2 + p[1]**2))) * p[0] / np.sqrt(0.5 * (p[0]**2 + p[1]**2)) + \
                np.pi * np.sin(2*np.pi*p[0]) * np.exp(0.5 * (np.cos(2*np.pi*p[0]) + np.cos(2*np.pi*p[1]))),
                2 * np.exp(-0.2 * np.sqrt(0.5 * (p[0]**2 + p[1]**2))) * p[1] / np.sqrt(0.5 * (p[0]**2 + p[1]**2)) + \
                np.pi * np.sin(2*np.pi*p[1]) * np.exp(0.5 * (np.cos(2*np.pi*p[0]) + np.cos(2*np.pi*p[1])))
            ])
        },
        "Griewank": {
            "func": lambda p: (p[0]**2 + p[1]**2) / 4000 - np.cos(p[0]) * np.cos(p[1] / np.sqrt(2)) + 1,
            "grad": lambda p: np.array([
                p[0] / 2000 + np.sin(p[0]) * np.cos(p[1] / np.sqrt(2)),
                p[1] / 2000 + np.cos(p[0]) * np.sin(p[1] / np.sqrt(2)) / np.sqrt(2)
            ])
        },
        "Schwefel": {
            "func": lambda p: 418.9829 * 2 - p[0] * np.sin(np.sqrt(np.abs(p[0]))) - p[1] * np.sin(np.sqrt(np.abs(p[1]))),
            "grad": lambda p: np.array([
                -np.sin(np.sqrt(np.abs(p[0]))) - (p[0] * np.cos(np.sqrt(np.abs(p[0])))) / (2 * np.sqrt(np.abs(p[0]))) if p[0] != 0 else 0,
                -np.sin(np.sqrt(np.abs(p[1]))) - (p[1] * np.cos(np.sqrt(np.abs(p[1])))) / (2 * np.sqrt(np.abs(p[1]))) if p[1] != 0 else 0
            ])
        },
        "Levy": {
            "func": lambda p: np.sin(np.pi * (1 + (p[0] - 1) / 4))**2 + \
                              ((1 + (p[0] - 1) / 4) - 1)**2 * (1 + 10 * np.sin(np.pi * (1 + (p[0] - 1) / 4) + 1)**2) + \
                              ((1 + (p[1] - 1) / 4) - 1)**2 * (1 + np.sin(2 * np.pi * (1 + (p[1] - 1) / 4))**2),
            "grad": lambda p: np.array([
                np.pi * np.cos(np.pi * (1 + (p[0] - 1) / 4)) * np.sin(np.pi * (1 + (p[0] - 1) / 4)) / 2 + \
                (p[0] - 1) / 8 * (1 + 10 * np.sin(np.pi * (1 + (p[0] - 1) / 4) + 1)**2) + \
                5 * np.pi * (p[0] - 1)**2 / 16 * np.cos(np.pi * (1 + (p[0] - 1) / 4) + 1) * \
                np.sin(np.pi * (1 + (p[0] - 1) / 4) + 1),
                (p[1] - 1) / 8 * (1 + np.sin(2 * np.pi * (1 + (p[1] - 1) / 4))**2) + \
                np.pi * (p[1] - 1)**2 / 8 * np.cos(2 * np.pi * (1 + (p[1] - 1) / 4)) * \
                np.sin(2 * np.pi * (1 + (p[1] - 1) / 4))
            ])
        },
        "Beale": {
            "func": lambda p: (1.5 - p[0] + p[0]*p[1])**2 + (2.25 - p[0] + p[0]*p[1]**2)**2 + (2.625 - p[0] + p[0]*p[1]**3)**2,
            "grad": lambda p: np.array([
                2 * (1.5 - p[0] + p[0]*p[1]) * (-1 + p[1]) + \
                2 * (2.25 - p[0] + p[0]*p[1]**2) * (-1 + p[1]**2) + \
                2 * (2.625 - p[0] + p[0]*p[1]**3) * (-1 + p[1]**3),
                2 * (1.5 - p[0] + p[0]*p[1]) * p[0] + \
                2 * (2.25 - p[0] + p[0]*p[1]**2) * (2*p[0]*p[1]) + \
                2 * (2.625 - p[0] + p[0]*p[1]**3) * (3*p[0]*p[1]**2)
            ])
        }
    }


def generate_test_data(name: str, res: int, bnd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate mesh X, Y, Z for surface/contour plots.

    Args:
        name: Test function name.
        res: Grid resolution.
        bnd: Axis bound (-bnd to bnd).

    Returns:
        Tuple of (X, Y, Z) mesh arrays. Pure NumPy, no caching.
    """
    x = np.linspace(-bnd, bnd, res)
    y = np.linspace(-bnd, bnd, res)
    X, Y = np.meshgrid(x, y)
    if name == "Quadratic":
        Z = X**2 + Y**2
    elif name == "Rastrigin":
        Z = 10 * 2 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))
    elif name == "Rosenbrock":
        Z = (1 - X)**2 + 100 * (Y - X**2)**2
    elif name == "Himmelblau":
        Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
    elif name == "Ackley":
        Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2))) - \
            np.exp(0.5 * (np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + np.e + 20
    elif name == "Griewank":
        Z = (X**2 + Y**2) / 4000 - np.cos(X) * np.cos(Y / np.sqrt(2)) + 1
    elif name == "Schwefel":
        Z = 418.9829 * 2 - X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))
    elif name == "Levy":
        w_x = 1 + (X - 1) / 4
        w_y = 1 + (Y - 1) / 4
        Z = np.sin(np.pi * w_x)**2 + \
            (w_x - 1)**2 * (1 + 10 * np.sin(np.pi * w_x + 1)**2) + \
            (w_y - 1)**2 * (1 + np.sin(2 * np.pi * w_y)**2)
    elif name == "Beale":
        Z = (1.5 - X + X*Y)**2 + (2.25 - X + X*Y**2)**2 + (2.625 - X + X*Y**3)**2
    else:
        Z = X**2 + Y**2
    return X, Y, Z
