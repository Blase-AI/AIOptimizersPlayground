"""Preset scenarios for quick comparison and learning."""

from typing import Any, Dict, List

PRESETS: List[Dict[str, Any]] = [
    {
        "name": "Свои настройки",
        "optimizers": ["AdamW"],
        "test_func": "Quadratic",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "iterations": 100,
    },
    {
        "name": "SGD vs Adam на Rastrigin",
        "optimizers": ["SGD", "Adam"],
        "test_func": "Rastrigin",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "iterations": 200,
    },
    {
        "name": "Влияние момента: SGD на Rosenbrock",
        "optimizers": ["SGD"],
        "test_func": "Rosenbrock",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "iterations": 150,
    },
    {
        "name": "Адаптивные: Adam vs AdamW vs Lion",
        "optimizers": ["Adam", "AdamW", "Lion"],
        "test_func": "Ackley",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "iterations": 250,
    },
    {
        "name": "Классика: GD vs SGD на Quadratic",
        "optimizers": ["GD", "SGD"],
        "test_func": "Quadratic",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "iterations": 80,
    },
    {
        "name": "Сложный ландшафт: Rosenbrock",
        "optimizers": ["SGD", "Adam", "LARS"],
        "test_func": "Rosenbrock",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "iterations": 300,
    },
    {
        "name": "Мультимодальная: Rastrigin",
        "optimizers": ["Adam", "AdamW", "Sophia"],
        "test_func": "Rastrigin",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "iterations": 400,
    },
]


def get_preset_names() -> List[str]:
    """Return list of preset display names."""
    return [p["name"] for p in PRESETS]


def get_preset_by_name(name: str) -> Dict[str, Any] | None:
    """Return preset dict by name or None."""
    for p in PRESETS:
        if p["name"] == name:
            return p
    return None
