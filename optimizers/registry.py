"""Registry of optimizers: names, classes, UI param specs, and factory.

To add an optimizer: add one entry to OPTIMIZER_REGISTRY and implement the class.
"""
from typing import Dict, Tuple, Any, Type, List, Union

from .base import BaseOptimizer
from .SGD import StochasticGradientDescent
from .GD import GradientDescent
from .rmsprop import RMSProp
from .adamw import AdamW
from .lion import Lion
from .adan import Adan
from .mars import MARS
from .adagrad import Adagrad
from .Sophia import Sophia
from .AMSGrad import AMSGrad
from .LARS import LARS
from .adam import Adam

ParamSpec = Tuple[float, float, float, float, str]

_LR = "_lr"
_MOMENTUM = "_momentum"
_WD = "_wd"


def _param(key: str, default: float) -> Tuple[str, float]:
    """Mark a kwarg to be taken from params[prefix+key] with default."""
    return (key, default)


OPTIMIZER_REGISTRY: Dict[
    str,
    Tuple[
        Type[BaseOptimizer],
        Dict[str, ParamSpec],
        Dict[str, Union[str, Tuple[str, float]]],
    ],
] = {
    "SGD": (StochasticGradientDescent, {}, {"learning_rate": _LR, "momentum": _MOMENTUM}),
    "GD": (GradientDescent, {}, {"learning_rate": _LR}),
    "RMSProp": (RMSProp, {}, {"learning_rate": _LR, "momentum": _MOMENTUM}),
    "AMSGrad": (
        AMSGrad,
        {
            "beta1": (0.0, 1.0, 0.9, 0.01, "%.2f"),
            "beta2": (0.0, 1.0, 0.999, 0.001, "%.3f"),
        },
        {"learning_rate": _LR, "beta1": _param("beta1", 0.9), "beta2": _param("beta2", 0.999)},
    ),
    "Adagrad": (Adagrad, {}, {"learning_rate": _LR}),
    "Adam": (
        Adam,
        {
            "beta1": (0.0, 1.0, 0.9, 0.01, "%.2f"),
            "beta2": (0.0, 1.0, 0.999, 0.001, "%.3f"),
        },
        {"learning_rate": _LR, "beta1": _param("beta1", 0.9), "beta2": _param("beta2", 0.999)},
    ),
    "AdamW": (
        AdamW,
        {
            "beta1": (0.0, 1.0, 0.9, 0.01, "%.2f"),
            "beta2": (0.0, 1.0, 0.999, 0.001, "%.3f"),
        },
        {
            "learning_rate": _LR,
            "beta1": _param("beta1", 0.9),
            "beta2": _param("beta2", 0.999),
            "weight_decay": _WD,
        },
    ),
    "Sophia": (
        Sophia,
        {
            "beta1": (0.0, 1.0, 0.9, 0.01, "%.2f"),
            "beta2": (0.0, 1.0, 0.999, 0.001, "%.3f"),
        },
        {"learning_rate": _LR, "beta1": _param("beta1", 0.9), "beta2": _param("beta2", 0.999)},
    ),
    "Lion": (
        Lion,
        {"beta": (0.0, 1.0, 0.9, 0.01, "%.2f")},
        {"learning_rate": _LR, "beta": _param("beta", 0.9), "weight_decay": _WD},
    ),
    "Adan": (Adan, {}, {"learning_rate": _LR, "weight_decay": _WD}),
    "MARS": (MARS, {}, {"learning_rate": _LR, "momentum": _MOMENTUM}),
    "LARS": (
        LARS,
        {"trust_coeff": (0.0001, 0.01, 0.001, 0.0001, "%.4f")},
        {
            "learning_rate": _LR,
            "momentum": _MOMENTUM,
            "trust_coeff": _param("trust_coeff", 0.001),
            "weight_decay": _WD,
        },
    ),
}


def get_optimizer_names() -> List[str]:
    """Return list of registered optimizer names."""
    return list(OPTIMIZER_REGISTRY.keys())


def get_param_spec(opt_name: str) -> Dict[str, ParamSpec]:
    """Return UI param spec for optimizer (min, max, default, step, format)."""
    if opt_name not in OPTIMIZER_REGISTRY:
        return {}
    return OPTIMIZER_REGISTRY[opt_name][1].copy()


def create_optimizer(
    opt_name: str,
    lr: float,
    momentum: float,
    wd: float,
    params: Dict[str, Any],
) -> BaseOptimizer:
    """Build optimizer instance from registry. params keys are like 'AdamW_beta1'."""
    if opt_name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    cls, _param_spec, build_spec = OPTIMIZER_REGISTRY[opt_name]
    prefix = f"{opt_name}_"
    kwargs: Dict[str, Any] = {}
    for key, value in build_spec.items():
        if value == _LR:
            kwargs[key] = lr
        elif value == _MOMENTUM:
            kwargs[key] = momentum
        elif value == _WD:
            kwargs[key] = wd
        elif isinstance(value, tuple):
            param_key, default = value
            kwargs[key] = params.get(f"{prefix}{param_key}", default)
        else:
            kwargs[key] = value
    return cls(**kwargs)
