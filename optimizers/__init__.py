"""
Оптимизаторы и вспомогательные инструменты для машинного обучения.

Этот модуль включает в себя реализации популярных оптимизаторов, таких как:
- Стандартный градиентный спуск (Gradient Descent)
- Стохастический градиентный спуск (SGD)
- RMSProp
- AdamW
- Lion
- Adan
- MARS

Также присутствует вспомогательная функция `timed`, которая используется для замера времени выполнения операций.

Каждый оптимизатор реализован как класс, и их можно использовать для настройки параметров обучения нейронных сетей или других моделей машинного обучения.
"""

from .SGD import StochasticGradientDescent
from .GD import GradientDescent
from .RMSp import RMSProp
from .Aw import AdamW
from .L import Lion
from .An import Adan
from .Make_vAriance_Reduction_Shine import MARS
from .dtime import timed 

__all__ = [
    "StochasticGradientDescent",
    "GradientDescent",
    "RMSProp",
    "AdamW",
    "Lion",
    "Adan",
    "MARS"
]
