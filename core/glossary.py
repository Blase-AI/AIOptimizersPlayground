"""Glossary entries (term → definition) and parameter help strings. Bilingual RU/EN."""

from typing import Dict, List, Tuple

from core.i18n import get_lang

_GLOSSARY: Dict[str, Dict[str, Tuple[str, str]]] = {
    "learning_rate": {
        "ru": (
            "Learning rate (скорость обучения)",
            "Множитель шага обновления параметров. Слишком большой — нестабильность или расхождение; "
            "слишком маленький — медленная сходимость.",
        ),
        "en": (
            "Learning rate",
            "Step size multiplier for parameter updates. Too large — instability or divergence; "
            "too small — slow convergence.",
        ),
    },
    "momentum": {
        "ru": (
            "Momentum (моментум, β)",
            "Инерция обновления: усреднение прошлых градиентов. Ускоряет сходимость и сглаживает осцилляции. "
            "Обычно 0.9 для SGD.",
        ),
        "en": (
            "Momentum (β)",
            "Update inertia: averaging past gradients. Speeds up convergence and smooths oscillations. "
            "Typically 0.9 for SGD.",
        ),
    },
    "weight_decay": {
        "ru": (
            "Weight decay",
            "L2-регуляризация параметров: штраф за большие веса. Улучшает обобщение и в AdamW применяется "
            "не к градиенту, а к весам напрямую (decoupled).",
        ),
        "en": (
            "Weight decay",
            "L2 regularization of parameters: penalty on large weights. Improves generalization; in AdamW "
            "applied to weights directly (decoupled), not to the gradient.",
        ),
    },
    "beta1_beta2": {
        "ru": (
            "Beta1, Beta2",
            "Коэффициенты экспоненциального скользящего среднего в Adam-семействе: beta1 — для первого момента (градиент), "
            "beta2 — для второго момента (квадраты градиентов). Типично 0.9 и 0.999.",
        ),
        "en": (
            "Beta1, Beta2",
            "Exponential moving average coefficients in Adam family: beta1 for first moment (gradient), "
            "beta2 for second moment (squared gradients). Typically 0.9 and 0.999.",
        ),
    },
    "trust_coeff": {
        "ru": (
            "Trust coefficient (LARS)",
            "В LARS задаёт соотношение между нормой весов и нормой градиента при адаптивном масштабировании шага. "
            "Позволяет использовать больший learning rate для больших весов.",
        ),
        "en": (
            "Trust coefficient (LARS)",
            "In LARS, controls the ratio of weight norm to gradient norm for adaptive step scaling. "
            "Allows using a larger learning rate for larger weights.",
        ),
    },
    "adaptive_optimizer": {
        "ru": (
            "Адаптивный оптимизатор",
            "Метод, который подстраивает величину шага по каждой координате (например, Adam, RMSProp, Adagrad). "
            "Часто быстрее сходятся, но требуют больше памяти.",
        ),
        "en": (
            "Adaptive optimizer",
            "A method that adapts the step size per coordinate (e.g. Adam, RMSProp, Adagrad). "
            "Often converges faster but uses more memory.",
        ),
    },
    "global_minimum": {
        "ru": (
            "Глобальный минимум",
            "Точка, в которой функция принимает наименьшее значение на всей области определения. "
            "Оптимизаторы стремятся к локальному или глобальному минимуму в зависимости от ландшафта.",
        ),
        "en": (
            "Global minimum",
            "The point where the function attains its smallest value over the domain. "
            "Optimizers aim for local or global minimum depending on the landscape.",
        ),
    },
    "gradient": {
        "ru": (
            "Градиент",
            "Вектор частных производных функции по параметрам. Показывает направление наискорейшего роста; "
            "оптимизаторы двигаются в направлении антиградиента (спуск).",
        ),
        "en": (
            "Gradient",
            "Vector of partial derivatives of the function w.r.t. parameters. Points in the direction of steepest increase; "
            "optimizers move in the opposite direction (descent).",
        ),
    },
}

_PARAM_HELP: Dict[str, Dict[str, str]] = {
    "beta1": {
        "ru": "Экспоненциальное затухание для первого момента (среднее градиентов).",
        "en": "Exponential decay for the first moment (gradient average).",
    },
    "beta2": {
        "ru": "Экспоненциальное затухание для второго момента (среднее квадратов градиентов).",
        "en": "Exponential decay for the second moment (squared gradient average).",
    },
    "beta": {
        "ru": "Коэффициент момента в Lion (аналогично momentum).",
        "en": "Momentum coefficient in Lion (similar to momentum).",
    },
    "trust_coeff": {
        "ru": "Коэффициент доверия в LARS: масштаб шага относительно норм весов и градиента.",
        "en": "LARS trust coefficient: step scale relative to weight and gradient norms.",
    },
}


def get_glossary_entries(lang: str = None) -> List[Tuple[str, str]]:
    """Return (term, definition) pairs for the Glossary page in the given or current language.

    Args:
        lang: 'ru' or 'en'. If None, uses current UI language from session.

    Returns:
        List of (term, definition) for the chosen language.
    """
    if lang is None:
        lang = get_lang()
    lang = "en" if lang not in ("ru", "en") else lang
    return [
        (entry[lang][0], entry[lang][1])
        for entry in _GLOSSARY.values()
        if lang in entry
    ]


def get_param_help(param_name: str, lang: str = None) -> str:
    """Return help string for a sidebar parameter in the given or current language.

    Args:
        param_name: Key used in UI (e.g. 'beta1', 'trust_coeff').
        lang: 'ru' or 'en'. If None, uses current UI language.

    Returns:
        Help text or empty string if unknown.
    """
    if lang is None:
        lang = get_lang()
    lang = "en" if lang not in ("ru", "en") else lang
    return _PARAM_HELP.get(param_name, {}).get(lang, "")
