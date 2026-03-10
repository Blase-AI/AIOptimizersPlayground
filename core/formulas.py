"""LaTeX/Markdown formulas for optimizers and test functions (Glossary and Description UI). Bilingual RU/EN."""
from typing import Dict, Any, Optional

from core.i18n import get_lang

OPTIMIZER_FORMULAS: Dict[str, str] = {
    "GD": r"""
**Правило обновления:**
- $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$
""",
    "SGD": r"""
**Правило обновления (с моментумом):**
- $v_t = \beta v_{t-1} + \nabla f(\theta_t)$
- $\theta_{t+1} = \theta_t - \eta v_t$
""",
    "RMSProp": r"""
**Правило обновления:**
- $s_t = \beta s_{t-1} + (1-\beta) g_t^2$
- $\theta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{s_t} + \epsilon}$
""",
    "Adam": r"""
**Правило обновления:**
- $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- $\hat{m}_t = m_t/(1-\beta_1^t)$, $\quad \hat{v}_t = v_t/(1-\beta_2^t)$
- $\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
""",
    "AdamW": r"""
**Правило обновления:** как Adam, плюс декуплированный weight decay:
- $\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$
""",
    "AMSGrad": r"""
**Правило обновления:** как Adam, но $\hat{v}_t = \max(\hat{v}_t, \hat{v}_{t-1})$ (незатухающая оценка второго момента).
""",
    "Adagrad": r"""
**Правило обновления:**
- $G_t = G_{t-1} + g_t^2$
- $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$
""",
    "Lion": r"""
**Правило обновления:**
- $c_t = \beta m_{t-1} + (1-\beta) g_t$, $\quad m_t = \beta m_{t-1} + (1-\beta) g_t$
- $\theta_{t+1} = \theta_t - \eta (\text{sign}(c_t) + \lambda \theta_t)$
""",
    "LARS": r"""
**Правило обновления:** layer-wise adaptive scaling:
- $\lambda_t = \gamma \frac{\|\theta\|}{\|\nabla f(\theta)\| + \lambda \|\theta\|}$
- $\theta_{t+1} = \theta_t - \eta \lambda_t (\nabla f(\theta_t) + \lambda \theta_t)$
""",
    "Sophia": r"""
**Правило обновления:** адаптивный шаг с оценкой гессиана (упрощённо: второй момент и поправка по кривизне).
""",
    "Adan": r"""
**Правило обновления:** адаптивные моменты с предсказанием градиентов (Nesterov-подобный шаг).
""",
    "MARS": r"""
**Правило обновления:** моментум и адаптивное масштабирование шага.
""",
}

OPTIMIZER_FORMULAS_EN: Dict[str, str] = {
    "GD": r"""
**Update rule:**
- $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$
""",
    "SGD": r"""
**Update rule (with momentum):**
- $v_t = \beta v_{t-1} + \nabla f(\theta_t)$
- $\theta_{t+1} = \theta_t - \eta v_t$
""",
    "RMSProp": r"""
**Update rule:**
- $s_t = \beta s_{t-1} + (1-\beta) g_t^2$
- $\theta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{s_t} + \epsilon}$
""",
    "Adam": r"""
**Update rule:**
- $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- $\hat{m}_t = m_t/(1-\beta_1^t)$, $\quad \hat{v}_t = v_t/(1-\beta_2^t)$
- $\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
""",
    "AdamW": r"""
**Update rule:** as Adam, plus decoupled weight decay:
- $\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$
""",
    "AMSGrad": r"""
**Update rule:** as Adam, but $\hat{v}_t = \max(\hat{v}_t, \hat{v}_{t-1})$ (non-decreasing second moment).
""",
    "Adagrad": r"""
**Update rule:**
- $G_t = G_{t-1} + g_t^2$
- $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$
""",
    "Lion": r"""
**Update rule:**
- $c_t = \beta m_{t-1} + (1-\beta) g_t$, $\quad m_t = \beta m_{t-1} + (1-\beta) g_t$
- $\theta_{t+1} = \theta_t - \eta (\text{sign}(c_t) + \lambda \theta_t)$
""",
    "LARS": r"""
**Update rule:** layer-wise adaptive scaling:
- $\lambda_t = \gamma \frac{\|\theta\|}{\|\nabla f(\theta)\| + \lambda \|\theta\|}$
- $\theta_{t+1} = \theta_t - \eta \lambda_t (\nabla f(\theta_t) + \lambda \theta_t)$
""",
    "Sophia": r"""
**Update rule:** adaptive step with Hessian estimate (simplified: second moment and curvature correction).
""",
    "Adan": r"""
**Update rule:** adaptive moments with gradient prediction (Nesterov-like step).
""",
    "MARS": r"""
**Update rule:** momentum and adaptive step scaling.
""",
}

FUNCTION_FORMULAS: Dict[str, Dict[str, Any]] = {
    "Quadratic": {
        "formula": r"f(x, y) = x^2 + y^2",
        "minimum": "(0, 0)",
    },
    "Rastrigin": {
        "formula": r"f(x,y) = 20 + x^2 + y^2 - 10(\cos(2\pi x) + \cos(2\pi y))",
        "minimum": "(0, 0)",
    },
    "Rosenbrock": {
        "formula": r"f(x,y) = (1-x)^2 + 100(y - x^2)^2",
        "minimum": "(1, 1)",
    },
    "Himmelblau": {
        "formula": r"f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        "minimum": "Четыре минимума: (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)",
    },
    "Ackley": {
        "formula": r"f(x,y) = -20\exp(-0.2\sqrt{0.5(x^2+y^2)}) - \exp(0.5(\cos(2\pi x)+\cos(2\pi y))) + e + 20",
        "minimum": "(0, 0)",
    },
    "Griewank": {
        "formula": r"f(x,y) = \frac{x^2+y^2}{4000} - \cos x \cos(y/\sqrt{2}) + 1",
        "minimum": "(0, 0)",
    },
    "Schwefel": {
        "formula": r"f(x,y) = 418.9829 \cdot 2 - \sum_i x_i \sin(\sqrt{|x_i|})",
        "minimum": "≈ (420.9687, 420.9687)",
    },
    "Levy": {
        "formula": r"Сложная композиция синусов и квадратов; глобальный минимум в (1, 1).",
        "minimum": "(1, 1)",
    },
    "Beale": {
        "formula": r"f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2",
        "minimum": "(3, 0.5)",
    },
}

FUNCTION_FORMULAS_EN: Dict[str, Dict[str, Any]] = {
    "Himmelblau": {
        "minimum": "Four minima: (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)",
    },
    "Levy": {
        "formula": r"f(x,y)\text{ — composition of sines and squares; global minimum at }(1,1).",
        "minimum": "(1, 1)",
    },
}


def get_optimizer_formula(optimizer_name: str, lang: Optional[str] = None) -> str:
    """Return Markdown/LaTeX formula block for the given optimizer in the given or current language.

    Args:
        optimizer_name: Registered optimizer name.
        lang: 'ru' or 'en'. If None, uses current UI language.

    Returns:
        Formula string or empty string if unknown.
    """
    if lang is None:
        lang = get_lang()
    if lang == "en":
        return OPTIMIZER_FORMULAS_EN.get(optimizer_name, "")
    return OPTIMIZER_FORMULAS.get(optimizer_name, "")


def get_function_formula(func_name: str, lang: Optional[str] = None) -> Dict[str, Any]:
    """Return formula and global minimum for the given test function in the given or current language.

    Args:
        func_name: Test function name (e.g. from TEST_FUNCTION_NAMES).
        lang: 'ru' or 'en'. If None, uses current UI language.

    Returns:
        Dict with keys 'formula' and 'minimum'; empty strings if unknown.
    """
    base = FUNCTION_FORMULAS.get(func_name, {"formula": "", "minimum": ""})
    if lang is None:
        lang = get_lang()
    if lang != "en":
        return dict(base)
    en_overrides = FUNCTION_FORMULAS_EN.get(func_name, {})
    return {
        "formula": en_overrides.get("formula", base.get("formula", "")),
        "minimum": en_overrides.get("minimum", base.get("minimum", "")),
    }
