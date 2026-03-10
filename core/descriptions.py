"""Text descriptions for UI (test functions and optimizers). Bilingual RU/EN."""
from typing import Dict, Optional

from core.i18n import get_lang

FUNCTION_DESCRIPTIONS: Dict[str, str] = {
    "Quadratic": "Простая параболическая функция f(x, y) = x² + y² с глобальным минимумом в (0,0). Идеальна для проверки базовой сходимости оптимизаторов.",
    "Rastrigin": "Мультимодальная функция f(x, y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy)) с глобальным минимумом в (0,0). Множество локальных минимумов усложняет оптимизацию.",
    "Rosenbrock": "Нелинейная функция f(x, y) = (1 - x)² + 100(y - x²)² с узкой долиной и глобальным минимумом в (1,1). Тестирует способность следовать сложным траекториям.",
    "Himmelblau": "Мультимодальная функция f(x, y) = (x² + y - 11)² + (x + y² - 7)² с четырьмя минимумами. Проверяет устойчивость к неоднозначным ландшафтам.",
    "Ackley": "Мультимодальная функция с глобальным минимумом в (0,0). Множество локальных минимумов затрудняет сходимость.",
    "Griewank": "Мультимодальная функция с глобальным минимумом в (0,0). Широкая структура и локальные минимумы тестируют глобальный и локальный поиск.",
    "Schwefel": "Мультимодальная функция с глобальным минимумом в (420.9687, 420.9687). Глубокие локальные минимумы усложняют глобальный поиск.",
    "Levy": "Нелинейная функция с глобальным минимумом в (1,1). Плоские участки и локальные минимумы делают её сложной для оптимизации.",
    "Beale": "Нелинейная функция с глобальным минимумом в (3, 0.5). Узкие долины тестируют точность оптимизаторов.",
}

FUNCTION_DESCRIPTIONS_EN: Dict[str, str] = {
    "Quadratic": "Simple parabolic function f(x, y) = x² + y² with global minimum at (0,0). Good for testing basic convergence.",
    "Rastrigin": "Multimodal function f(x, y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy)) with global minimum at (0,0). Many local minima.",
    "Rosenbrock": "Nonlinear function f(x, y) = (1 - x)² + 100(y - x²)² with a narrow valley and global minimum at (1,1).",
    "Himmelblau": "Multimodal function with four minima. Tests robustness on ambiguous landscapes.",
    "Ackley": "Multimodal function with global minimum at (0,0). Many local minima hinder convergence.",
    "Griewank": "Multimodal function with global minimum at (0,0). Tests global and local search.",
    "Schwefel": "Multimodal function with global minimum at (420.9687, 420.9687). Deep local minima.",
    "Levy": "Nonlinear function with global minimum at (1,1). Flat regions and local minima.",
    "Beale": "Nonlinear function with global minimum at (3, 0.5). Narrow valleys test optimizer precision.",
}

OPTIMIZER_DESCRIPTIONS: Dict[str, str] = {
    "SGD": """
**Стохастический градиентный спуск (SGD)** обновляет параметры на основе градиентов. Моментум сглаживает изменения.
- **Плюсы**: Простота, масштабируемость, устойчивость с правильной настройкой.
- **Минусы**: Чувствителен к выбору скорости обучения.
- **Рекомендации**: `learning_rate=0.01–0.1`, `momentum=0.9`.
""",
    "GD": """
**Градиентный спуск (GD)** — классический метод.
- **Плюсы**: Надежен на выпуклых функциях.
- **Минусы**: Медленный на больших данных.
- **Рекомендации**: `learning_rate=0.001–0.01`.
""",
    "RMSProp": """
**RMSProp** адаптирует скорость обучения по экспоненциальной средней квадратов градиентов.
- **Плюсы**: Устойчив к изменяющимся градиентам.
- **Рекомендации**: `learning_rate=0.001`, `momentum=0.9`.
""",
    "AMSGrad": """
**AMSGrad** — модификация Adam с максимумом второй экспоненциальной средней.
- **Плюсы**: Более устойчив на сложных ландшафтах.
- **Рекомендации**: `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`.
""",
    "Adagrad": """
**Adagrad** адаптирует lr для каждого параметра по сумме квадратов градиентов.
- **Плюсы**: Хорош для разреженных данных.
- **Рекомендации**: `learning_rate=0.01`.
""",
    "Adam": """
**Adam** сочетает моментум и адаптивное обучение.
- **Плюсы**: Быстрая сходимость, устойчивость.
- **Рекомендации**: `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`.
""",
    "AdamW": """
**AdamW** — Adam с декуплированной регуляризацией весов.
- **Плюсы**: Лучше обобщает, чем Adam.
- **Рекомендации**: `learning_rate=0.001`, `weight_decay=0.01`.
""",
    "Sophia": """
**Sophia** — адаптивные обновления на основе гессиана для больших моделей.
- **Рекомендации**: `learning_rate=0.0001`, `beta1=0.9`, `beta2=0.999`.
""",
    "Lion": """
**Lion** использует sign градиента, меньше памяти.
- **Рекомендации**: `learning_rate=0.0001`, `beta=0.9–0.95`, `weight_decay=0.01`.
""",
    "Adan": """
**Adan** — адаптивные обновления с предсказанием градиентов.
- **Рекомендации**: `learning_rate=0.001`, `weight_decay=0.01`.
""",
    "MARS": """
**MARS** — кастомный оптимизатор с моментумом.
- **Рекомендации**: `learning_rate=0.001`, `momentum=0.9`.
""",
    "LARS": """
**LARS** — layer-wise адаптивная скорость обучения.
- **Рекомендации**: `learning_rate=0.0001–0.01`, `trust_coeff=0.0005–0.002`, `momentum=0.9`.
""",
}

OPTIMIZER_DESCRIPTIONS_EN: Dict[str, str] = {
    "SGD": """
**Stochastic Gradient Descent (SGD)** updates parameters from gradients. Momentum smooths updates.
- **Pros**: Simple, scalable, stable with proper tuning.
- **Cons**: Sensitive to learning rate.
- **Recommendations**: `learning_rate=0.01–0.1`, `momentum=0.9`.
""",
    "GD": """
**Gradient Descent (GD)** — classic method.
- **Pros**: Reliable on convex functions.
- **Cons**: Slow on large data.
- **Recommendations**: `learning_rate=0.001–0.01`.
""",
    "RMSProp": """
**RMSProp** adapts learning rate by exponential average of squared gradients.
- **Pros**: Robust to varying gradients.
- **Recommendations**: `learning_rate=0.001`, `momentum=0.9`.
""",
    "AMSGrad": """
**AMSGrad** — Adam variant with max of second moment.
- **Pros**: More stable on complex landscapes.
- **Recommendations**: `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`.
""",
    "Adagrad": """
**Adagrad** adapts per-parameter lr from sum of squared gradients.
- **Pros**: Good for sparse data.
- **Recommendations**: `learning_rate=0.01`.
""",
    "Adam": """
**Adam** combines momentum and adaptive learning.
- **Pros**: Fast convergence, robust.
- **Recommendations**: `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`.
""",
    "AdamW": """
**AdamW** — Adam with decoupled weight decay.
- **Pros**: Better generalization than Adam.
- **Recommendations**: `learning_rate=0.001`, `weight_decay=0.01`.
""",
    "Sophia": """
**Sophia** — adaptive updates based on Hessian for large models.
- **Recommendations**: `learning_rate=0.0001`, `beta1=0.9`, `beta2=0.999`.
""",
    "Lion": """
**Lion** uses sign of gradient; lower memory.
- **Recommendations**: `learning_rate=0.0001`, `beta=0.9–0.95`, `weight_decay=0.01`.
""",
    "Adan": """
**Adan** — adaptive updates with gradient prediction.
- **Recommendations**: `learning_rate=0.001`, `weight_decay=0.01`.
""",
    "MARS": """
**MARS** — custom optimizer with momentum.
- **Recommendations**: `learning_rate=0.001`, `momentum=0.9`.
""",
    "LARS": """
**LARS** — layer-wise adaptive learning rate.
- **Recommendations**: `learning_rate=0.0001–0.01`, `trust_coeff=0.0005–0.002`, `momentum=0.9`.
""",
}


def get_function_description(name: str, lang: Optional[str] = None) -> str:
    """Return function description in the given or current language."""
    if lang is None:
        lang = get_lang()
    if lang == "en":
        return FUNCTION_DESCRIPTIONS_EN.get(name, "")
    return FUNCTION_DESCRIPTIONS.get(name, "")


def get_optimizer_description(name: str, lang: Optional[str] = None) -> str:
    """Return optimizer description in the given or current language."""
    if lang is None:
        lang = get_lang()
    if lang == "en":
        return OPTIMIZER_DESCRIPTIONS_EN.get(name, "")
    return OPTIMIZER_DESCRIPTIONS.get(name, "")
