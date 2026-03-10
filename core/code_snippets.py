"""Python code snippets (step logic) per optimizer for the Description tab."""
from typing import Dict

CODE_SNIPPETS: Dict[str, str] = {
    "GD": """def step(self, params, grads):
    lr = self._effective_lr()
    return [p - lr * g for p, g in zip(params, grads)]""",
    "SGD": """def step(self, params, grads):
    v = self.momentum * v_prev + lr * g
    return [p - v for p, v in zip(params, velocities)]""",
    "RMSProp": """def step(self, params, grads):
    s = beta * s_prev + (1 - beta) * (g * g)
    return [p - lr * g / (sqrt(s) + eps) for ...]""",
    "Adam": """def step(self, params, grads):
    m = beta1 * m_prev + (1 - beta1) * g
    v = beta2 * v_prev + (1 - beta2) * (g * g)
    m_hat = m / (1 - beta1**t); v_hat = v / (1 - beta2**t)
    return [p - lr * m_hat / (sqrt(v_hat) + eps) for ...]""",
    "AdamW": """# Как Adam, плюс после обновления:
    new_param = new_param - lr * weight_decay * p""",
    "AMSGrad": """# Как Adam, но v_hat = max(v_hat, v_hat_prev) (незатухающий второй момент)""",
    "Adagrad": """def step(self, params, grads):
    G += g * g
    return [p - lr * g / (sqrt(G) + eps) for ...]""",
    "Lion": """def step(self, params, grads):
    c = beta * m_prev + (1 - beta) * g
    m = beta * m_prev + (1 - beta) * g
    update = sign(c) + weight_decay * p
    return [p - lr * update for ...]""",
    "LARS": """def step(self, params, grads):
    scale = trust_coeff * norm(p) / (norm(g) + wd * norm(p))
    return [p - lr * scale * (g + wd * p) for ...]""",
    "Sophia": """# Адаптивный шаг с оценкой кривизны (упрощённо: второй момент + гессиан)""",
    "Adan": """# Моменты с предсказанием градиентов (Nesterov-подобный шаг)""",
    "MARS": """# Моментум и адаптивное масштабирование шага""",
}


def get_code_snippet(optimizer_name: str) -> str:
    """Return code snippet for optimizer, or empty string."""
    return CODE_SNIPPETS.get(optimizer_name, "")
