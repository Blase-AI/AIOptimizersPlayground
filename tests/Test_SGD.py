import numpy as np
from optimizers import StochasticGradientDescent

def test_sgd_no_momentum():
    params = [np.array([1.0, -2.0]), np.array([0.5])]
    grads = [np.array([0.1, -0.1]), np.array([0.2])]
    optimizer = StochasticGradientDescent(learning_rate=0.01, momentum=0.0)

    updated = optimizer.update(params, grads)

    expected = [np.array([0.999, -1.999]), np.array([0.498])]
    for u, e in zip(updated, expected):
        assert np.allclose(u, e, rtol=1e-6)

def test_sgd_with_momentum_first_step():
    params = [np.array([1.0, -2.0]), np.array([0.5])]
    grads = [np.array([0.1, -0.1]), np.array([0.2])]
    optimizer = StochasticGradientDescent(learning_rate=0.01, momentum=0.9)

    updated = optimizer.update(params, grads)

    expected = [np.array([0.999, -1.999]), np.array([0.498])]
    for u, e in zip(updated, expected):
        assert np.allclose(u, e, rtol=1e-6)

def test_sgd_clip_norm():
    params = [np.array([1.0, -1.0])]
    grads = [np.array([10.0, -10.0])]
    clip_value = 1.0
    optimizer = StochasticGradientDescent(learning_rate=1.0, clip_norm=clip_value)

    updated = optimizer.update(params, grads)

    clipped_grad = grads[0] * (clip_value / (np.linalg.norm(grads[0]) + 1e-6))
    expected = [params[0] - clipped_grad]
    assert np.allclose(updated[0], expected[0], rtol=1e-6)

def test_sgd_decay():
    params = [np.array([1.0])]
    grads = [np.array([1.0])]
    optimizer = StochasticGradientDescent(learning_rate=1.0, decay_rate=0.5, momentum=0.0)

    updated1 = optimizer.update(params, grads)
    expected1 = [np.array([0.0])]
    assert np.allclose(updated1[0], expected1[0], rtol=1e-6)

    updated2 = optimizer.update(updated1, grads)
    expected2 = [np.array([-0.5])]
    assert np.allclose(updated2[0], expected2[0], rtol=1e-6)

def test_sgd_hook_called():
    hook_called = {}

    def hook(p, g, u):
        hook_called["ok"] = True

    params = [np.array([1.0])]
    grads = [np.array([1.0])]
    optimizer = StochasticGradientDescent(learning_rate=0.1, on_step=hook)

    optimizer.update(params, grads)

    assert hook_called.get("ok", False)
