import numpy as np
import pytest
from optimizers import GradientDescent


def test_step_basic_update():
    params = [np.array([1.0]), np.array([2.0])]
    grads = [np.array([0.5]), np.array([1.0])]
    optimizer = GradientDescent(learning_rate=0.1, decay_rate=1.0)
    
    updated = optimizer.step(params, grads)
    
    expected = [np.array([0.95]), np.array([1.9])]
    for u, e in zip(updated, expected):
        np.testing.assert_allclose(u, e, rtol=1e-6)


def test_step_with_decay_rate():
    params = [np.array([1.0])]
    grads = [np.array([1.0])]
    optimizer = GradientDescent(learning_rate=0.1, decay_rate=0.5)

    optimizer.iteration = 2  
    updated = optimizer.step(params, grads)
    
    expected = [np.array([1.0 - 0.025])]
    np.testing.assert_allclose(updated[0], expected[0], rtol=1e-6)


def test_on_step_called():
    called = {}

    def hook(params, grads, updated):
        called['called'] = True
        called['params'] = params
        called['grads'] = grads
        called['updated'] = updated

    params = [np.array([1.0])]
    grads = [np.array([0.1])]
    optimizer = GradientDescent(learning_rate=0.1, on_step=hook)
    optimizer.step(params, grads)

    assert called.get('called', False)
    np.testing.assert_allclose(called['updated'][0], np.array([0.99]), rtol=1e-6)


def test_assert_decay_rate_invalid():
    with pytest.raises(AssertionError, match="decay_rate must be in \\(0, 1\\]"):
        GradientDescent(decay_rate=0.0)

    with pytest.raises(AssertionError):
        GradientDescent(decay_rate=-1)

    GradientDescent(decay_rate=1.0)
    GradientDescent(decay_rate=0.5)


def test_step_basic():
    params = [np.array([1.0, 2.0])]
    grads = [np.array([0.1, 0.2])]
    optimizer = GradientDescent(learning_rate=0.5)

    updated = optimizer.step(params, grads)

    np.testing.assert_allclose(updated[0], np.array([0.95, 1.9]), rtol=1e-6)


def test_l2_regularization():
    params = [np.array([1.0])]
    grads = [np.array([0.0])]  
    optimizer = GradientDescent(
        learning_rate=0.1,
        reg_type='l2',
        weight_decay=0.5
    )

    updated = optimizer.update(params, grads)
    expected = 1.0 - 0.1 * (0.5 * 1.0)  

    np.testing.assert_allclose(updated[0], np.array([0.95]), rtol=1e-6)



def test_track_history_every_iteration():
    params = [np.array([1.0])]
    grads = [np.array([0.1])]
    optimizer = GradientDescent(
        learning_rate=0.1,
        track_history=True,
        track_interval=1
    )

    optimizer.update(params, grads)
    optimizer.update(params, grads)

    assert len(optimizer.history) == 2


def test_learning_rate_decay():
    params = [np.array([1.0])]
    grads = [np.array([1.0])]
    optimizer = GradientDescent(
        learning_rate=1.0,
        decay_rate=0.5  
    )

    updated1 = optimizer.update(params, grads)  
    updated2 = optimizer.update(updated1, grads)  

    assert np.allclose(updated1[0], np.array([0.0]), rtol=1e-6)
    assert np.allclose(updated2[0], np.array([-0.5]), rtol=1e-6)



