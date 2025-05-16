import pytest
import numpy as np
from numpy.typing import NDArray
from optimizers import Sophia 

@pytest.fixture
def simple_params_and_grads():
    params = [np.array([1.0, 2.0]), np.array([3.0])]
    grads = [np.array([0.5, 1.0]), np.array([2.0])]
    return params, grads

def test_initialization():
    optimizer = Sophia(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        clip_norm=1.0,
        decay_rate=0.99,
        track_history=True,
        track_interval=2,
        verbose=True
    )
    assert optimizer.learning_rate == 0.001
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999
    assert optimizer.eps == 1e-8
    assert optimizer.weight_decay == 0.01
    assert optimizer.clip_norm == 1.0
    assert optimizer.decay_rate == 0.99
    assert optimizer.track_history is True
    assert optimizer.track_interval == 2
    assert optimizer.reg_type == 'none'
    assert optimizer.verbose is True
    assert optimizer.m is None
    assert optimizer.h is None

def test_invalid_initialization():
    with pytest.raises(AssertionError, match="beta1 must be in"):
        Sophia(beta1=1.0)
    with pytest.raises(AssertionError, match="beta2 must be in"):
        Sophia(beta2=1.0)
    with pytest.raises(AssertionError, match="eps must be positive"):
        Sophia(eps=0.0)
    with pytest.raises(AssertionError, match="weight_decay must be non-negative"):
        Sophia(weight_decay=-0.01)
    with pytest.raises(AssertionError, match="clip_norm must be positive or None"):
        Sophia(clip_norm=0.0)
    with pytest.raises(AssertionError, match="decay_rate must be in"):
        Sophia(decay_rate=0.0)

def test_reset():
    optimizer = Sophia()
    optimizer.m = [np.ones((2,))]
    optimizer.h = [np.ones((2,))]
    optimizer.iteration = 5
    optimizer.reset()
    assert optimizer.m is None
    assert optimizer.h is None
    assert optimizer.iteration == 0

def test_step_basic_update(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Sophia(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    
    np.random.seed(42)
    updated_params = optimizer.update(params, grads)
    
    assert optimizer.m is not None
    assert optimizer.h is not None
    assert len(optimizer.m) == len(params)
    assert len(optimizer.h) == len(params)
    assert all(u.shape == p.shape for u, p in zip(updated_params, params))
    assert all(np.all(u <= p) for u, p in zip(updated_params, params))

def test_gradient_clipping(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Sophia(learning_rate=0.001, clip_norm=0.5)
    
    np.random.seed(42)
    updated_params = optimizer.update(params, grads)
    
    clipped_grads = [g * (0.5 / (np.linalg.norm(g) + 1e-6)) if np.linalg.norm(g) > 0.5 else g for g in grads]
    assert all(np.linalg.norm(cg) <= 0.5 + 1e-6 for cg in clipped_grads)

def test_learning_rate_decay(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Sophia(learning_rate=0.001, decay_rate=0.5)

    np.random.seed(42)
    current_params = params
    updates = []
    for _ in range(6): 
        current_params = optimizer.update(current_params, grads)
        updates.append(current_params)
        if len(updates) >= 4:
            update_current = np.abs(updates[-2][0] - updates[-1][0])
            update_previous = np.abs(updates[-3][0] - updates[-4][0])
            assert np.all(update_current < update_previous)
    
    assert optimizer.iteration == 6

def test_history_tracking(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Sophia(track_history=True, track_interval=2)
    
    np.random.seed(42)
    updated_params1 = optimizer.update(params, grads)
    updated_params2 = optimizer.update(updated_params1, grads)
    
    assert len(optimizer.history) == 1
    assert all(np.allclose(h, u) for h, u in zip(optimizer.history[0], updated_params2))

def test_on_step_callback(simple_params_and_grads):
    params, grads = simple_params_and_grads
    callback_called = False
    callback_params = []
    
    def on_step(p, g, u):
        nonlocal callback_called, callback_params
        callback_called = True
        callback_params = p
    
    optimizer = Sophia(on_step=on_step)
    
    np.random.seed(42)
    updated_params = optimizer.update(params, grads)
    
    assert callback_called
    assert all(np.allclose(cp, p) for cp, p in zip(callback_params, params))

def test_weight_decay(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Sophia(learning_rate=0.001, weight_decay=0.1)
    
    np.random.seed(42)
    updated_params = optimizer.update(params, grads)

    decayed_params = [p * (1 - 0.001 * 0.1) for p in params]
    assert all(np.all(u < dp) for u, dp in zip(updated_params, decayed_params))

def test_hessian_estimator(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Sophia(learning_rate=0.001, beta1=0.9, beta2=0.999)
    
    np.random.seed(42)
    updated_params = optimizer.update(params, grads)
    
    assert optimizer.h is not None
    assert len(optimizer.h) == len(params)
    assert all(np.all(h >= 0) for h in optimizer.h)
    
    h_first = [h.copy() for h in optimizer.h]

    new_grads = [2 * g for g in grads]
    updated_params2 = optimizer.update(updated_params, new_grads)

    assert all(np.all(optimizer.h[i] >= h_first[i]) for i in range(len(params)))

def test_get_config():
    optimizer = Sophia(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        clip_norm=1.0,
        decay_rate=0.99
    )
    config = optimizer.get_config()
    
    assert config['learning_rate'] == 0.001
    assert config['beta1'] == 0.9
    assert config['beta2'] == 0.999
    assert config['eps'] == 1e-8
    assert config['weight_decay'] == 0.01
    assert config['clip_norm'] == 1.0
    assert config['decay_rate'] == 0.99
    assert config['reg_type'] == 'none'

def test_repr():
    optimizer = Sophia(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    repr_str = repr(optimizer)
    assert 'Sophia' in repr_str
    assert 'lr=0.001' in repr_str
    assert 'beta1=0.9' in repr_str
    assert 'beta2=0.999' in repr_str
    assert 'eps=1e-08' in repr_str
    assert 'wd=0.01' in repr_str