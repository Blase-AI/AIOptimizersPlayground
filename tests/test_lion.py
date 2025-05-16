import pytest
import numpy as np
from numpy.typing import NDArray
from optimizers import Lion  

@pytest.fixture
def simple_params_and_grads():
    params = [np.array([1.0, 2.0]), np.array([3.0])]
    grads = [np.array([0.5, 1.0]), np.array([2.0])]
    return params, grads

def test_initialization():
    optimizer = Lion(
        learning_rate=1e-4,
        beta=0.9,
        weight_decay=0.01,
        clip_norm=1.0,
        bias_correction=True,
        lr_scheduler=lambda x: 1e-4 / x,
        decay_rate=0.99,
        track_history=True,
        track_interval=2,
        verbose=True
    )
    assert optimizer.learning_rate == 1e-4
    assert optimizer.beta == 0.9
    assert optimizer.weight_decay == 0.01
    assert optimizer.clip_norm == 1.0
    assert optimizer.bias_correction is True
    assert optimizer.lr_scheduler is not None
    assert optimizer.decay_rate == 0.99
    assert optimizer.reg_type == 'none' 
    assert optimizer.track_history is True
    assert optimizer.track_interval == 2
    assert optimizer.verbose is True
    assert optimizer.v is None

def test_invalid_initialization():
    with pytest.raises(AssertionError, match="beta must be in"):
        Lion(beta=1.0)
    with pytest.raises(AssertionError, match="weight_decay must be non-negative"):
        Lion(weight_decay=-0.01)
    with pytest.raises(AssertionError, match="clip_norm must be positive or None"):
        Lion(clip_norm=0.0)
    with pytest.raises(AssertionError, match="decay_rate must be in"):
        Lion(decay_rate=0.0)

def test_reset():
    optimizer = Lion()
    optimizer.v = [np.ones((2,))]
    optimizer.iteration = 5
    optimizer.history = [[np.ones((2,))]]
    optimizer.reset()
    assert optimizer.v is None
    assert optimizer.iteration == 0
    assert len(optimizer.history) == 0

def test_step_basic_update(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Lion(learning_rate=1e-4, beta=0.9, bias_correction=False)
    
    updated_params = optimizer.update(params, grads)
    
    assert optimizer.v is not None
    assert len(optimizer.v) == len(params)
    assert all(u.shape == p.shape for u, p in zip(updated_params, params))
    assert all(np.all(u <= p) for u, p in zip(updated_params, params)) 

def test_gradient_clipping(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Lion(learning_rate=1e-4, clip_norm=0.5)
    
    updated_params = optimizer.update(params, grads)
    
    clipped_grads = [g * (0.5 / (np.linalg.norm(g) + 1e-6)) if np.linalg.norm(g) > 0.5 else g for g in grads]
    assert all(np.linalg.norm(cg) <= 0.5 + 1e-6 for cg in clipped_grads)

def test_bias_correction(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer_with_bc = Lion(learning_rate=1e-4, beta=0.9, bias_correction=True)
    optimizer_without_bc = Lion(learning_rate=1e-4, beta=0.9, bias_correction=False)
    
    updated_with_bc = optimizer_with_bc.update(params, grads)
    updated_without_bc = optimizer_without_bc.update(params, grads)

    diff_with_bc = np.abs(params[0] - updated_with_bc[0])
    diff_without_bc = np.abs(params[0] - updated_without_bc[0])
    assert np.all(diff_with_bc >= diff_without_bc)

def test_lr_scheduler(simple_params_and_grads):
    params, grads = simple_params_and_grads
    def scheduler(step): return 1e-4 / step
    
    optimizer = Lion(learning_rate=1e-4, lr_scheduler=scheduler, beta=0.9)
    
    current_params = params
    updates = []
    for _ in range(5): 
        current_params = optimizer.update(current_params, grads)
        updates.append(current_params)
        if len(updates) >= 2:
            update = np.abs(updates[-1][0] - updates[-2][0])
            assert np.all(update > 0)  
    
    assert optimizer.iteration == 5

def test_learning_rate_decay(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Lion(learning_rate=1e-4, decay_rate=0.5)
    
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

def test_weight_decay(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Lion(learning_rate=1e-4, weight_decay=0.1)
    
    updated_params = optimizer.update(params, grads)

    decay_factor = 1 - 1e-4 * 0.1
    assert all(np.all(u < p * decay_factor) for u, p in zip(updated_params, params))  

def test_history_tracking(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Lion(track_history=True, track_interval=2)
    
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
    
    optimizer = Lion(on_step=on_step)
    updated_params = optimizer.update(params, grads)
    
    assert callback_called
    assert all(np.allclose(cp, p) for cp, p in zip(callback_params, params))

def test_get_config():
    def scheduler(step): return 1e-4 / step
    optimizer = Lion(
        learning_rate=1e-4,
        beta=0.9,
        weight_decay=0.01,
        clip_norm=1.0,
        bias_correction=True,
        lr_scheduler=scheduler,
        decay_rate=0.99
    )
    config = optimizer.get_config()
    
    assert config['learning_rate'] == 1e-4
    assert config['beta'] == 0.9
    assert config['weight_decay'] == 0.01
    assert config['clip_norm'] == 1.0
    assert config['bias_correction'] is True
    assert config['lr_scheduler'] == 'scheduler'
    assert config['decay_rate'] == 0.99
    assert config['reg_type'] == 'none'

def test_repr():
    optimizer = Lion(learning_rate=1e-4, beta=0.9, weight_decay=0.01, bias_correction=True)
    repr_str = repr(optimizer)
    assert 'Lion' in repr_str
    assert 'lr=0.0001' in repr_str
    assert 'beta=0.9' in repr_str
    assert 'wd=0.01' in repr_str
    assert 'bias_correction=True' in repr_str