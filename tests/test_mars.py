import pytest
import numpy as np
from numpy.typing import NDArray
from optimizers.Make_vAriance_Reduction_Shine import MARS 

@pytest.fixture
def simple_params_and_grads():
    params = [np.array([1.0, 2.0]), np.array([3.0])]
    grads = [np.array([0.5, 1.0]), np.array([2.0])]
    return params, grads

def test_initialization():
    optimizer = MARS(
        learning_rate=0.01,
        momentum=0.9,
        avg_beta=0.1,
        reg_type='l2',
        weight_decay=0.01,
        l1_ratio=0.5,
        clip_norm=1.0,
        bias_correction=True,
        decay_rate=0.99,
        track_history=True,
        track_interval=2,
        verbose=True
    )
    assert optimizer.learning_rate == 0.01
    assert optimizer.momentum == 0.9
    assert optimizer.avg_beta == 0.1
    assert optimizer.reg_type == 'l2'
    assert optimizer.weight_decay == 0.01
    assert optimizer.l1_ratio == 0.5
    assert optimizer.clip_norm == 1.0
    assert optimizer.bias_correction is True
    assert optimizer.decay_rate == 0.99
    assert optimizer.track_history is True
    assert optimizer.track_interval == 2
    assert optimizer.verbose is True
    assert optimizer.velocities is None
    assert optimizer.avg_grads is None

def test_invalid_initialization():
    with pytest.raises(AssertionError, match="momentum must be in"):
        MARS(momentum=1.0)
    with pytest.raises(AssertionError, match="avg_beta must be in"):
        MARS(avg_beta=1.0)
    with pytest.raises(AssertionError, match="clip_norm must be positive or None"):
        MARS(clip_norm=0.0)
    with pytest.raises(AssertionError, match="decay_rate must be in"):
        MARS(decay_rate=0.0)
    with pytest.raises(AssertionError, match="reg_type must be one of"):
        MARS(reg_type='invalid')

def test_reset():
    optimizer = MARS()
    optimizer.velocities = [np.ones((2,))]
    optimizer.avg_grads = [np.ones((2,))]
    optimizer.iteration = 5
    optimizer.reset()
    assert optimizer.velocities is None
    assert optimizer.avg_grads is None
    assert optimizer.iteration == 0

def test_step_basic_update(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = MARS(learning_rate=0.01, momentum=0.9, avg_beta=0.1)
    
    updated_params = optimizer.update(params, grads)
    
    assert optimizer.velocities is not None
    assert optimizer.avg_grads is not None
    assert len(optimizer.velocities) == len(params)
    assert len(optimizer.avg_grads) == len(params)
    assert all(u.shape == p.shape for u, p in zip(updated_params, params))
    assert all(np.all(u <= p) for u, p in zip(updated_params, params))

def test_gradient_clipping(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = MARS(learning_rate=0.01, clip_norm=0.5)
    
    updated_params = optimizer.update(params, grads)
    
    clipped_grads = [g * (0.5 / (np.linalg.norm(g) + 1e-6)) if np.linalg.norm(g) > 0.5 else g for g in grads]
    assert all(np.linalg.norm(cg) <= 0.5 + 1e-6 for cg in clipped_grads)

def test_learning_rate_decay(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = MARS(learning_rate=0.01, decay_rate=0.5)
    
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

def test_lr_scheduler(simple_params_and_grads):
    params, grads = simple_params_and_grads
    def scheduler(step): return 0.01 / step
    
    optimizer = MARS(learning_rate=0.01, lr_scheduler=scheduler, momentum=0.9, avg_beta=0.1)
    
    current_params = params
    updates = []
    for i in range(1, 6):  
        current_params = optimizer.update(current_params, grads)
        updates.append(current_params)
        expected_lr = 0.01 / i
        if len(updates) >= 2:
            update = np.abs(updates[-1][0] - updates[-2][0])
            assert np.all(update > 0)  
    
    assert optimizer.iteration == 5

def test_bias_correction(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = MARS(learning_rate=0.01, momentum=0.9, bias_correction=True)
    
    updated_params = optimizer.update(params, grads)

    optimizer_no_bc = MARS(learning_rate=0.01, momentum=0.9, bias_correction=False)
    updated_params_no_bc = optimizer_no_bc.update(params, grads)
    
    update_with_bc = np.abs(params[0] - updated_params[0])
    update_no_bc = np.abs(params[0] - updated_params_no_bc[0])
    assert np.all(update_with_bc > update_no_bc)

def test_history_tracking(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = MARS(track_history=True, track_interval=2)
    
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
    
    optimizer = MARS(on_step=on_step)
    updated_params = optimizer.update(params, grads)
    
    assert callback_called
    assert all(np.allclose(cp, p) for cp, p in zip(callback_params, params))

def test_regularization(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer_l2 = MARS(learning_rate=0.01, reg_type='l2', weight_decay=0.1)
    
    updated_params = optimizer_l2.update(params, grads)

    effective_grads = [g + 0.1 * p for g, p in zip(grads, params)]
    assert all(np.all(u < p) for u, p in zip(updated_params, params))

def test_get_config():
    def scheduler(step): return 0.01 / step
    optimizer = MARS(
        learning_rate=0.01,
        momentum=0.9,
        avg_beta=0.1,
        reg_type='l2',
        weight_decay=0.01,
        l1_ratio=0.5,
        clip_norm=1.0,
        bias_correction=True,
        lr_scheduler=scheduler,
        decay_rate=0.99
    )
    config = optimizer.get_config()
    
    assert config['learning_rate'] == 0.01
    assert config['momentum'] == 0.9
    assert config['avg_beta'] == 0.1
    assert config['reg_type'] == 'l2'
    assert config['weight_decay'] == 0.01
    assert config['l1_ratio'] == 0.5
    assert config['clip_norm'] == 1.0
    assert config['bias_correction'] is True
    assert config['lr_scheduler'] == 'scheduler'
    assert config['decay_rate'] == 0.99

def test_repr():
    optimizer = MARS(learning_rate=0.01, momentum=0.9, avg_beta=0.1, bias_correction=True)
    repr_str = repr(optimizer)
    assert 'MARS' in repr_str
    assert 'lr=0.01' in repr_str
    assert 'momentum=0.9' in repr_str
    assert 'avg_beta=0.1' in repr_str
    assert 'bias_correction=True' in repr_str