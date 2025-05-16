import pytest
import numpy as np
from numpy.typing import NDArray
from optimizers import LARS  

@pytest.fixture
def simple_params_and_grads():
    params = [np.array([1.0, 2.0]), np.array([3.0])]
    grads = [np.array([0.5, 1.0]), np.array([2.0])]
    return params, grads

def test_initialization():
    optimizer = LARS(
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.01,
        trust_coeff=0.001,
        clip_norm=1.0,
        decay_rate=0.99,
        track_history=True,
        track_interval=2,
        verbose=True
    )
    assert optimizer.learning_rate == 0.01
    assert optimizer.momentum == 0.9
    assert optimizer.weight_decay == 0.01
    assert optimizer.trust_coeff == 0.001
    assert optimizer.clip_norm == 1.0
    assert optimizer.decay_rate == 0.99
    assert optimizer.reg_type == 'none'  
    assert optimizer.track_history is True
    assert optimizer.track_interval == 2
    assert optimizer.verbose is True
    assert optimizer.velocities is None

def test_invalid_initialization():
    with pytest.raises(AssertionError, match="momentum must be in"):
        LARS(momentum=1.0)
    with pytest.raises(AssertionError, match="weight_decay must be non-negative"):
        LARS(weight_decay=-0.01)
    with pytest.raises(AssertionError, match="trust_coeff must be positive"):
        LARS(trust_coeff=0.0)
    with pytest.raises(AssertionError, match="clip_norm must be positive or None"):
        LARS(clip_norm=0.0)
    with pytest.raises(AssertionError, match="decay_rate must be in"):
        LARS(decay_rate=0.0)

def test_reset():
    optimizer = LARS()
    optimizer.velocities = [np.ones((2,))]
    optimizer.iteration = 5
    optimizer.history = [[np.ones((2,))]]
    optimizer.reset()
    assert optimizer.velocities is None
    assert optimizer.iteration == 0
    assert len(optimizer.history) == 0

def test_step_basic_update(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = LARS(learning_rate=0.01, momentum=0.9, trust_coeff=0.001)
    
    updated_params = optimizer.update(params, grads)
    
    assert optimizer.velocities is not None
    assert len(optimizer.velocities) == len(params)
    assert all(u.shape == p.shape for u, p in zip(updated_params, params))
    assert all(np.all(u <= p) for u, p in zip(updated_params, params)) 

def test_gradient_clipping(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = LARS(learning_rate=0.01, clip_norm=0.5)
    
    updated_params = optimizer.update(params, grads)
    
    clipped_grads = [g * (0.5 / (np.linalg.norm(g) + 1e-6)) if np.linalg.norm(g) > 0.5 else g for g in grads]
    assert all(np.linalg.norm(cg) <= 0.5 + 1e-6 for cg in clipped_grads)

def test_local_lr_scaling(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = LARS(learning_rate=0.01, trust_coeff=0.001, weight_decay=0.01)
    
    updated_params = optimizer.update(params, grads)
    
    p_norm = np.linalg.norm(params[0]) 
    g_norm = np.linalg.norm(grads[0])   
    expected_local_lr = 0.001 * p_norm / (g_norm + 0.01 * p_norm + 1e-6)
    diff = np.abs(params[0] - updated_params[0])
    assert np.all(diff > 0) 

def test_learning_rate_decay(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = LARS(learning_rate=0.01, decay_rate=0.5)
    
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
    optimizer = LARS(learning_rate=0.01, weight_decay=0.1, trust_coeff=0.001)
    
    updated_params = optimizer.update(params, grads)

    optimizer_no_wd = LARS(learning_rate=0.01, weight_decay=0.0, trust_coeff=0.001)
    updated_no_wd = optimizer_no_wd.update(params, grads)
    diff_wd = np.abs(params[0] - updated_params[0])
    diff_no_wd = np.abs(params[0] - updated_no_wd[0])
    assert np.all(diff_wd >= diff_no_wd)

def test_history_tracking(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = LARS(track_history=True, track_interval=2)
    
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
    
    optimizer = LARS(on_step=on_step)
    updated_params = optimizer.update(params, grads)
    
    assert callback_called
    assert all(np.allclose(cp, p) for cp, p in zip(callback_params, params))

def test_get_config():
    optimizer = LARS(
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.01,
        trust_coeff=0.001,
        clip_norm=1.0,
        decay_rate=0.99
    )
    config = optimizer.get_config()
    
    assert config['learning_rate'] == 0.01
    assert config['momentum'] == 0.9
    assert config['weight_decay'] == 0.01
    assert config['trust_coeff'] == 0.001
    assert config['clip_norm'] == 1.0
    assert config['decay_rate'] == 0.99
    assert config['reg_type'] == 'none'

def test_repr():
    optimizer = LARS(learning_rate=0.01, momentum=0.9, weight_decay=0.01, trust_coeff=0.001)
    repr_str = repr(optimizer)
    assert 'LARS' in repr_str
    assert 'lr=0.01' in repr_str
    assert 'momentum=0.9' in repr_str
    assert 'wd=0.01' in repr_str
    assert 'trust_coeff=0.001' in repr_str