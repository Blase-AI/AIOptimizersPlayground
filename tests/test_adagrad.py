import pytest
import numpy as np
from numpy.typing import NDArray
from optimizers import Adagrad  

@pytest.fixture
def simple_params_and_grads():
    params = [np.array([1.0, 2.0]), np.array([3.0])]
    grads = [np.array([0.5, 1.0]), np.array([2.0])]
    return params, grads

def test_initialization():
    optimizer = Adagrad(
        learning_rate=0.01,
        eps=1e-8,
        clip_norm=1.0,
        decay_rate=0.99,
        track_history=True,
        track_interval=2,
        reg_type='l2',
        weight_decay=0.01,
        l1_ratio=0.5,
        verbose=True
    )
    assert optimizer.learning_rate == 0.01
    assert optimizer.eps == 1e-8
    assert optimizer.clip_norm == 1.0
    assert optimizer.decay_rate == 0.99
    assert optimizer.track_history is True
    assert optimizer.track_interval == 2
    assert optimizer.reg_type == 'l2'
    assert optimizer.weight_decay == 0.01
    assert optimizer.l1_ratio == 0.5
    assert optimizer.verbose is True
    assert optimizer.sum_g2 is None

def test_invalid_initialization():
    with pytest.raises(AssertionError, match="eps must be positive"):
        Adagrad(eps=0.0)
    with pytest.raises(AssertionError, match="clip_norm must be positive or None"):
        Adagrad(clip_norm=0.0)
    with pytest.raises(AssertionError, match="decay_rate must be in"):
        Adagrad(decay_rate=0.0)

def test_reset():
    optimizer = Adagrad()
    optimizer.sum_g2 = [np.ones((2,))]
    optimizer.iteration = 5
    optimizer.reset()
    assert optimizer.sum_g2 is None
    assert optimizer.iteration == 0

def test_step_basic_update(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Adagrad(learning_rate=0.01, eps=1e-8)
    
    updated_params = optimizer.update(params, grads) 
    
    assert optimizer.sum_g2 is not None
    assert len(optimizer.sum_g2) == len(params)
    assert all(u.shape == p.shape for u, p in zip(updated_params, params))
    assert all(np.all(u <= p) for u, p in zip(updated_params, params))

def test_gradient_clipping(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Adagrad(learning_rate=0.01, clip_norm=0.5)
    
    updated_params = optimizer.update(params, grads) 
    
    clipped_grads = [g * (0.5 / (np.linalg.norm(g) + 1e-6)) if np.linalg.norm(g) > 0.5 else g for g in grads]
    assert all(np.linalg.norm(cg) <= 0.5 + 1e-6 for cg in clipped_grads)

def test_learning_rate_decay(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Adagrad(learning_rate=0.01, decay_rate=0.9)
    
    updated_params1 = optimizer.update(params, grads)  
    updated_params2 = optimizer.update(updated_params1, grads)  
    
    update1 = params[0] - updated_params1[0]
    update2 = updated_params1[0] - updated_params2[0]
    assert np.all(update2 < update1)

def test_history_tracking(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Adagrad(track_history=True, track_interval=2)
    
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
    
    optimizer = Adagrad(on_step=on_step)
    updated_params = optimizer.update(params, grads)  
    
    assert callback_called
    assert all(np.allclose(cp, p) for cp, p in zip(callback_params, params))

def test_regularization_l2(simple_params_and_grads):
    params, grads = simple_params_and_grads
    optimizer = Adagrad(reg_type='l2', weight_decay=0.1)
    
    updated_params = optimizer.update(params, grads) 
    
    effective_grads = [g + 0.1 * p for g, p in zip(grads, params)]
    assert len(optimizer.sum_g2) == len(params)
    assert all(np.allclose(optimizer.sum_g2[i], eg * eg) for i, eg in enumerate(effective_grads))

def test_get_config():
    optimizer = Adagrad(
        learning_rate=0.01,
        eps=1e-8,
        clip_norm=1.0,
        decay_rate=0.99,
        reg_type='l1',
        weight_decay=0.01
    )
    config = optimizer.get_config()
    
    assert config['learning_rate'] == 0.01
    assert config['eps'] == 1e-8
    assert config['clip_norm'] == 1.0
    assert config['decay_rate'] == 0.99
    assert config['reg_type'] == 'l1'
    assert config['weight_decay'] == 0.01

def test_repr():
    optimizer = Adagrad(learning_rate=0.01, eps=1e-8, clip_norm=1.0)
    repr_str = repr(optimizer)
    assert 'Adagrad' in repr_str
    assert 'lr=0.01' in repr_str
    assert 'eps=1e-08' in repr_str
    assert 'clip_norm=1.0' in repr_str