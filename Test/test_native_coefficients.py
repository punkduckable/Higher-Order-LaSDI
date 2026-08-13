import os
import sys

import numpy
import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from LatentDynamics import SINDy
from Interpolate import GPInterpolate
import Interpolate.GaussianProcess as GPModule
from Plotting.Metrics import flatten_coefficients


def _sindy_config(lstsq_reg=1.0, trainable=True):
    return {
        "type": "sindy",
        "trainable": trainable,
        "sindy": {"lstsq_reg": lstsq_reg},
    }


def _spring_config(lstsq_reg=1.0, trainable=True):
    return {
        "type": "spring",
        "trainable": trainable,
        "spring": {"lstsq_reg": lstsq_reg},
    }


def _sindy_w_config(test_func_type="PC-poly", trainable=True):
    return {
        "type": "sindy_w",
        "trainable": trainable,
        "sindy_w": {
            "test_func_type": test_func_type,
            "test_func_width": 0.5,
            "overlap": 0.5,
        },
    }


def _spring_w_config(test_func_type="PC-poly", trainable=True):
    return {
        "type": "spring_w",
        "trainable": trainable,
        "spring_w": {
            "test_func_type": test_func_type,
            "test_func_width": 0.5,
            "overlap": 0.5,
        },
    }


def _switch_w_config(test_func_type="PC-poly", trainable=True):
    return {
        "type": "switch_w",
        "trainable": trainable,
        "switch_w": {
            "test_func_type": test_func_type,
            "test_func_width": 0.5,
            "overlap": 0.5,
        },
    }


def test_missing_train_coefs_raises_keyerror():
    ld = SINDy(n_z=1, Uniform_t_Grid=True, config=_sindy_config())
    with pytest.raises(KeyError):
        ld.get_train_coefs(numpy.array([0.0]))


def test_sindy_initialize_coefficients_stores_native_trainable_dict():
    ld = SINDy(n_z=1, Uniform_t_Grid=True, config=_sindy_config(lstsq_reg=0.0))
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.exp(-t).reshape(-1, 1)
    params = numpy.array([[0.25]])

    out = ld.initialize_coefficients(Latent_States=[[z]], t_Grid=[t], device=torch.device("cpu"), params=params)

    assert out is None
    coefs = ld.get_train_coefs(params[0])
    assert set(coefs.keys()) == {"A", "b"}
    assert coefs["A"].shape == (1, 1)
    assert coefs["b"].shape == (1,)
    assert coefs["A"].requires_grad
    assert coefs["b"].requires_grad
    assert coefs["A"].is_leaf
    assert coefs["b"].is_leaf
    assert ld.trainable_coef_tensors() == [coefs["A"], coefs["b"]]


def test_latent_dynamics_export_load_restores_trainable_coefs():
    ld = SINDy(n_z=1, Uniform_t_Grid=True, config=_sindy_config())
    ld.set_train_coefs(numpy.array([1.0]), {"A": torch.ones(1, 1), "b": torch.zeros(1)}, torch.device("cpu"))
    exported = ld.export()

    ld2 = SINDy(n_z=1, Uniform_t_Grid=True, config=_sindy_config())
    ld2.load(exported)
    coefs = ld2.get_train_coefs(numpy.array([1.0]))

    assert torch.allclose(coefs["A"], torch.ones(1, 1))
    assert torch.allclose(coefs["b"], torch.zeros(1))
    assert coefs["A"].requires_grad and coefs["A"].is_leaf
    assert coefs["b"].requires_grad and coefs["b"].is_leaf


def test_interpolate_sample_mean_and_std_preserve_keys_and_shapes():
    train_coefs = {
        (0.0,): {"A": torch.zeros(1, 1), "b": torch.zeros(1)},
        (1.0,): {"A": torch.ones(1, 1), "b": torch.ones(1)},
    }
    interp = GPInterpolate()
    interp.update_train_coefs(train_coefs)

    mean = interp.mean(numpy.array([0.5]))
    std = interp.std(numpy.array([0.5]))
    sample = interp.sample(numpy.array([0.5]))

    assert set(mean.keys()) == {"A", "b"}
    assert set(std.keys()) == {"A", "b"}
    assert set(sample.keys()) == {"A", "b"}
    assert mean["A"].shape == (1, 1)
    assert mean["b"].shape == (1,)
    assert std["A"].shape == (1, 1)
    assert std["b"].shape == (1,)
    assert sample["A"].shape == (1, 1)
    assert sample["b"].shape == (1,)


def test_gp_interpolate_update_train_coefs_skips_unchanged_refit(monkeypatch):
    calls = {"n": 0}

    def fake_fit_gps(X, Y):
        calls["n"] += 1
        return [object() for _ in range(Y.shape[1])]

    monkeypatch.setattr(GPModule, "fit_gps", fake_fit_gps)
    train_coefs = {
        (0.0,): {"A": torch.zeros(1, 1), "b": torch.zeros(1)},
        (1.0,): {"A": torch.ones(1, 1), "b": torch.ones(1)},
    }
    interp = GPInterpolate()
    assert calls["n"] == 0

    interp.update_train_coefs(train_coefs)
    assert calls["n"] == 2

    interp.update_train_coefs(train_coefs)
    assert calls["n"] == 2

    train_coefs[(1.0,)]["b"] = 2.0 * torch.ones(1)
    interp.update_train_coefs(train_coefs)
    assert calls["n"] == 4


def test_base_flatten_coefficients_concatenates_native_dict_items():
    native_coefs = [
        {"A": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "b": torch.tensor([5.0, 6.0])},
        {"A": torch.tensor([[7.0, 8.0], [9.0, 10.0]]), "b": torch.tensor([11.0, 12.0])},
    ]

    flat = flatten_coefficients(native_coefs)

    assert flat.shape == (2, 6)
    assert numpy.allclose(flat, numpy.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ]))


def test_interpolate_rejects_non_tensor_values():
    with pytest.raises(AssertionError):
        interp = GPInterpolate()
        interp.update_train_coefs({(0.0,): {"A": numpy.zeros((1, 1))}})

from LatentDynamics import DampedSpring, DampedSpring_weak, SINDy_weak, SwitchSINDy_weak
from Utilities.FiniteDifference import Derivative1_Order4


def test_damped_spring_initialize_coefficients_uses_K_C_b_names():
    ld = DampedSpring(n_z=1, Uniform_t_Grid=True, config=_spring_config())
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.sin(t).reshape(-1, 1)
    dz = torch.cos(t).reshape(-1, 1)
    params = numpy.array([[0.5]])

    out = ld.initialize_coefficients(Latent_States=[[z, dz]], t_Grid=[t], device=torch.device("cpu"), params=params)

    assert out is None
    coefs = ld.get_train_coefs(params[0])
    assert set(coefs.keys()) == {"K", "C", "b"}
    assert coefs["K"].shape == (1, 1)
    assert coefs["C"].shape == (1, 1)
    assert coefs["b"].shape == (1,)
    assert all(tensor.requires_grad and tensor.is_leaf for tensor in coefs.values())


def test_damped_spring_compute_losses_uses_native_K_C_b_rhs():
    ld = DampedSpring(n_z=1, Uniform_t_Grid=True, config=_spring_config())
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.sin(t).reshape(-1, 1)
    dz = torch.cos(t).reshape(-1, 1)
    params = numpy.array([[0.25]])
    K = torch.tensor([[2.0]])
    C = torch.tensor([[-0.5]])
    b = torch.tensor([0.1])
    ld.set_train_coefs(params[0], {"K": K, "C": C, "b": b}, torch.device("cpu"))

    loss_LD_list, loss_coef_list, loss_stab_list = ld.compute_losses([[z, dz]], "MSE", [t], params)

    d2z = Derivative1_Order4(dz, float((t[1] - t[0]).item()))
    rhs = z @ K.T + dz @ C.T + b.reshape(1, -1)
    expected_loss = torch.mean((d2z - rhs) ** 2)
    assert torch.allclose(loss_LD_list[0], expected_loss)
    assert len(loss_coef_list) == 1
    assert len(loss_stab_list) == 1


def test_damped_spring_weak_simulate_uses_native_K_C_b_names():
    ld = DampedSpring_weak(n_z=1, Uniform_t_Grid=True, config=_spring_w_config(test_func_type="bump"))
    coefs = {"K": torch.zeros(1, 1), "C": torch.zeros(1, 1), "b": torch.ones(1)}
    D0 = torch.zeros(1, 1)
    V0 = torch.zeros(1, 1)
    t = torch.linspace(0.0, 0.2, 3)
    params = numpy.array([[0.25]])
    ld.set_train_coefs(params[0], coefs, torch.device("cpu"))

    D, V = ld.simulate(IC=[[D0, V0]], t_Grid=[t], params=params)[0]

    assert D.shape == (3, 1, 1)
    assert V.shape == (3, 1, 1)


def test_sindy_simulate_handles_multiple_parameters_without_recursion():
    ld = SINDy(n_z=1, Uniform_t_Grid=True, config=_sindy_config())
    coefs = [
        {"A": torch.zeros(1, 1), "b": torch.ones(1)},
        {"A": torch.zeros(1, 1), "b": 2.0 * torch.ones(1)},
    ]
    IC = [[torch.zeros(1, 1)], [torch.zeros(1, 1)]]
    t_Grid = [torch.linspace(0.0, 0.2, 3), torch.linspace(0.0, 0.2, 3)]
    params = numpy.array([[0.25], [0.75]])
    for i in range(params.shape[0]):
        ld.set_train_coefs(params[i, :], coefs[i], torch.device("cpu"))

    Z = ld.simulate(IC=IC, t_Grid=t_Grid, params=params)

    assert len(Z) == 2
    assert Z[0][0].shape == (3, 1, 1)
    assert Z[1][0].shape == (3, 1, 1)


def test_interpolatable_simulate_uses_train_coefs_before_interpolator():
    class DummyInterpolator:
        def __init__(self):
            self.mean_calls = 0
            self.sample_calls = 0

        def update_train_coefs(self, train_coefs):
            return None

        def mean(self, param):
            self.mean_calls += 1
            return {"A": torch.zeros(1, 1), "b": 7.0 * torch.ones(1)}

        def sample(self, param):
            self.sample_calls += 1
            return {"A": torch.zeros(1, 1), "b": 11.0 * torch.ones(1)}

    ld = SINDy(n_z=1, Uniform_t_Grid=True, config=_sindy_config())
    dummy = DummyInterpolator()
    ld.interpolator = dummy
    train_params = numpy.array([[0.25]])
    ld.set_train_coefs(train_params[0], {"A": torch.zeros(1, 1), "b": 3.0 * torch.ones(1)}, torch.device("cpu"))

    Z_train = ld.simulate(IC=[[torch.zeros(1, 1)]], t_Grid=[torch.tensor([0.0, 0.1])], params=train_params, sample=True)[0][0]
    assert torch.allclose(Z_train[-1, 0, 0], torch.tensor(0.3), atol=1.0e-6)
    assert dummy.sample_calls == 0

    test_params = numpy.array([[0.75]])
    Z_test = ld.simulate(IC=[[torch.zeros(1, 1)]], t_Grid=[torch.tensor([0.0, 0.1])], params=test_params, sample=False)[0][0]
    assert torch.allclose(Z_test[-1, 0, 0], torch.tensor(0.7), atol=1.0e-6)
    assert dummy.mean_calls == 1


from LatentDynamics import LatentDynamics, WeakLatentDynamics


def _weak_base_config(test_func_type="PC-poly"):
    return {
        "type": "dummy",
        "trainable": True,
        "dummy": {
            "test_func_type": test_func_type,
            "test_func_width": 0.5,
            "overlap": 0.5,
        },
    }


def test_weak_latent_dynamics_requires_weak_config_keys():
    bad_config = {"type": "dummy", "trainable": True, "dummy": {"test_func_width": 0.5, "overlap": 0.5}}
    with pytest.raises(AssertionError):
        WeakLatentDynamics(n_z=1, n_coefs=1, n_IC=2, Uniform_t_Grid=True, trainable=True, config=bad_config)


def test_add_and_get_weight_functions_store_arbitrary_derivatives():
    ld = WeakLatentDynamics(n_z=1, n_coefs=1, n_IC=2, Uniform_t_Grid=True, trainable=True, config=_weak_base_config())
    params = numpy.array([0.25])
    t = torch.linspace(0.0, 1.0, 11)

    ld.add_weight_functions(params, t)
    weights = ld.get_test_functions(params)

    assert len(weights) == 3
    assert len(ld.weight_function_derivatives) == 3
    assert all(ld._param_key(params) in d for d in ld.weight_function_derivatives)
    assert weights[0].shape == weights[1].shape == weights[2].shape
    assert weights[0].shape[1] == t.shape[0]


def test_get_test_functions_missing_param_raises_keyerror():
    ld = WeakLatentDynamics(n_z=1, n_coefs=1, n_IC=2, Uniform_t_Grid=True, trainable=True, config=_weak_base_config())
    with pytest.raises(KeyError):
        ld.get_test_functions(numpy.array([0.25]))


def test_damped_spring_weak_fit_zero_initializes_and_compute_losses_requires_weights():
    ld = DampedSpring_weak(n_z=1, Uniform_t_Grid=True, config=_spring_w_config())
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.sin(t).reshape(-1, 1)
    dz = torch.cos(t).reshape(-1, 1)
    params = numpy.array([[0.25]])

    out = ld.initialize_coefficients([[z, dz]], [t], torch.device("cpu"), params)
    coefs = ld.get_train_coefs(params[0])

    assert out is None
    assert torch.allclose(coefs["K"], torch.zeros(1, 1))
    assert torch.allclose(coefs["C"], torch.zeros(1, 1))
    assert torch.allclose(coefs["b"], torch.zeros(1))
    assert all(tensor.requires_grad and tensor.is_leaf for tensor in coefs.values())

    with pytest.raises(KeyError):
        ld.compute_losses([[z, dz]], "MSE", [t], params)


def test_sindy_weak_fit_zero_initializes_and_compute_losses_requires_weights():
    ld = SINDy_weak(n_z=1, Uniform_t_Grid=True, config=_sindy_w_config())
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.sin(t).reshape(-1, 1)
    params = numpy.array([[0.25]])

    out = ld.initialize_coefficients([[z]], [t], torch.device("cpu"), params)
    coefs = ld.get_train_coefs(params[0])

    assert out is None
    assert set(coefs.keys()) == {"A", "b"}
    assert torch.allclose(coefs["A"], torch.zeros(1, 1))
    assert torch.allclose(coefs["b"], torch.zeros(1))
    assert all(tensor.requires_grad and tensor.is_leaf for tensor in coefs.values())
    assert ld.trainable_coef_tensors() == [coefs["A"], coefs["b"]]

    with pytest.raises(KeyError):
        ld.compute_losses([[z]], "MSE", [t], params)


def test_sindy_weak_compute_losses_with_weight_functions_returns_losses():
    ld = SINDy_weak(n_z=1, Uniform_t_Grid=True, config=_sindy_w_config())
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.sin(t).reshape(-1, 1)
    params = numpy.array([[0.25]])

    ld.add_weight_functions(params[0], t)
    ld.initialize_coefficients([[z]], [t], torch.device("cpu"), params)
    loss_LD_list, loss_coef_list, loss_stab_list = ld.compute_losses([[z]], "MSE", [t], params)

    assert len(loss_LD_list) == 1
    assert len(loss_coef_list) == 1
    assert len(loss_stab_list) == 1
    assert all(loss.ndim == 0 for loss in loss_LD_list + loss_coef_list + loss_stab_list)


def test_switch_sindy_weak_fit_zero_initializes_native_names():
    ld = SwitchSINDy_weak(n_z=1, Uniform_t_Grid=True, switch_time=lambda p: 0.5, config=_switch_w_config())
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.sin(t).reshape(-1, 1)
    params = numpy.array([[0.25]])

    out = ld.initialize_coefficients([[z]], [t], torch.device("cpu"), params)
    coefs = ld.get_train_coefs(params[0])

    assert out is None
    assert set(coefs.keys()) == {"A_before", "b_before", "A_after", "b_after"}
    assert torch.allclose(coefs["A_before"], torch.zeros(1, 1))
    assert torch.allclose(coefs["b_before"], torch.zeros(1))
    assert torch.allclose(coefs["A_after"], torch.zeros(1, 1))
    assert torch.allclose(coefs["b_after"], torch.zeros(1))
    assert all(tensor.requires_grad and tensor.is_leaf for tensor in coefs.values())


def test_switch_sindy_weak_simulate_returns_first_order_trajectory_shape():
    ld = SwitchSINDy_weak(n_z=1, Uniform_t_Grid=True, switch_time=lambda p: 0.5, config=_switch_w_config())
    coefs = {
        "A_before": torch.zeros(1, 1),
        "b_before": torch.ones(1),
        "A_after": torch.zeros(1, 1),
        "b_after": torch.zeros(1),
    }
    Z0 = torch.zeros(1, 1)
    t = torch.linspace(0.0, 1.0, 5)
    params = numpy.array([[0.25]])
    ld.set_train_coefs(params[0], coefs, torch.device("cpu"))

    Z = ld.simulate(IC=[[Z0]], t_Grid=[t], params=params)[0][0]

    assert Z.shape == (5, 1, 1)


def test_get_uniform_grid_no_p_argument():
    ld = WeakLatentDynamics(n_z=1, n_coefs=1, n_IC=1, Uniform_t_Grid=True, trainable=True, config=_weak_base_config())
    a_s, b_s = ld._get_support_intervals(T=1.0, L=0.5, s=0.25)

    assert numpy.allclose(a_s, numpy.array([0.0, 0.25, 0.5]))
    assert numpy.allclose(b_s, numpy.array([0.5, 0.75, 1.0]))

from Trainer import Trainer


def test_base_trainer_noise_uses_clean_backup_and_preserves_initial_frame():
    trainer = Trainer.__new__(Trainer)
    trainer.noise_ratio = 0.2
    clean = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    trainer.U_Train = [[clean.clone()]]
    trainer.U_Train_Clean = []

    torch.manual_seed(0)
    trainer.apply_noise_to_U_Train()

    assert len(trainer.U_Train_Clean) == 1
    assert torch.allclose(trainer.U_Train_Clean[0][0], clean)
    assert trainer.U_Train[0][0].shape == clean.shape
    assert trainer.U_Train[0][0].dtype == clean.dtype
    assert trainer.U_Train[0][0].device == clean.device
    assert torch.allclose(trainer.U_Train[0][0][0], clean[0])
    assert not torch.allclose(trainer.U_Train[0][0][1:], clean[1:])

    # Re-noising should use U_Train_Clean, not add noise on top of the current U_Train contents.
    trainer.U_Train[0][0].fill_(999.0)
    torch.manual_seed(1)
    trainer.apply_noise_to_U_Train()
    assert torch.allclose(trainer.U_Train_Clean[0][0], clean)
    assert torch.allclose(trainer.U_Train[0][0][0], clean[0])
    assert not torch.allclose(trainer.U_Train[0][0], torch.full_like(clean, 999.0))
