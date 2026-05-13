import os
import sys

import numpy
import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.extend([
    SRC,
    os.path.join(SRC, "LatentDynamics"),
    os.path.join(SRC, "Utilities"),
    os.path.join(SRC, "Interpolate"),
])

from SINDy import SINDy
from Interpolate import Interpolate


def test_missing_train_coefs_raises_keyerror():
    ld = SINDy(n_z=1, Uniform_t_Grid=True, config={})
    with pytest.raises(KeyError):
        ld.get_train_coefs(numpy.array([0.0]))


def test_sindy_fit_coefficients_stores_native_trainable_dict():
    ld = SINDy(n_z=1, Uniform_t_Grid=True, config={"lstsq_reg": 0.0})
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.exp(-t).reshape(-1, 1)
    params = numpy.array([[0.25]])

    out = ld.fit_coefficients(Latent_States=[[z]], t_Grid=[t], params=params)

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
    ld = SINDy(n_z=1, Uniform_t_Grid=True, config={})
    ld.set_train_coefs(numpy.array([1.0]), {"A": torch.ones(1, 1), "b": torch.zeros(1)})
    exported = ld.export()

    ld2 = SINDy(n_z=1, Uniform_t_Grid=True, config={})
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
    interp = Interpolate(train_coefs)

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


def test_base_flatten_coefficients_concatenates_native_dict_items():
    ld = SINDy(n_z=2, Uniform_t_Grid=True, config={})
    native_coefs = [
        {"A": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "b": torch.tensor([5.0, 6.0])},
        {"A": torch.tensor([[7.0, 8.0], [9.0, 10.0]]), "b": torch.tensor([11.0, 12.0])},
    ]

    flat = ld.flatten_coefficients(native_coefs)

    assert flat.shape == (2, 6)
    assert numpy.allclose(flat, numpy.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ]))


def test_interpolate_rejects_non_tensor_values():
    with pytest.raises(AssertionError):
        Interpolate({(0.0,): {"A": numpy.zeros((1, 1))}})

from DampedSpring import DampedSpring
from DampedSpring_weak import DampedSpring_weak
from FiniteDifference import Derivative1_Order4


def test_damped_spring_fit_coefficients_uses_K_C_b_names():
    ld = DampedSpring(n_z=1, Uniform_t_Grid=True, config={"lstsq_reg": 1.0})
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.sin(t).reshape(-1, 1)
    dz = torch.cos(t).reshape(-1, 1)
    params = numpy.array([[0.5]])

    out = ld.fit_coefficients(Latent_States=[[z, dz]], t_Grid=[t], params=params)

    assert out is None
    coefs = ld.get_train_coefs(params[0])
    assert set(coefs.keys()) == {"K", "C", "b"}
    assert coefs["K"].shape == (1, 1)
    assert coefs["C"].shape == (1, 1)
    assert coefs["b"].shape == (1,)
    assert all(tensor.requires_grad and tensor.is_leaf for tensor in coefs.values())


def test_damped_spring_calibrate_uses_native_K_C_b_rhs():
    ld = DampedSpring(n_z=1, Uniform_t_Grid=True, config={})
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.sin(t).reshape(-1, 1)
    dz = torch.cos(t).reshape(-1, 1)
    params = numpy.array([[0.25]])
    K = torch.tensor([[2.0]])
    C = torch.tensor([[-0.5]])
    b = torch.tensor([0.1])
    ld.set_train_coefs(params[0], {"K": K, "C": C, "b": b})

    loss_LD_list, loss_coef_list, loss_stab_list = ld.calibrate([[z, dz]], "MSE", [t], params)

    d2z = Derivative1_Order4(dz, float((t[1] - t[0]).item()))
    rhs = z @ K.T + dz @ C.T + b.reshape(1, -1)
    expected_loss = torch.mean((d2z - rhs) ** 2)
    assert torch.allclose(loss_LD_list[0], expected_loss)
    assert len(loss_coef_list) == 1
    assert len(loss_stab_list) == 1


def test_damped_spring_weak_simulate_uses_native_K_C_b_names():
    config = {
        "type": "spring_w",
        "spring_w": {
            "test_func": "bump",
            "test_func_width": 0.5,
            "overlap": 0.5,
            "LS_loss_type": "MSE",
        },
    }
    ld = DampedSpring_weak(n_z=1, Uniform_t_Grid=True, config=config)
    coefs = {"K": torch.zeros(1, 1), "C": torch.zeros(1, 1), "b": torch.ones(1)}
    D0 = torch.zeros(1, 1)
    V0 = torch.zeros(1, 1)
    t = torch.linspace(0.0, 0.2, 3)

    D, V = ld.simulate(coefs=coefs, IC=[[D0, V0]], t_Grid=[t])[0]

    assert D.shape == (3, 1, 1)
    assert V.shape == (3, 1, 1)
