import os
import sys

import numpy
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from LatentDynamics import CABLE, DampedSpring, DampedSpring_weak, SINDy, SINDy_weak, SwitchSINDy, SwitchSINDy_weak
from Schemas import (
    CABLELatentDynamicsConfig,
    DampedSpringLatentDynamicsConfig,
    DampedSpringWeakLatentDynamicsConfig,
    SINDyLatentDynamicsConfig,
    SINDyWeakLatentDynamicsConfig,
    SwitchSINDyLatentDynamicsConfig,
    SwitchSINDyWeakLatentDynamicsConfig,
)


def _sindy_config(trainable=True):
    return SINDyLatentDynamicsConfig.model_validate({
        "type": "sindy",
        "interpolator_type": "GP",
        "trainable": trainable,
        "sindy": {"lstsq_reg": 0.0},
    })


def _sindy_w_config(trainable=True):
    return SINDyWeakLatentDynamicsConfig.model_validate({
        "type": "sindy_w",
        "interpolator_type": "GP",
        "trainable": trainable,
        "sindy_w": {"test_func_type": "PC-poly", "test_func_width": 0.5, "overlap": 0.5},
    })


def _spring_config(trainable=True):
    return DampedSpringLatentDynamicsConfig.model_validate({
        "type": "spring",
        "interpolator_type": "GP",
        "trainable": trainable,
        "spring": {"lstsq_reg": 0.0},
    })


def _spring_w_config(trainable=True):
    return DampedSpringWeakLatentDynamicsConfig.model_validate({
        "type": "spring_w",
        "interpolator_type": "GP",
        "trainable": trainable,
        "spring_w": {"test_func_type": "PC-poly", "test_func_width": 0.5, "overlap": 0.5},
    })


def _switch_config(trainable=True):
    return SwitchSINDyLatentDynamicsConfig.model_validate({
        "type": "switch",
        "interpolator_type": "GP",
        "trainable": trainable,
        "switch": {"lstsq_reg": 0.0},
    })


def _switch_w_config(trainable=True):
    return SwitchSINDyWeakLatentDynamicsConfig.model_validate({
        "type": "switch_w",
        "interpolator_type": "GP",
        "trainable": trainable,
        "switch_w": {"test_func_type": "PC-poly", "test_func_width": 0.5, "overlap": 0.5},
    })


def _cable_config(trainable=True, top_k=2):
    return CABLELatentDynamicsConfig.model_validate({
        "type": "cable",
        "interpolator_type": "GP",
        "trainable": trainable,
        "cable": {"n_experts": 2, "top_k": top_k, "hidden_widths": [2], "activations": ["tanh"]},
    })


def _zero_cable_gate(ld):
    for layer in ld.w.layers:
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)


def test_sindy_rhs_matches_affine_model_for_strong_and_weak():
    params = numpy.array([[0.25]])
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    coefs = {
        "A": torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64),
        "b": torch.tensor([0.25, -0.75], dtype=torch.float64),
    }
    expected = z @ coefs["A"].T + coefs["b"].reshape(1, -1)

    for ld in [
        SINDy(n_z=2, Uniform_t_Grid=True, n_p=1, config=_sindy_config()),
        SINDy_weak(n_z=2, Uniform_t_Grid=True, n_p=1, config=_sindy_w_config()),
    ]:
        ld.set_train_coefs(params[0], {name: value.clone() for name, value in coefs.items()}, torch.device("cpu"))

        rhs = ld.RHS(Z=[[z]], t_Grid=[t], params=params)[0]

        assert isinstance(rhs, torch.Tensor)
        assert rhs.dtype == z.dtype
        assert rhs.device == z.device
        assert torch.allclose(rhs, expected)


def test_damped_spring_rhs_matches_second_order_model_for_strong_and_weak_with_batch_dimension():
    params = numpy.array([[0.25]])
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    z = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]],
    ], dtype=torch.float64)
    dz = 0.5 * z
    coefs = {
        "K": torch.tensor([[1.0, -1.0], [2.0, 0.5]], dtype=torch.float64),
        "C": torch.tensor([[0.25, 0.0], [0.0, -0.5]], dtype=torch.float64),
        "b": torch.tensor([1.0, -2.0], dtype=torch.float64),
    }
    expected = torch.matmul(z, coefs["K"].T) + torch.matmul(dz, coefs["C"].T) + coefs["b"].reshape(1, 1, -1)

    for ld in [
        DampedSpring(n_z=2, Uniform_t_Grid=True, n_p=1, config=_spring_config()),
        DampedSpring_weak(n_z=2, Uniform_t_Grid=True, n_p=1, config=_spring_w_config()),
    ]:
        ld.set_train_coefs(params[0], {name: value.clone() for name, value in coefs.items()}, torch.device("cpu"))

        rhs = ld.RHS(Z=[[z, dz]], t_Grid=[t], params=params)[0]

        assert isinstance(rhs, torch.Tensor)
        assert rhs.shape == z.shape
        assert torch.allclose(rhs, expected)


def test_switch_sindy_rhs_applies_before_after_systems_for_strong_and_weak_numpy_inputs():
    params = numpy.array([[0.25]])
    t = numpy.array([0.0, 0.5, 1.0])
    z = numpy.array([[0.0], [1.0], [2.0]])
    coefs = {
        "A_before": torch.tensor([[1.0]]),
        "b_before": torch.tensor([10.0]),
        "A_after": torch.tensor([[2.0]]),
        "b_after": torch.tensor([20.0]),
    }
    expected = numpy.array([[10.0], [22.0], [24.0]])

    for ld in [
        SwitchSINDy(n_z=1, Uniform_t_Grid=True, n_p=1, switch_time=lambda p: 0.5, config=_switch_config()),
        SwitchSINDy_weak(n_z=1, Uniform_t_Grid=True, n_p=1, switch_time=lambda p: 0.5, config=_switch_w_config()),
    ]:
        ld.set_train_coefs(params[0], {name: value.clone() for name, value in coefs.items()}, torch.device("cpu"))

        rhs = ld.RHS(Z=[[z]], t_Grid=[t], params=params)[0]

        assert isinstance(rhs, numpy.ndarray)
        assert rhs.shape == z.shape
        assert numpy.allclose(rhs, expected)


def test_cable_rhs_matches_uniform_mixture_of_affine_experts():
    params = numpy.array([[0.25]])
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    z = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)

    ld = CABLE(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_config())
    _zero_cable_gate(ld)
    ld.A = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32, requires_grad=True)
    ld.b = torch.tensor([[[10.0]], [[20.0]]], dtype=torch.float32, requires_grad=True)

    rhs = ld.RHS(Z=[[z]], t_Grid=[t], params=params)[0]

    expected = 2.0*z + 15.0
    assert isinstance(rhs, torch.Tensor)
    assert rhs.dtype == z.dtype
    assert rhs.shape == z.shape
    assert torch.allclose(rhs, expected)


def test_cable_simulate_integrates_constant_uniform_expert_mixture_numpy_inputs():
    params = numpy.array([[0.25]])
    t = numpy.array([0.0, 0.25, 0.5])
    z0 = numpy.array([[1.0], [2.0]])

    ld = CABLE(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_config())
    _zero_cable_gate(ld)
    ld.A = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)
    ld.b = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32, requires_grad=True)

    z = ld.simulate(IC=[[z0]], t_Grid=[t], params=params)[0][0]

    expected = z0.reshape(1, 2, 1) + 2.0*t.reshape(-1, 1, 1)
    assert isinstance(z, numpy.ndarray)
    assert z.shape == expected.shape
    assert numpy.allclose(z, expected)


def test_cable_compute_losses_uses_dense_pre_topk_weights_for_diversity_and_tail_diagnostic():
    params = numpy.array([[0.25]])
    t = torch.linspace(0.0, 1.0, 5)
    z = torch.zeros((5, 1))

    ld = CABLE(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_config(top_k=1))
    _zero_cable_gate(ld)
    ld.A = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)
    ld.b = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)

    loss_ld, loss_coef, loss_stab = ld.compute_losses(
        Latent_States=[[z]],
        loss_type="MSE",
        t_Grid=[t],
        params=params,
    )

    assert torch.allclose(loss_ld[0], torch.tensor(0.0))
    assert torch.allclose(loss_coef[0], torch.tensor(0.0))
    # The dense pre-top-k weights are [0.5, 0.5] at every time, so the global CV diversity loss is
    # zero even though the top-k RHS uses only one expert.
    assert torch.allclose(loss_stab[0], torch.tensor(0.0))
    # With two uniform dense weights and top_k = 1, the mass outside top-k is 0.5 at every sample.
    # The diagnostic tail-mass penalty is therefore mean(0.5**2) = 0.25. It is intentionally not
    # returned yet because the Trainer loss API still expects the legacy three-loss tuple.
    assert torch.allclose(ld.last_tail_mass_loss, torch.tensor(0.25))
    assert len(ld.last_tail_mass_loss_list) == 1
    assert torch.allclose(ld.last_tail_mass_loss_list[0], torch.tensor(0.25))
