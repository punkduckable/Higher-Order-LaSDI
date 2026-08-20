import os
import sys

import numpy
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.LatentDynamics import CABLE, CABLE_weak, DampedSpring, DampedSpring_weak, SINDy, SINDy_weak, SwitchSINDy, SwitchSINDy_weak
from HLaSDI.Schemas import (
    CABLELatentDynamicsConfig,
    DampedSpringLatentDynamicsConfig,
    DampedSpringWeakLatentDynamicsConfig,
    SINDyLatentDynamicsConfig,
    SINDyWeakLatentDynamicsConfig,
    SwitchSINDyLatentDynamicsConfig,
    SwitchSINDyWeakLatentDynamicsConfig,
    WeakCABLELatentDynamicsConfig,
)


def _gp_interpolator_config():
    return {
        "type": "GP",
        "GP": {
            "kernel": {
                "type": "Matern",
                "length_scale": 1.0,
                "length_scale_bounds": (1.0, 1.0e3),
                "nu": 2.5,
            },
            "constant_value": 1.0,
            "constant_value_bounds": (1.0e-3, 1.0e3),
            "alpha": 4.0e-4,
            "n_restarts_optimizer": 10,
            "random_state": 1,
        },
    }


def _sindy_config(trainable=True):
    return SINDyLatentDynamicsConfig.model_validate({
        "type": "sindy",
        "interpolator": _gp_interpolator_config(),
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "stab": 1.0},
        "sindy": {"lstsq_reg": 0.0},
    })


def _sindy_w_config(trainable=True):
    return SINDyWeakLatentDynamicsConfig.model_validate({
        "type": "sindy_w",
        "interpolator": _gp_interpolator_config(),
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "stab": 1.0},
        "sindy_w": {"test_func_type": "PC-poly", "test_func_width": 0.5, "overlap": 0.5},
    })


def _spring_config(trainable=True):
    return DampedSpringLatentDynamicsConfig.model_validate({
        "type": "spring",
        "interpolator": _gp_interpolator_config(),
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "stab": 1.0},
        "spring": {"lstsq_reg": 0.0},
    })


def _spring_w_config(trainable=True):
    return DampedSpringWeakLatentDynamicsConfig.model_validate({
        "type": "spring_w",
        "interpolator": _gp_interpolator_config(),
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "stab": 1.0},
        "spring_w": {"test_func_type": "PC-poly", "test_func_width": 0.5, "overlap": 0.5},
    })


def _switch_config(trainable=True):
    return SwitchSINDyLatentDynamicsConfig.model_validate({
        "type": "switch",
        "interpolator": _gp_interpolator_config(),
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "stab": 1.0},
        "switch": {"lstsq_reg": 0.0},
    })


def _switch_w_config(trainable=True):
    return SwitchSINDyWeakLatentDynamicsConfig.model_validate({
        "type": "switch_w",
        "interpolator": _gp_interpolator_config(),
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "stab": 1.0},
        "switch_w": {"test_func_type": "PC-poly", "test_func_width": 0.5, "overlap": 0.5},
    })


def _cable_config(trainable=True, n_active=2):
    return CABLELatentDynamicsConfig.model_validate({
        "type": "cable",
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "diversity": 1.0, "tail": 1.0},
        "cable": {
            "n_experts": 2,
            "n_active": n_active,
            "hidden_widths": [2],
            "activations": ["tanh"],
            "use_biases": True,
            "coef_norm": "l2",
            "use_mask": False,
        },
    })


def _cable_config_with_settings(trainable=True, n_active=2, **settings):
    cable_settings = {
        "n_experts": 2,
        "n_active": n_active,
        "hidden_widths": [2],
        "activations": ["tanh"],
        "use_biases": True,
        "coef_norm": "l2",
        "use_mask": False,
    }
    cable_settings.update(settings)
    return CABLELatentDynamicsConfig.model_validate({
        "type": "cable",
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "diversity": 1.0, "tail": 1.0},
        "cable": cable_settings,
    })


def _cable_w_config(trainable=True, n_active=2):
    return WeakCABLELatentDynamicsConfig.model_validate({
        "type": "cable_w",
        "trainable": trainable,
        "loss_weights": {"LD": 1.0, "coef": 1.0, "diversity": 1.0, "tail": 1.0},
        "cable": {
            "n_experts": 2,
            "n_active": n_active,
            "hidden_widths": [2],
            "activations": ["tanh"],
            "use_biases": True,
            "coef_norm": "l2",
            "use_mask": False,
        },
        "weak": {"test_func_type": "PC-poly", "test_func_width": 0.5, "overlap": 0.5},
    })


def _zero_cable_gate(ld):
    for layer in ld.w.layers:
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)


def _loss_metric_keys(metrics):
    return {key for key in metrics if key.startswith("loss/")}


def _assert_loss_metrics_have_total_suffix(metrics):
    for key in _loss_metric_keys(metrics):
        assert key.endswith("/total")


def test_cable_rhs_can_use_latent_state_in_gate_inputs():
    params = numpy.array([[0.25]])
    t = torch.tensor([0.0, 1.0], dtype=torch.float64)
    z = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)

    ld = CABLE(
        n_z=1,
        Uniform_t_Grid=True,
        n_p=1,
        config=_cable_config_with_settings(use_z_in_gate=True, hidden_widths=[1]),
    )
    with torch.no_grad():
        first, second = ld.w.layers
        first.weight.zero_()
        first.bias.zero_()
        second.weight.zero_()
        second.bias.zero_()
        # Gate inputs are [tau, params, z]. Make the logits depend only on z.
        first.weight[0, -1] = 1.0
        second.weight[0, 0] = 1.0
        second.weight[1, 0] = -1.0
    ld.unmasked_A = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float32, requires_grad=True)

    rhs = ld.RHS(Z=[[z]], t_Grid=[t], params=params)[0]

    hidden = torch.tanh(z[:, 0].to(dtype=torch.float32))
    expected_weights = torch.softmax(torch.stack([hidden, -hidden], dim=1), dim=1)
    expected = expected_weights[:, 1].to(dtype=z.dtype).reshape(-1, 1)
    assert isinstance(rhs, torch.Tensor)
    assert rhs.shape == z.shape
    assert torch.allclose(rhs, expected)
    assert rhs[0, 0] > rhs[1, 0]


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
    ld.unmasked_A = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.tensor([[[10.0]], [[20.0]]], dtype=torch.float32, requires_grad=True)

    rhs = ld.RHS(Z=[[z]], t_Grid=[t], params=params)[0]

    expected = 2.0*z + 15.0
    assert isinstance(rhs, torch.Tensor)
    assert rhs.dtype == z.dtype
    assert rhs.shape == z.shape
    assert torch.allclose(rhs, expected)


def test_cable_rhs_omits_bias_when_biases_are_disabled():
    params = numpy.array([[0.25]])
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    z = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)

    ld = CABLE(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_config_with_settings(use_biases=False))
    _zero_cable_gate(ld)
    ld.unmasked_A = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32, requires_grad=True)

    rhs = ld.RHS(Z=[[z]], t_Grid=[t], params=params)[0]

    expected = 2.0*z
    assert ld.b is None
    assert isinstance(rhs, torch.Tensor)
    assert rhs.dtype == z.dtype
    assert rhs.shape == z.shape
    assert torch.allclose(rhs, expected)


def test_cable_simulate_integrates_constant_uniform_expert_mixture_numpy_inputs():
    params = numpy.array([[0.25]])
    t = numpy.array([0.0, 0.25, 0.5])
    z0 = numpy.array([1.0])

    ld = CABLE(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_config())
    _zero_cable_gate(ld)
    ld.unmasked_A = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32, requires_grad=True)

    z = ld.simulate(IC=[[z0]], t_Grid=[t], params=params)[0][0]

    expected = z0.reshape(1, 1) + 2.0*t.reshape(-1, 1)
    assert isinstance(z, numpy.ndarray)
    assert z.shape == expected.shape
    assert numpy.allclose(z, expected)


def test_cable_simulate_integrates_constant_uniform_expert_mixture_torch_inputs():
    params = numpy.array([[0.25]])
    t = numpy.array([0.0, 0.25, 0.5])
    z0 = torch.tensor([1.0], dtype=torch.float64)

    ld = CABLE(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_config())
    _zero_cable_gate(ld)
    ld.unmasked_A = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32, requires_grad=True)

    z = ld.simulate(IC=[[z0]], t_Grid=[t], params=params)[0][0]

    expected = z0.reshape(1, 1) + 2.0*torch.tensor(t, dtype=z0.dtype).reshape(-1, 1)
    assert isinstance(z, torch.Tensor)
    assert z.dtype == z0.dtype
    assert z.shape == expected.shape
    assert torch.allclose(z, expected)


def test_cable_compute_losses_updates_and_applies_hard_coefficient_masks():
    params = numpy.array([[0.25]])
    t = torch.linspace(0.0, 1.0, 5)
    z = torch.zeros((5, 1))

    ld = CABLE(
        n_z=1,
        Uniform_t_Grid=True,
        n_p=1,
        config=_cable_config_with_settings(
            use_mask=True,
            mask_threshold=0.5,
            first_mask_step=2,
            mask_update_freq=3,
        ),
    )
    _zero_cable_gate(ld)
    ld.unmasked_A = torch.tensor([[[0.25]], [[2.0]]], dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.tensor([[[0.25]], [[3.0]]], dtype=torch.float32, requires_grad=True)

    ld.compute_losses(Latent_States=[[z]], t_Grid=[t], step=1, params=params)

    assert torch.allclose(ld.A_mask, torch.ones_like(ld.A_mask))
    assert torch.allclose(ld.b_mask, torch.ones_like(ld.b_mask))

    ld.compute_losses(Latent_States=[[z]], t_Grid=[t], step=2, params=params)

    assert torch.allclose(ld.A_mask, torch.tensor([[[0.0]], [[1.0]]]))
    assert torch.allclose(ld.b_mask, torch.tensor([[[0.0]], [[1.0]]]))
    assert torch.allclose(ld.A.detach(), torch.tensor([[[0.0]], [[2.0]]]))
    assert torch.allclose(ld.b.detach(), torch.tensor([[[0.0]], [[3.0]]]))


def test_cable_export_load_restores_unmasked_coefficients_and_masks():
    ld = CABLE(
        n_z=1,
        Uniform_t_Grid=True,
        n_p=1,
        config=_cable_config_with_settings(
            use_mask=True,
            mask_threshold=0.5,
            first_mask_step=1,
            mask_update_freq=1,
        ),
    )
    ld.unmasked_A = torch.tensor([[[0.0]], [[2.0]]], dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.tensor([[[0.0]], [[3.0]]], dtype=torch.float32, requires_grad=True)
    ld.A_mask = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float32)
    ld.b_mask = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float32)

    exported = ld.export()

    ld2 = CABLE(
        n_z=1,
        Uniform_t_Grid=True,
        n_p=1,
        config=_cable_config_with_settings(
            use_mask=True,
            mask_threshold=0.5,
            first_mask_step=1,
            mask_update_freq=1,
        ),
    )
    ld2.load(exported)

    assert torch.allclose(ld2.unmasked_A, ld.unmasked_A)
    assert torch.allclose(ld2.unmasked_b, ld.unmasked_b)
    assert torch.allclose(ld2.A_mask, ld.A_mask)
    assert torch.allclose(ld2.b_mask, ld.b_mask)
    assert torch.allclose(ld2.A, torch.tensor([[[0.0]], [[2.0]]]))
    assert torch.allclose(ld2.b, torch.tensor([[[0.0]], [[3.0]]]))
    assert ld2.unmasked_A.requires_grad and ld2.unmasked_A.is_leaf
    assert ld2.unmasked_b.requires_grad and ld2.unmasked_b.is_leaf


def test_cable_compute_losses_uses_dense_pre_topk_weights_for_diversity_and_tail_diagnostic():
    params = numpy.array([[0.25]])
    t = torch.linspace(0.0, 1.0, 5)
    z = torch.zeros((5, 1))

    ld = CABLE(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_config(n_active=1))
    _zero_cable_gate(ld)
    ld.unmasked_A = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)

    result = ld.compute_losses(
        Latent_States=[[z]],
        t_Grid=[t],
        step=0,
        params=params,
    )
    losses = result.losses

    assert set(losses.keys()) == {"LD", "coef", "diversity", "tail"}
    assert torch.allclose(losses["LD"], torch.tensor(0.0))
    assert torch.allclose(losses["coef"], torch.tensor(0.0))
    # The dense weights are [0.5, 0.5] at every time, so the global CV diversity loss is zero.
    assert torch.allclose(losses["diversity"], torch.tensor(0.0))
    # With two uniform dense weights and n_active = 1, the mass outside the largest active-weight
    # set is 0.5 at every sample.
    # The tail-mass penalty is therefore mean(0.5**2) = 0.25.
    assert torch.allclose(losses["tail"], torch.tensor(0.25))
    assert torch.allclose(result.metrics["loss/tail/total"], torch.tensor(0.25))
    assert torch.allclose(result.metrics["expert/num_engaged/mean"], torch.tensor(2.0))
    assert torch.allclose(result.metrics["expert/num_engaged/std"], torch.tensor(0.0))
    assert torch.allclose(result.metrics["expert/num_engaged/min"], torch.tensor(2.0))
    assert torch.allclose(result.metrics["expert/num_engaged/max"], torch.tensor(2.0))
    assert torch.allclose(result.metrics["expert/times_engaged/mean"], torch.tensor(5.0))
    assert torch.allclose(result.metrics["expert/times_engaged/std"], torch.tensor(0.0))
    assert torch.allclose(result.metrics["expert/times_engaged/min"], torch.tensor(5.0))
    assert torch.allclose(result.metrics["expert/times_engaged/max"], torch.tensor(5.0))
    assert torch.allclose(result.metrics["expert/num_ever_engaged"], torch.tensor(2.0))
    assert _loss_metric_keys(result.metrics) == {
        "loss/LD/total",
        "loss/coef/A",
        "loss/coef/b",
        "loss/coef/total",
        "loss/diversity/total",
        "loss/tail/total",
    }
    _assert_loss_metrics_have_total_suffix(result.metrics)
    assert torch.allclose(ld.last_tail_mass_loss, torch.tensor(0.25))
    assert len(ld.last_tail_mass_loss_list) == 1
    assert torch.allclose(ld.last_tail_mass_loss_list[0], torch.tensor(0.25))


def test_non_cable_latent_dynamics_log_loss_totals_only():
    params = numpy.array([[0.25], [0.75]])
    t = torch.linspace(0.0, 1.0, 5)
    z = torch.zeros((5, 1))

    ld = SINDy(n_z=1, Uniform_t_Grid=True, n_p=1, config=_sindy_config())
    for param in params:
        ld.set_train_coefs(
            param,
            {"A": torch.zeros((1, 1)), "b": torch.zeros(1)},
            torch.device("cpu"),
        )

    result = ld.compute_losses(
        Latent_States=[[z], [z]],
        t_Grid=[t, t],
        step=0,
        params=params,
    )

    assert _loss_metric_keys(result.metrics) == {
        "loss/LD/total",
        "loss/coef/total",
        "loss/stab/total",
    }
    _assert_loss_metrics_have_total_suffix(result.metrics)


def test_cable_global_losses_are_not_divided_by_number_of_parameters():
    params = numpy.array([[0.25], [0.75]])
    t = torch.linspace(0.0, 1.0, 5)
    z = torch.zeros((5, 1))

    ld = CABLE(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_config(n_active=2))
    _zero_cable_gate(ld)
    ld.unmasked_A = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)

    losses = ld.compute_losses(
        Latent_States=[[z], [z]],
        t_Grid=[t, t],
        step=0,
        params=params,
    ).losses

    assert torch.allclose(losses["coef"], torch.tensor(4.0))
    assert torch.allclose(losses["diversity"], torch.tensor(0.0))


def test_cable_weak_compute_losses_returns_scalar_totals_and_metrics():
    params = numpy.array([[0.25]])
    t = torch.linspace(0.0, 1.0, 9)
    z = torch.zeros((9, 1))

    ld = CABLE_weak(n_z=1, Uniform_t_Grid=True, n_p=1, config=_cable_w_config(n_active=1))
    _zero_cable_gate(ld)
    ld.unmasked_A = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)
    ld.unmasked_b = torch.zeros((2, 1, 1), dtype=torch.float32, requires_grad=True)
    ld.add_weight_functions(params[0], t)

    result = ld.compute_losses(
        Latent_States=[[z]],
        t_Grid=[t],
        step=0,
        params=params,
    )

    assert set(result.losses.keys()) == {"LD", "coef", "diversity", "tail"}
    assert all(loss.ndim == 0 for loss in result.losses.values())
    assert torch.allclose(result.losses["LD"], torch.tensor(0.0))
    assert torch.allclose(result.metrics["loss/LD/total"], torch.tensor(0.0))
    assert torch.allclose(result.metrics["expert/num_engaged/mean"], torch.tensor(2.0))
    assert torch.allclose(result.metrics["expert/num_engaged/std"], torch.tensor(0.0))
    assert torch.allclose(result.metrics["expert/num_engaged/min"], torch.tensor(2.0))
    assert torch.allclose(result.metrics["expert/num_engaged/max"], torch.tensor(2.0))
    assert torch.allclose(result.metrics["expert/times_engaged/mean"], torch.tensor(9.0))
    assert torch.allclose(result.metrics["expert/times_engaged/std"], torch.tensor(0.0))
    assert torch.allclose(result.metrics["expert/times_engaged/min"], torch.tensor(9.0))
    assert torch.allclose(result.metrics["expert/times_engaged/max"], torch.tensor(9.0))
    assert torch.allclose(result.metrics["expert/num_ever_engaged"], torch.tensor(2.0))
    assert _loss_metric_keys(result.metrics) == {
        "loss/LD/total",
        "loss/coef/total",
        "loss/diversity/total",
        "loss/tail/total",
    }
    _assert_loss_metrics_have_total_suffix(result.metrics)
