import os
import sys

import numpy
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from LatentDynamics import DampedSpring, DampedSpring_weak, SINDy, SINDy_weak, SwitchSINDy, SwitchSINDy_weak
from Schemas import (
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
        SINDy(n_z=2, Uniform_t_Grid=True, config=_sindy_config()),
        SINDy_weak(n_z=2, Uniform_t_Grid=True, config=_sindy_w_config()),
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
        DampedSpring(n_z=2, Uniform_t_Grid=True, config=_spring_config()),
        DampedSpring_weak(n_z=2, Uniform_t_Grid=True, config=_spring_w_config()),
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
        SwitchSINDy(n_z=1, Uniform_t_Grid=True, switch_time=lambda p: 0.5, config=_switch_config()),
        SwitchSINDy_weak(n_z=1, Uniform_t_Grid=True, switch_time=lambda p: 0.5, config=_switch_w_config()),
    ]:
        ld.set_train_coefs(params[0], {name: value.clone() for name, value in coefs.items()}, torch.device("cpu"))

        rhs = ld.RHS(Z=[[z]], t_Grid=[t], params=params)[0]

        assert isinstance(rhs, numpy.ndarray)
        assert rhs.shape == z.shape
        assert numpy.allclose(rhs, expected)
