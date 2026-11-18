"""Numerical and gradient tests for the batched affine second-order RK4 path."""

import os
import sys

import numpy
import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.LatentDynamics import SOME, SOME_weak
from HLaSDI.Schemas import SOMELatentDynamicsConfig, WeakSOMELatentDynamicsConfig
from HLaSDI.Utilities.SecondOrderSolvers import RK4, RK4_affine_batched


def _some_config(
        weak: bool, use_biases: bool
        ) -> SOMELatentDynamicsConfig | WeakSOMELatentDynamicsConfig:
    """Build matching strong/weak SOME models with time-only gates."""
    data = {
        "type": "some_w" if weak else "some",
        "trainable": True,
        "loss_weights": {
            "LD": 1.0, "coef": 1.0, "diversity/D": 1.0, "diversity/V": 1.0,
            "tail/D": 1.0, "tail/V": 1.0,
        },
        "some": {
            "n_experts_D": 2, "n_experts_V": 2,
            "n_active_D": 1, "n_active_V": 1,
            "hidden_widths": [2], "activations": ["tanh"],
            "use_biases": use_biases, "coef_norm": "l2", "use_mask": True,
            "mask_threshold": 1.0e-3, "first_mask_step": 100,
            "mask_update_freq": 100,
        },
    }
    if weak:
        data["weak"] = {
            "test_func_type": "PC-poly", "test_func_width": 0.5, "overlap": 0.5,
        }
        return WeakSOMELatentDynamicsConfig.model_validate(data)
    return SOMELatentDynamicsConfig.model_validate(data)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_affine_batched_rk4_matches_scalar_on_nonuniform_padded_windows(device: str) -> None:
    """Different grid lengths and stage times match independent scalar RK4 solves."""
    dtype = torch.float64
    grid = torch.tensor([
        [0.1, 0.2, 0.45, 0.9],
        [0.0, 0.4, 0.8, 0.8],
        [0.3, 0.3, 0.3, 0.3],
    ], dtype=dtype, device=device)
    lengths = torch.tensor([4, 3, 1], device=device)
    y0 = torch.tensor([[1.0, -0.5], [0.2, 0.8], [-0.3, 0.4]], dtype=dtype, device=device)
    v0 = torch.tensor([[0.1, 0.2], [-0.4, 0.3], [0.5, -0.2]], dtype=dtype, device=device)
    parameter = torch.tensor([0.2, 0.6, 0.9], dtype=dtype, device=device)

    start = grid[:, :-1]
    h = grid[:, 1:] - start
    stage_times = torch.stack((start, start + h / 2, start + h), dim=2)
    eye = torch.eye(2, dtype=dtype, device=device)
    K = (0.1 + 0.2 * parameter[:, None, None] + 0.3 * stage_times)[..., None, None] * eye
    C = (-0.1 + 0.05 * stage_times)[..., None, None] * eye
    b = (0.2 + 0.1 * stage_times)[..., None] * torch.tensor([1.0, -1.0], dtype=dtype, device=device)

    y_batch, v_batch = RK4_affine_batched(y0, v0, grid, lengths, K, C, b)
    assert y_batch.shape == v_batch.shape == (3, 4, 2)
    assert y_batch.dtype == v_batch.dtype == dtype
    assert y_batch.device == v_batch.device == grid.device

    for row, length in enumerate(lengths.tolist()):
        def rhs(t: float, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return ((0.1 + 0.2 * parameter[row] + 0.3 * t) * y
                    + (-0.1 + 0.05 * t) * v
                    + (0.2 + 0.1 * t) * torch.tensor([1.0, -1.0], dtype=dtype, device=device))

        y_scalar, v_scalar = RK4(rhs, y0[row], v0[row], grid[row, :length].cpu().numpy())
        torch.testing.assert_close(y_batch[row, :length], y_scalar, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(v_batch[row, :length], v_scalar, rtol=1.0e-12, atol=1.0e-12)
        assert torch.equal(y_batch[row, length:], y_batch[row, length - 1].expand(4 - length, 2))
        assert torch.equal(v_batch[row, length:], v_batch[row, length - 1].expand(4 - length, 2))


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_affine_batched_rk4_gradients_match_scalar_reference(device: str) -> None:
    """Initial-state and shared affine-parameter gradients include only valid frames."""
    dtype = torch.float64
    grid = torch.tensor([[0.0, 0.2, 0.5], [0.1, 0.4, 0.4]], dtype=dtype, device=device)
    lengths = torch.tensor([3, 2], device=device)
    y_values = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=dtype, device=device)
    v_values = torch.tensor([[-0.3, 0.4], [0.1, -0.2]], dtype=dtype, device=device)
    K_values = torch.tensor([[0.1, 0.2], [-0.1, 0.3]], dtype=dtype, device=device)
    C_values = torch.tensor([[-0.2, 0.1], [0.05, -0.1]], dtype=dtype, device=device)
    b_values = torch.tensor([0.1, -0.2], dtype=dtype, device=device)

    y0, v0, K0, C0, b0 = [x.clone().requires_grad_() for x in
                          (y_values, v_values, K_values, C_values, b_values)]
    K = K0.expand(2, 2, 3, 2, 2)
    C = C0.expand(2, 2, 3, 2, 2)
    b = b0.expand(2, 2, 3, 2)
    y_batch, v_batch = RK4_affine_batched(y0, v0, grid, lengths, K, C, b)
    valid = (torch.arange(3, device=device)[None, :] < lengths[:, None])[..., None]
    loss_batch = ((y_batch.square() + v_batch.square()) * valid).sum()
    grads_batch = torch.autograd.grad(loss_batch, (y0, v0, K0, C0, b0))

    ys, vs, Ks, Cs, bs = [x.clone().requires_grad_() for x in
                          (y_values, v_values, K_values, C_values, b_values)]
    loss_scalar = torch.zeros((), dtype=dtype, device=device)
    for row, length in enumerate(lengths.tolist()):
        def rhs(t: float, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return Ks @ y + Cs @ v + bs

        y_ref, v_ref = RK4(rhs, ys[row], vs[row], grid[row, :length].cpu().numpy())
        loss_scalar = loss_scalar + y_ref.square().sum() + v_ref.square().sum()
    grads_scalar = torch.autograd.grad(loss_scalar, (ys, vs, Ks, Cs, bs))

    for actual, expected in zip(grads_batch, grads_scalar):
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_some_batched_single_frame_returns_initial_state() -> None:
    """A one-frame window needs no gate evaluation or RK stages."""
    model = SOME(n_z=1, Uniform_t_Grid=True, n_p=1, config=_some_config(False, True))
    y0 = torch.tensor([[0.4], [-0.3]], requires_grad=True)
    v0 = torch.tensor([[0.2], [0.1]], requires_grad=True)
    grid = torch.tensor([[0.1], [0.5]])
    lengths = torch.ones(2, dtype=torch.long)
    params = torch.tensor([[0.2], [0.8]])
    reference_t0 = torch.zeros(2)
    reference_t_span = torch.ones(2)

    K, C, b = model.precompute_affine_stage_coefficients(
        grid, lengths, params, reference_t0, reference_t_span, state_dtype=y0.dtype)
    assert K.shape == C.shape == (2, 0, 3, 1, 1)
    assert b.shape == (2, 0, 3, 1)
    y, v = model.simulate_batched(
        y0, v0, grid, lengths, params, reference_t0, reference_t_span)
    assert torch.equal(y[:, 0], y0)
    assert torch.equal(v[:, 0], v0)
    (y.sum() + v.sum()).backward()
    assert torch.equal(y0.grad, torch.ones_like(y0))
    assert torch.equal(v0.grad, torch.ones_like(v0))


@pytest.mark.parametrize("weak", [False, True])
@pytest.mark.parametrize("use_biases", [False, True])
def test_some_batched_stage_coefficients_match_scalar_simulation_and_gradients(
        weak: bool, use_biases: bool) -> None:
    """SOME and SOME_weak preserve full-domain gate scaling and parameter gradients."""
    torch.manual_seed(7)
    model_type = SOME_weak if weak else SOME
    config = _some_config(weak, use_biases)
    batched = model_type(n_z=1, Uniform_t_Grid=False, n_p=1, config=config)
    batched.K_mask[0, 0, 0] = 0.0
    batched.C_mask[1, 0, 0] = 0.0
    if use_biases:
        batched.b_mask[0, 0, 0] = 0.0
    scalar = model_type(n_z=1, Uniform_t_Grid=False, n_p=1, config=config)
    scalar.load(batched.export())

    grid = torch.tensor([[0.1, 0.22, 0.5, 0.8], [0.4, 0.6, 0.9, 0.9]], dtype=torch.float32)
    lengths = torch.tensor([4, 3])
    params = torch.tensor([[0.2], [0.7]], dtype=torch.float32)
    reference_t0 = torch.zeros(2)
    reference_t_span = torch.ones(2)
    initial_y_values = torch.tensor([[0.3], [-0.2]], dtype=torch.float32)
    initial_v_values = torch.tensor([[0.1], [0.4]], dtype=torch.float32)
    y0 = initial_y_values.clone().requires_grad_()
    v0 = initial_v_values.clone().requires_grad_()

    K, C, b = batched.precompute_affine_stage_coefficients(
        grid, lengths, params, reference_t0, reference_t_span, state_dtype=y0.dtype)
    assert K.shape == C.shape == (2, 3, 3, 1, 1)
    assert b.shape == (2, 3, 3, 1)
    # Window 1's last interval is padding: all three stages use its final valid time.
    for coefficients in (K, C, b):
        padded_stages = coefficients[1, 2]
        torch.testing.assert_close(
            padded_stages, padded_stages[0].expand_as(padded_stages), rtol=0, atol=0)
    terminal_weights_D, terminal_weights_V = batched._weights(
        t_Grid=grid[1, 2:3], params=params[1],
        t0=reference_t0[1], t_span=reference_t_span[1])
    terminal_coefficients = batched._effective_coefficients(
        terminal_weights_D, terminal_weights_V, grid.device, y0.dtype)
    for padded, terminal in zip((K, C, b), terminal_coefficients):
        torch.testing.assert_close(padded[1, 2, 0], terminal[0], rtol=1.0e-6, atol=1.0e-7)
    assert K.requires_grad and C.requires_grad
    assert b.requires_grad == use_biases
    if not use_biases:
        assert torch.count_nonzero(b) == 0
    y_batch, v_batch = batched.simulate_batched(
        y0, v0, grid, lengths, params, reference_t0, reference_t_span)
    valid = (torch.arange(4)[None, :] < lengths[:, None])[..., None]
    loss_batch = ((y_batch.square() + v_batch.square()) * valid).sum()
    grads_batch = torch.autograd.grad(loss_batch, (y0, v0, *batched.parameters()))

    ys = initial_y_values.clone().requires_grad_()
    vs = initial_v_values.clone().requires_grad_()
    loss_scalar = torch.zeros(())
    for row, length in enumerate(lengths.tolist()):
        grid_np = grid[row, :length].numpy()
        domain = numpy.array([0.0, 1.0], dtype=numpy.float32)
        y_ref, v_ref = scalar.simulate(
            IC=[[ys[row], vs[row]]],
            t_Grid=[grid_np],
            params=params[row:row + 1].numpy(),
            time_domain=[domain])[0]
        torch.testing.assert_close(y_batch[row, :length], y_ref, rtol=3.0e-5, atol=3.0e-6)
        torch.testing.assert_close(v_batch[row, :length], v_ref, rtol=3.0e-5, atol=3.0e-6)
        loss_scalar = loss_scalar + y_ref.square().sum() + v_ref.square().sum()
    grads_scalar = torch.autograd.grad(loss_scalar, (ys, vs, *scalar.parameters()))
    for actual, expected in zip(grads_batch, grads_scalar):
        torch.testing.assert_close(actual, expected, rtol=5.0e-4, atol=5.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_some_batched_cuda_forward_and_backward() -> None:
    """The gate, experts, states, and precomputed coefficients stay on the GPU."""
    model = SOME(n_z=1, Uniform_t_Grid=False, n_p=1, config=_some_config(False, True))
    model.move_parameters_to_device("cuda")
    y0 = torch.tensor([[0.3], [0.1]], device="cuda", requires_grad=True)
    v0 = torch.tensor([[0.2], [-0.4]], device="cuda", requires_grad=True)
    grid = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.5, 0.5]], device="cuda")
    lengths = torch.tensor([3, 2], device="cuda")
    params = torch.tensor([[0.2], [0.7]], device="cuda")
    reference_t0 = torch.zeros(2, device="cuda")
    reference_t_span = torch.ones(2, device="cuda")

    y, v = model.simulate_batched(
        y0, v0, grid, lengths, params, reference_t0, reference_t_span)
    assert y.device.type == v.device.type == "cuda"
    (y.sum() + v.sum()).backward()
    assert y0.grad is not None and torch.isfinite(y0.grad).all()
    assert v0.grad is not None and torch.isfinite(v0.grad).all()
    assert model.unmasked_K.grad is not None and torch.isfinite(model.unmasked_K.grad).all()
    assert model.unmasked_C.grad is not None and torch.isfinite(model.unmasked_C.grad).all()
    assert any(param.grad is not None for param in model.w.parameters())
