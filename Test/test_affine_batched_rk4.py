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
from HLaSDI.Utilities import SecondOrderSolvers as second_order_solvers
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

    y_batch, v_batch = RK4_affine_batched(
        y0, v0, grid.T.contiguous(), lengths,
        K.permute(1, 2, 0, 3, 4).contiguous(),
        C.permute(1, 2, 0, 3, 4).contiguous(),
        b.permute(1, 2, 0, 3).contiguous())
    assert y_batch.shape == v_batch.shape == (4, 3, 2)
    assert y_batch.dtype == v_batch.dtype == dtype
    assert y_batch.device == v_batch.device == grid.device

    for row, length in enumerate(lengths.tolist()):
        def rhs(t: float, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return ((0.1 + 0.2 * parameter[row] + 0.3 * t) * y
                    + (-0.1 + 0.05 * t) * v
                    + (0.2 + 0.1 * t) * torch.tensor([1.0, -1.0], dtype=dtype, device=device))

        y_scalar, v_scalar = RK4(rhs, y0[row], v0[row], grid[row, :length].cpu().numpy())
        torch.testing.assert_close(y_batch[:length, row], y_scalar, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(v_batch[:length, row], v_scalar, rtol=1.0e-12, atol=1.0e-12)
        assert torch.equal(y_batch[length:, row], y_batch[length - 1, row].expand(4 - length, 2))
        assert torch.equal(v_batch[length:, row], v_batch[length - 1, row].expand(4 - length, 2))


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
    K = K0.expand(2, 3, 2, 2, 2).contiguous()
    C = C0.expand(2, 3, 2, 2, 2).contiguous()
    b = b0.expand(2, 3, 2, 2).contiguous()
    y_batch, v_batch = RK4_affine_batched(y0, v0, grid.T.contiguous(), lengths, K, C, b)
    valid = (torch.arange(3, device=device)[:, None] < lengths[None, :])[..., None]
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


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_affine_batched_rk4_dense_time_dependent_coefficients_match_scalar_reference(
        device: str) -> None:
    """Non-symmetric stages and gradients match scalar RK4 across a chunk boundary."""
    dtype = torch.float64
    grid = torch.tensor([
        [0.1, 0.17, 0.25, 0.31, 0.40, 0.48, 0.55, 0.66, 0.75, 0.82, 0.9, 1.0],
        [0.0, 0.08, 0.16, 0.24, 0.35, 0.48, 0.60, 0.72, 0.8, 0.8, 0.8, 0.8],
    ], dtype=dtype, device=device)
    lengths = torch.tensor([12, 9], device=device)
    values = (
        torch.tensor([[0.3, -0.2], [-0.4, 0.5]], dtype=dtype, device=device),
        torch.tensor([[0.1, 0.4], [0.2, -0.3]], dtype=dtype, device=device),
        torch.tensor([[[0.2, -0.3], [0.1, -0.1]], [[-0.1, 0.2], [0.3, 0.1]]], dtype=dtype, device=device),
        torch.tensor([[[0.1, 0.2], [-0.2, 0.05]], [[0.03, -0.1], [0.2, -0.04]]], dtype=dtype, device=device),
        torch.tensor([[[-0.2, 0.1], [0.05, -0.1]], [[0.1, 0.04], [-0.2, 0.03]]], dtype=dtype, device=device),
        torch.tensor([[[0.03, -0.08], [0.06, 0.02]], [[-0.05, 0.07], [0.01, -0.03]]], dtype=dtype, device=device),
        torch.tensor([[0.04, -0.07], [0.08, 0.02]], dtype=dtype, device=device),
        torch.tensor([[-0.03, 0.05], [0.02, -0.06]], dtype=dtype, device=device),
    )
    y0, v0, K0, K1, C0, C1, b0, b1 = (value.clone().requires_grad_() for value in values)

    start = grid[:, :-1]
    h = grid[:, 1:] - start
    stage_times = torch.stack((start, start + h / 2, start + h), dim=2)
    K = K0[:, None, None] + stage_times[..., None, None] * K1[:, None, None]
    C = C0[:, None, None] + stage_times[..., None, None] * C1[:, None, None]
    b = b0[:, None, None] + stage_times[..., None] * b1[:, None, None]
    y_batch, v_batch = RK4_affine_batched(
        y0, v0, grid.T.contiguous(), lengths,
        K.permute(1, 2, 0, 3, 4).contiguous(),
        C.permute(1, 2, 0, 3, 4).contiguous(),
        b.permute(1, 2, 0, 3).contiguous())

    valid = (torch.arange(grid.shape[1], device=device)[:, None] < lengths[None, :])[..., None]
    loss_batch = ((y_batch.square() + 0.37 * v_batch.square()) * valid).sum()
    grads_batch = torch.autograd.grad(loss_batch, (y0, v0, K0, K1, C0, C1, b0, b1))

    ys, vs, Ks0, Ks1, Cs0, Cs1, bs0, bs1 = (value.clone().requires_grad_() for value in values)
    loss_scalar = torch.zeros((), dtype=dtype, device=device)
    for row, length in enumerate(lengths.tolist()):
        def rhs(t: float, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return ((Ks0[row] + t * Ks1[row]) @ y
                    + (Cs0[row] + t * Cs1[row]) @ v + bs0[row] + t * bs1[row])

        y_ref, v_ref = RK4(rhs, ys[row], vs[row], grid[row, :length].cpu().numpy())
        torch.testing.assert_close(y_batch[:length, row], y_ref, rtol=1.0e-11, atol=1.0e-12)
        torch.testing.assert_close(v_batch[:length, row], v_ref, rtol=1.0e-11, atol=1.0e-12)
        loss_scalar = loss_scalar + y_ref.square().sum() + 0.37 * v_ref.square().sum()
    grads_scalar = torch.autograd.grad(loss_scalar, (ys, vs, Ks0, Ks1, Cs0, Cs1, bs0, bs1))
    for actual, expected in zip(grads_batch, grads_scalar):
        torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-12)


def test_affine_rk4_chunk_uses_one_packed_baddbmm_per_stage(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Each RK stage in the chunk evaluates packed K/C with one column vector."""
    batch_size, state_size = 3, 2
    y = torch.randn(batch_size, state_size, 1)
    v = torch.randn(batch_size, state_size, 1)
    matrix = torch.eye(state_size).expand(batch_size, -1, -1)
    packed_matrix = torch.cat((matrix, matrix), dim=-1)
    bias = torch.zeros(batch_size, state_size)
    h = torch.tensor([0.1, 0.1, 0.1, 0.0])
    step_factors = torch.stack((h, h / 2, h.square() / 4,
                                h.square() / 2, h.square() / 6, h / 6))
    original_baddbmm = torch.baddbmm
    baddbmm_calls = 0

    def check_baddbmm(bias: torch.Tensor, left: torch.Tensor,
                      right: torch.Tensor) -> torch.Tensor:
        """Verify the packed operand layout without changing the operation."""
        nonlocal baddbmm_calls
        assert bias.shape == (batch_size, state_size, 1)
        assert left.shape == (batch_size, state_size, 2 * state_size)
        assert right.shape == (batch_size, 2 * state_size, 1)
        assert right.stride() == (2 * state_size, 1, 1)
        baddbmm_calls += 1
        return original_baddbmm(bias, left, right)

    monkeypatch.setattr(torch, "baddbmm", check_baddbmm)
    n_steps = second_order_solvers._RK4_AFFINE_CHUNK_SIZE
    KC_chunk = packed_matrix.expand(n_steps, 3, -1, -1, -1)
    b_chunk = bias.expand(n_steps, 3, -1, -1)
    factors_chunk = step_factors.expand(n_steps, -1, -1)
    D, V = second_order_solvers._rk4_affine_chunk(y, v, KC_chunk, b_chunk, factors_chunk)
    assert baddbmm_calls == 4 * n_steps
    assert D.shape == V.shape == (n_steps, batch_size, state_size, 1)


def test_affine_rk4_chunk_matches_classical_rk4_and_gradients() -> None:
    """The chunk matches independent first-order RK4 and freezes zero-step padding."""
    torch.manual_seed(19)
    batch_size, state_size = 3, 2
    n_steps = second_order_solvers._RK4_AFFINE_CHUNK_SIZE
    values = (
        torch.randn(batch_size, state_size, 1, dtype=torch.float64),
        torch.randn(batch_size, state_size, 1, dtype=torch.float64),
        0.1 * torch.randn(n_steps, 3, batch_size, state_size, 2 * state_size,
                          dtype=torch.float64),
        0.1 * torch.randn(n_steps, 3, batch_size, state_size, dtype=torch.float64),
        torch.full((n_steps, 4), 0.02, dtype=torch.float64),
    )
    values[-1][5:] = 0.0
    values[-1][:, batch_size:] = 0.0

    def solve(use_chunk: bool) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Evaluate the chunk or classical RK4 on independent differentiable inputs."""
        y0, v0, KC, b, h = (value.detach().clone().requires_grad_() for value in values)
        factors = torch.stack((h, h / 2, h.square() / 4, h.square() / 2,
                               h.square() / 6, h / 6), dim=1)
        if use_chunk:
            D, V = second_order_solvers._rk4_affine_chunk(y0, v0, KC, b, factors)
        else:
            D_list: list[torch.Tensor] = []
            V_list: list[torch.Tensor] = []
            y, v = y0, v0
            for n in range(n_steps):
                h_n = h[n, :batch_size, None, None]
                l1 = torch.baddbmm(b[n, 0].unsqueeze(-1), KC[n, 0], torch.cat((y, v), dim=1))
                y2, v2 = y + h_n / 2 * v, v + h_n / 2 * l1
                l2 = torch.baddbmm(b[n, 1].unsqueeze(-1), KC[n, 1], torch.cat((y2, v2), dim=1))
                y3, v3 = y + h_n / 2 * v2, v + h_n / 2 * l2
                l3 = torch.baddbmm(b[n, 1].unsqueeze(-1), KC[n, 1], torch.cat((y3, v3), dim=1))
                y4, v4 = y + h_n * v3, v + h_n * l3
                l4 = torch.baddbmm(b[n, 2].unsqueeze(-1), KC[n, 2], torch.cat((y4, v4), dim=1))
                y = y + h_n / 6 * (v + 2 * v2 + 2 * v3 + v4)
                v = v + h_n / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
                D_list.append(y)
                V_list.append(v)
            D = torch.stack(D_list, dim=0)
            V = torch.stack(V_list, dim=0)
        loss = D[:5].square().sum() + 0.37 * V[:5].square().sum()
        gradients = torch.autograd.grad(loss, (y0, v0, KC, b, h))
        return D, V, gradients

    chunk_outputs = solve(True)
    eager_outputs = solve(False)
    for actual, expected in zip(chunk_outputs[:2], eager_outputs[:2]):
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
        assert torch.equal(actual[5:], actual[4].expand_as(actual[5:]))
    for actual, expected in zip(chunk_outputs[2], eager_outputs[2]):
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("n_steps", [8, 11])
def test_affine_batched_rk4_compiled_chunk_matches_eager(
        n_steps: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full and padded CUDA chunks preserve trajectories and gradients at the profiled B/d."""
    torch.manual_seed(21)
    device = "cuda"
    dtype = torch.float32
    batch_size, state_size = 16, 20
    grid = (0.03 * torch.arange(n_steps + 1, device=device, dtype=dtype)[:, None]).expand(
        -1, batch_size).contiguous()
    grid[n_steps - 2:, 1] = grid[n_steps - 3, 1]
    lengths = torch.full((batch_size,), n_steps + 1, dtype=torch.long, device=device)
    lengths[1] = n_steps - 2
    values = (
        torch.randn(batch_size, state_size, device=device, dtype=dtype),
        torch.randn(batch_size, state_size, device=device, dtype=dtype),
        0.1 * torch.randn(n_steps, 3, batch_size, state_size, state_size, device=device, dtype=dtype),
        0.1 * torch.randn(n_steps, 3, batch_size, state_size, state_size, device=device, dtype=dtype),
        0.1 * torch.randn(n_steps, 3, batch_size, state_size, device=device, dtype=dtype),
    )

    def solve() -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
        """Run one solve with fresh leaves so both backends have independent graphs."""
        y0, v0, K, C, b = (value.detach().clone().requires_grad_() for value in values)
        y, v = RK4_affine_batched(y0, v0, grid, lengths, K, C, b)
        gradients = torch.autograd.grad(y.square().sum() + v.square().sum(),
                                        (y0, v0, K, C, b))
        return (y, v), gradients

    compiled_outputs, compiled_gradients = solve()
    monkeypatch.setattr(second_order_solvers, "_compiled_rk4_affine_chunk",
                        second_order_solvers._rk4_affine_chunk)
    eager_outputs, eager_gradients = solve()

    for actual, expected in zip(compiled_outputs + compiled_gradients,
                                eager_outputs + eager_gradients):
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
    for trajectory in compiled_outputs:
        assert torch.equal(trajectory[n_steps - 2:, 1],
                           trajectory[n_steps - 3, 1].expand_as(trajectory[n_steps - 2:, 1]))


@pytest.mark.parametrize("batch_size", [3, 16])
def test_affine_batched_rk4_chunk_factors_are_aligned(
        batch_size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every chunk receives contiguous factors with aligned step rows."""
    grid = torch.tensor([[0.0, 0.1, 0.3, 0.6, 1.0]]).expand(batch_size, -1).T.contiguous()
    lengths = torch.full((batch_size,), 5, dtype=torch.long)
    lengths[-1] = 3
    n_steps = grid.shape[0] - 1
    coefficients = torch.zeros(n_steps, 3, batch_size, 2, 2)
    biases = torch.zeros(n_steps, 3, batch_size, 2)
    initial = torch.zeros(batch_size, 2)
    original_chunk = second_order_solvers._rk4_affine_chunk
    checked_chunks = 0

    def check_chunk(y: torch.Tensor, v: torch.Tensor, KC: torch.Tensor,
                    b: torch.Tensor, factors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inspect the packed factors before running the eager RK4 chunk."""
        nonlocal checked_chunks
        padded_size = batch_size + (-batch_size) % 4
        assert factors.shape == (second_order_solvers._RK4_AFFINE_CHUNK_SIZE, 6, padded_size)
        assert factors.is_contiguous()
        for n in range(n_steps):
            h = grid[n + 1] - grid[n]
            h = torch.where(n < lengths - 1, h, 0)
            expected = (h, h / 2, h.square() / 4, h.square() / 2,
                        h.square() / 6, h / 6)
            for actual, reference in zip(factors[n].unbind(dim=0), expected):
                assert actual.data_ptr() % 16 == 0
                torch.testing.assert_close(actual[:batch_size], reference, rtol=0, atol=0)
                assert torch.count_nonzero(actual[batch_size:]) == 0
        assert torch.count_nonzero(factors[n_steps:]) == 0
        checked_chunks += 1
        return original_chunk(y, v, KC, b, factors)

    monkeypatch.setattr(second_order_solvers, "_rk4_affine_chunk", check_chunk)
    RK4_affine_batched(initial, initial, grid, lengths, coefficients, coefficients, biases)
    assert checked_chunks == 1


@pytest.mark.parametrize("batch_size", [3, 16])
def test_affine_batched_rk4_stage_strides_do_not_depend_on_duration(
        batch_size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk coefficient inputs have the same contiguous layout for different T."""
    state_size = 2
    original_chunk = second_order_solvers._rk4_affine_chunk
    checked_chunks = 0

    def check_chunk(y: torch.Tensor, v: torch.Tensor, KC: torch.Tensor,
                    b: torch.Tensor, factors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Check all coefficient-stage views before evaluating the eager chunk."""
        nonlocal checked_chunks
        for KC_n, b_n in zip(KC.unbind(dim=0), b.unbind(dim=0)):
            for matrix in KC_n.unbind(dim=0):
                assert matrix.is_contiguous()
                assert matrix.stride() == (2 * state_size * state_size, 2 * state_size, 1)
            for bias in b_n.unbind(dim=0):
                assert bias.is_contiguous()
                assert bias.stride() == (state_size, 1)
        checked_chunks += 1
        return original_chunk(y, v, KC, b, factors)

    monkeypatch.setattr(second_order_solvers, "_rk4_affine_chunk", check_chunk)
    for n_steps in (2, 8, 11):
        grid = torch.arange(n_steps + 1, dtype=torch.float32)[:, None].expand(-1, batch_size)
        lengths = torch.full((batch_size,), n_steps + 1, dtype=torch.long)
        matrices = torch.zeros(n_steps, 3, batch_size, state_size, state_size)
        biases = torch.zeros(n_steps, 3, batch_size, state_size)
        initial = torch.zeros(batch_size, state_size)
        RK4_affine_batched(initial, initial, grid, lengths, matrices, matrices, biases)

    assert checked_chunks == 4


def test_affine_batched_rk4_backward_chunk_strides_do_not_depend_on_duration(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-window losses send duration-independent gradients to RK4 chunks."""
    batch_size, state_size = 3, 2
    original_chunk = second_order_solvers._rk4_affine_chunk
    observed_strides: list[tuple[int, ...]] = []

    def check_chunk(y: torch.Tensor, v: torch.Tensor, KC: torch.Tensor,
                    b: torch.Tensor, factors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Record the gradient layouts received by each eager chunk output."""
        D, V = original_chunk(y, v, KC, b, factors)
        D.register_hook(lambda grad: observed_strides.append(grad.stride()))
        V.register_hook(lambda grad: observed_strides.append(grad.stride()))
        return D, V

    monkeypatch.setattr(second_order_solvers, "_rk4_affine_chunk", check_chunk)
    for n_steps in (2, 8, 11):
        grid = torch.arange(n_steps + 1, dtype=torch.float32)[:, None].expand(-1, batch_size)
        lengths = torch.tensor([n_steps + 1, n_steps, n_steps - 1])
        matrices = torch.zeros(n_steps, 3, batch_size, state_size, state_size)
        biases = torch.zeros(n_steps, 3, batch_size, state_size)
        y0 = torch.ones(batch_size, state_size, requires_grad=True)
        v0 = torch.ones(batch_size, state_size, requires_grad=True)
        y, v = RK4_affine_batched(y0, v0, grid, lengths, matrices, matrices, biases)
        windows = [torch.cat((y[:length, i], v[:length, i]), dim=1)
                   for i, length in enumerate(lengths.tolist())]
        torch.cat(windows, dim=0).square().sum().backward()

    assert len(observed_strides) == 8
    assert set(observed_strides) == {(batch_size * state_size, state_size, 1, 1)}


def test_affine_batched_rk4_rejects_noncontiguous_coefficients() -> None:
    """A time-major shape alone cannot guarantee stable per-step coefficient strides."""
    batch_size, state_size, n_steps = 2, 2, 3
    grid = torch.zeros(n_steps + 1, batch_size)
    lengths = torch.full((batch_size,), n_steps + 1, dtype=torch.long)
    initial = torch.zeros(batch_size, state_size)
    matrices = torch.zeros(batch_size, n_steps, 3, state_size, state_size)
    noncontiguous = matrices.permute(1, 2, 0, 3, 4)
    biases = torch.zeros(n_steps, 3, batch_size, state_size)

    with pytest.raises(ValueError, match="contiguous time-major"):
        RK4_affine_batched(initial, initial, grid, lengths,
                           noncontiguous, noncontiguous, biases)


def test_affine_batched_rk4_rejects_invalid_ranks_with_value_error() -> None:
    """Shape validation must run before indexing a missing tensor dimension."""
    initial = torch.zeros(1, 1)
    grid = torch.zeros(2, 1)
    lengths = torch.ones(1, dtype=torch.long)
    matrices = torch.zeros(1, 3, 1, 1, 1)
    biases = torch.zeros(1, 3, 1, 1)

    with pytest.raises(ValueError, match="y0 and Dy0 must have matching shape"):
        RK4_affine_batched(initial[0], initial[0], grid, lengths,
                           matrices, matrices, biases)
    with pytest.raises(ValueError, match="t_grid must have shape"):
        RK4_affine_batched(initial, initial, grid[0, 0], lengths,
                           matrices, matrices, biases)


def test_some_batched_single_frame_returns_initial_state() -> None:
    """A one-frame window needs no gate evaluation or RK stages."""
    model = SOME(n_z=1, Uniform_t_Grid=True, n_p=1, config=_some_config(False, True))
    y0 = torch.tensor([[0.4], [-0.3]], requires_grad=True)
    v0 = torch.tensor([[0.2], [0.1]], requires_grad=True)
    grid = torch.tensor([[0.1, 0.5]])
    lengths = torch.ones(2, dtype=torch.long)
    params = torch.tensor([[0.2], [0.8]])
    reference_t0 = torch.zeros(2)
    reference_t_span = torch.ones(2)

    K, C, b = model._precompute_affine_stage_coefficients(
        grid, lengths, params, reference_t0, reference_t_span, state_dtype=y0.dtype)
    assert K.shape == C.shape == (0, 3, 2, 1, 1)
    assert b.shape == (0, 3, 2, 1)
    y, v = RK4_affine_batched(y0, v0, grid, lengths, K, C, b)
    assert torch.equal(y[0], y0)
    assert torch.equal(v[0], v0)
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

    K, C, b = batched._precompute_affine_stage_coefficients(
        grid.T.contiguous(), lengths, params, reference_t0, reference_t_span, state_dtype=y0.dtype)
    assert K.shape == C.shape == (3, 3, 2, 1, 1)
    assert b.shape == (3, 3, 2, 1)
    # Window 1's last interval is padding: all three stages use its final valid time.
    for coefficients in (K, C, b):
        padded_stages = coefficients[2, :, 1]
        torch.testing.assert_close(
            padded_stages, padded_stages[0].expand_as(padded_stages), rtol=0, atol=0)
    terminal_weights_D, terminal_weights_V = batched._weights(
        t_Grid=grid[1, 2:3], params=params[1],
        t0=reference_t0[1], t_span=reference_t_span[1])
    terminal_coefficients = batched._effective_coefficients(
        terminal_weights_D, terminal_weights_V, grid.device, y0.dtype)
    for padded, terminal in zip((K, C, b), terminal_coefficients):
        torch.testing.assert_close(padded[2, 0, 1], terminal[0], rtol=1.0e-6, atol=1.0e-7)
    assert K.requires_grad and C.requires_grad
    assert b.requires_grad == use_biases
    if not use_biases:
        assert torch.count_nonzero(b) == 0
    y_batch, v_batch = RK4_affine_batched(y0, v0, grid.T.contiguous(), lengths, K, C, b)
    valid = (torch.arange(4)[:, None] < lengths[None, :])[..., None]
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
        torch.testing.assert_close(y_batch[:length, row], y_ref, rtol=3.0e-5, atol=3.0e-6)
        torch.testing.assert_close(v_batch[:length, row], v_ref, rtol=3.0e-5, atol=3.0e-6)
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
    params = torch.tensor([[0.2], [0.7]], device="cuda")

    predictions = model.simulate(
        IC=[[y0[i], v0[i]] for i in range(2)],
        t_Grid=[grid[0], grid[1, :2]],
        params=params.cpu().numpy(),
        time_domain=[torch.tensor([0.0, 1.0], device="cuda") for _ in range(2)])
    assert all(state.device.type == "cuda" for pair in predictions for state in pair)
    sum(state.sum() for pair in predictions for state in pair).backward()
    assert y0.grad is not None and torch.isfinite(y0.grad).all()
    assert v0.grad is not None and torch.isfinite(v0.grad).all()
    assert model.unmasked_K.grad is not None and torch.isfinite(model.unmasked_K.grad).all()
    assert model.unmasked_C.grad is not None and torch.isfinite(model.unmasked_C.grad).all()
    assert any(param.grad is not None for param in model.w.parameters())


@pytest.mark.parametrize("weak", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_some_public_simulate_preserves_variable_windows_and_dispatches_cuda(
        weak: bool, device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """One public call returns valid frames; only CUDA takes the padded affine path."""
    torch.manual_seed(11)
    model_type = SOME_weak if weak else SOME
    config = _some_config(weak, True)
    model = model_type(n_z=1, Uniform_t_Grid=False, n_p=1, config=config)
    scalar = model_type(n_z=1, Uniform_t_Grid=False, n_p=1, config=config)
    scalar.load(model.export())
    model.move_parameters_to_device(device)

    grids = [torch.tensor([0.1, 0.25, 0.7], device=device),
             torch.tensor([0.4, 0.65], device=device),
             torch.tensor([0.8], device=device)]
    domains = [torch.tensor([0.0, 1.0], device=device) for _ in grids]
    params = numpy.array([[0.2], [0.7], [0.2]], dtype=numpy.float32)
    y_values = torch.tensor([[0.3], [-0.2], [0.5]])
    v_values = torch.tensor([[0.1], [0.4], [-0.1]])
    y0 = y_values.to(device).requires_grad_()
    v0 = v_values.to(device).requires_grad_()
    calls = []
    original = model._precompute_affine_stage_coefficients

    def counted(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_precompute_affine_stage_coefficients", counted)
    predictions = model.simulate(
        IC=[[y0[i], v0[i]] for i in range(3)],
        t_Grid=grids, params=params, time_domain=domains)
    assert len(calls) == (1 if device == "cuda" else 0)
    assert [pair[0].shape[0] for pair in predictions] == [3, 2, 1]
    assert all(pair[0].device.type == device for pair in predictions)

    ys = y_values.clone().requires_grad_()
    vs = v_values.clone().requires_grad_()
    expected = scalar.simulate(
        IC=[[ys[i], vs[i]] for i in range(3)],
        t_Grid=[grid.cpu().numpy() for grid in grids],
        params=params, time_domain=[domain.cpu().numpy() for domain in domains])
    for actual_pair, expected_pair in zip(predictions, expected):
        for actual, reference in zip(actual_pair, expected_pair):
            torch.testing.assert_close(actual.cpu(), reference, rtol=3.0e-5, atol=3.0e-6)

    loss = sum(tensor.square().sum() for pair in predictions for tensor in pair)
    reference_loss = sum(tensor.square().sum() for pair in expected for tensor in pair)
    gradients = torch.autograd.grad(loss, (y0, v0, *model.parameters()))
    reference_gradients = torch.autograd.grad(reference_loss, (ys, vs, *scalar.parameters()))
    for actual, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual.cpu(), reference, rtol=5.0e-4, atol=5.0e-6)

    if device == "cuda":
        # Moving the same model back to CPU must select readable scalar RK4.
        model.move_parameters_to_device("cpu")
        calls.clear()
        cpu_predictions = model.simulate(
            IC=[[y_values[i], v_values[i]] for i in range(3)],
            t_Grid=[grid.cpu() for grid in grids],
            params=params, time_domain=[domain.cpu() for domain in domains])
        assert calls == []
        for actual_pair, expected_pair in zip(cpu_predictions, expected):
            for actual, reference in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual, reference, rtol=3.0e-5, atol=3.0e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("domain_backend", ["numpy", "tensor"])
@pytest.mark.parametrize("domain_values", [[0.0, 0.0], [1.0, 0.0], [0.0, float("nan")]])
def test_some_simulate_rejects_nonpositive_reference_span(
        device: str, domain_backend: str, domain_values: list[float]) -> None:
    """Both scalar and CUDA paths reject zero, reversed, and NaN reference spans."""
    model = SOME(n_z=1, Uniform_t_Grid=False, n_p=1, config=_some_config(False, True))
    model.move_parameters_to_device(device)
    state = torch.tensor([0.1], device=device)
    grid = torch.tensor([0.0, 0.2], device=device)
    domain = (numpy.array(domain_values, dtype=numpy.float32) if domain_backend == "numpy"
              else torch.tensor(domain_values, device=device))
    with pytest.raises(ValueError, match="positive span"):
        model.simulate(
            IC=[[state, state]], t_Grid=[grid],
            params=numpy.array([[0.2]], dtype=numpy.float32),
            time_domain=[domain])


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_some_simulate_rejects_empty_rollout_grid(device: str) -> None:
    """An empty grid fails validation before either RK4 implementation is called."""
    model = SOME(n_z=1, Uniform_t_Grid=False, n_p=1, config=_some_config(False, True))
    model.move_parameters_to_device(device)
    state = torch.tensor([0.1], device=device)
    with pytest.raises(ValueError, match="nonempty"):
        model.simulate(
            IC=[[state, state]], t_Grid=[torch.empty(0, device=device)],
            params=numpy.array([[0.2]], dtype=numpy.float32),
            time_domain=[torch.tensor([0.0, 1.0], device=device)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_some_simulate_rejects_cuda_time_dtype_mismatch() -> None:
    """CUDA time grids and reference domains must match the latent-state dtype."""
    model = SOME(n_z=1, Uniform_t_Grid=False, n_p=1, config=_some_config(False, True))
    model.move_parameters_to_device("cuda")
    state = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    params = numpy.array([[0.2]], dtype=numpy.float32)
    for grid_dtype, domain_dtype in ((torch.float64, torch.float32),
                                     (torch.float32, torch.float64)):
        with pytest.raises(ValueError, match="latent-state dtype"):
            model.simulate(
                IC=[[state, state]],
                t_Grid=[torch.tensor([0.0, 0.2], dtype=grid_dtype, device="cuda")],
                params=params,
                time_domain=[torch.tensor([0.0, 1.0], dtype=domain_dtype, device="cuda")])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("cuda_first", [False, True])
def test_some_simulate_rejects_mixed_initial_state_devices(cuda_first: bool) -> None:
    """A mixed batch must not silently fall back to per-window scalar RK4."""
    model = SOME(n_z=1, Uniform_t_Grid=False, n_p=1, config=_some_config(False, True))
    cpu_state = torch.tensor([0.1])
    cuda_state = torch.tensor([0.2], device="cuda")
    states = [cuda_state, cpu_state] if cuda_first else [cpu_state, cuda_state]
    with pytest.raises(ValueError, match="cannot be mixed"):
        model.simulate(
            IC=[[state, state] for state in states],
            t_Grid=[numpy.array([0.0, 0.2], dtype=numpy.float32)] * 2,
            params=numpy.array([[0.2], [0.7]], dtype=numpy.float32),
            time_domain=[numpy.array([0.0, 1.0], dtype=numpy.float32)] * 2)
