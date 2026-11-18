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


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_affine_batched_rk4_dense_time_dependent_coefficients_match_scalar_reference(
        device: str) -> None:
    """Non-symmetric stage matrices and their gradients match independent scalar RK4."""
    dtype = torch.float64
    grid = torch.tensor([[0.1, 0.25, 0.5, 0.9], [0.0, 0.35, 0.8, 0.8]], dtype=dtype, device=device)
    lengths = torch.tensor([4, 3], device=device)
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
    y_batch, v_batch = RK4_affine_batched(y0, v0, grid, lengths, K, C, b)

    valid = (torch.arange(grid.shape[1], device=device)[None, :] < lengths[:, None])[..., None]
    loss_batch = ((y_batch.square() + 0.37 * v_batch.square()) * valid).sum()
    grads_batch = torch.autograd.grad(loss_batch, (y0, v0, K0, K1, C0, C1, b0, b1))

    ys, vs, Ks0, Ks1, Cs0, Cs1, bs0, bs1 = (value.clone().requires_grad_() for value in values)
    loss_scalar = torch.zeros((), dtype=dtype, device=device)
    for row, length in enumerate(lengths.tolist()):
        def rhs(t: float, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return ((Ks0[row] + t * Ks1[row]) @ y
                    + (Cs0[row] + t * Cs1[row]) @ v + bs0[row] + t * bs1[row])

        y_ref, v_ref = RK4(rhs, ys[row], vs[row], grid[row, :length].cpu().numpy())
        torch.testing.assert_close(y_batch[row, :length], y_ref, rtol=1.0e-11, atol=1.0e-12)
        torch.testing.assert_close(v_batch[row, :length], v_ref, rtol=1.0e-11, atol=1.0e-12)
        loss_scalar = loss_scalar + y_ref.square().sum() + 0.37 * v_ref.square().sum()
    grads_scalar = torch.autograd.grad(loss_scalar, (ys, vs, Ks0, Ks1, Cs0, Cs1, bs0, bs1))
    for actual, expected in zip(grads_batch, grads_scalar):
        torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-12)


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

    K, C, b = model._precompute_affine_stage_coefficients(
        grid, lengths, params, reference_t0, reference_t_span, state_dtype=y0.dtype)
    assert K.shape == C.shape == (2, 0, 3, 1, 1)
    assert b.shape == (2, 0, 3, 1)
    y, v = RK4_affine_batched(y0, v0, grid, lengths, K, C, b)
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

    K, C, b = batched._precompute_affine_stage_coefficients(
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
    y_batch, v_batch = RK4_affine_batched(y0, v0, grid, lengths, K, C, b)
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
