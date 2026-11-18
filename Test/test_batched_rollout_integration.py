"""Invariants of the shared second-order sampled-rollout loss path."""

import contextlib
import os
import sys
from types import SimpleNamespace
from typing import ContextManager

import numpy
import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Trainer.Second_Order_Rollout import Second_Order_Rollout
from HLaSDI.Trainer.Second_Order_Weak import Second_Order_Weak


class _Dynamics:
    """Record calls while producing differentiable, time-dependent trajectories."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def simulate(
            self,
            IC: list[list[torch.Tensor]],
            t_Grid: list[numpy.ndarray | torch.Tensor],
            params: numpy.ndarray,
            time_domain: list[numpy.ndarray | torch.Tensor]) -> list[list[torch.Tensor]]:
        """Return one displacement/velocity pair per supplied window."""
        self.calls.append(dict(IC=IC, t_Grid=t_Grid, params=params, time_domain=time_domain))
        predictions: list[list[torch.Tensor]] = []
        for states, grid, parameter in zip(IC, t_Grid, params):
            t = torch.as_tensor(grid, dtype=states[0].dtype, device=states[0].device)
            elapsed = (t - t[0])[:, None]
            predictions.append([
                states[0][None, :] + 0.5 * elapsed + float(parameter[0]) * 0.1,
                states[1][None, :] + elapsed.square() - float(parameter[0]) * 0.2,
            ])
        return predictions


class _Decoder:
    """Decode latent vectors into frames with one extra spatial axis."""

    def Decode(
            self, displacement: torch.Tensor,
            velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply two distinct affine maps to the valid predicted frames."""
        return (2.0 * displacement + 0.3).unsqueeze(-1), (-3.0 * velocity).unsqueeze(-1)


class _TrainerHarness:
    """Provide only the fields needed by the shared rollout helper."""

    def __init__(self, loss_type: str) -> None:
        self.latent_dynamics = _Dynamics()
        self.param_space = SimpleNamespace(
            train_space=numpy.array([[0.2], [0.7]], dtype=numpy.float32))
        self.n_rollouts = 2
        self.loss_types = {"rollout": loss_type}
        self.metrics: dict[str, float] = {}

    def _profile_region(self, name: str) -> ContextManager[None]:
        """Disable profiling for this focused CPU unit test."""
        return contextlib.nullcontext()

    def _cache_metric(self, key: str, value: float | int) -> None:
        """Retain packing metrics for explicit assertions."""
        self.metrics[key] = float(value)


@pytest.mark.parametrize("trainer_type", [Second_Order_Rollout, Second_Order_Weak])
@pytest.mark.parametrize("loss_type", ["MSE", "MAE"])
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_rollout_loss_batches_once_and_preserves_window_parameter_weighting(
        trainer_type: type[Second_Order_Rollout], loss_type: str, device: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Unequal window lengths/counts must not turn into a global frame mean."""
    def fixed_choice(rollable: numpy.ndarray, size: int, replace: bool) -> numpy.ndarray:
        """Choose the first and last available starts without random-state dependence."""
        assert not replace
        return numpy.array([rollable[0]] if size == 1 else [rollable[0], rollable[-1]])

    monkeypatch.setattr(numpy.random, "choice", fixed_choice)
    harness = _TrainerHarness(loss_type)
    grids = [torch.tensor([0.0, 0.1, 0.4, 1.0], device=device),
             torch.tensor([0.0, 0.7, 1.0], device=device)]
    latent_states = [
        [torch.tensor([[0.1, 0.2], [0.4, 0.3], [0.2, 0.6], [0.8, 0.1]], device=device, requires_grad=True),
         torch.tensor([[0.3, 0.5], [0.6, 0.1], [0.2, 0.4], [0.7, 0.9]], device=device, requires_grad=True)],
        [torch.tensor([[0.9, 0.2], [0.1, 0.5], [0.4, 0.7]], device=device, requires_grad=True),
         torch.tensor([[0.4, 0.3], [0.6, 0.2], [0.8, 0.1]], device=device, requires_grad=True)],
    ]
    FOM_targets = [
        [(1.2 * states[0].detach()).unsqueeze(-1),
         (0.8 * states[1].detach()).unsqueeze(-1)] for states in latent_states
    ]
    actual = trainer_type._sampled_rollout_losses(
        harness, latent_states, FOM_targets, grids,
        [grid.cpu().numpy() for grid in grids], _Decoder(), 0.4, device)

    assert len(harness.latent_dynamics.calls) == 1
    call = harness.latent_dynamics.calls[0]
    assert [len(grid) for grid in call["t_Grid"]] == [3, 2, 2]
    assert harness.metrics["rollout/batch_size"] == 3
    assert harness.metrics["rollout/padding_efficiency"] == pytest.approx(7 / 9)

    # Independent nested-loop reduction: mean over each whole window, mean
    # those windows within a parameter, then sum the two parameter losses.
    expected_per_parameter: list[list[list[torch.Tensor]]] = [[], []]
    predictions = harness.latent_dynamics.simulate(**call)
    for grid, parameter, prediction in zip(call["t_Grid"], call["params"], predictions):
        parameter_index = 0 if parameter[0] == harness.param_space.train_space[0, 0] else 1
        grid_values = numpy.asarray(grid) if isinstance(grid, numpy.ndarray) else grid.cpu().numpy()
        start = int(numpy.where(grids[parameter_index].cpu().numpy() == grid_values[0])[0][0])
        end = start + len(grid)
        Z_D, Z_V = prediction
        D, V = _Decoder().Decode(Z_D, Z_V)
        targets = (
            latent_states[parameter_index][0][start:end],
            latent_states[parameter_index][1][start:end],
            FOM_targets[parameter_index][0][start:end],
            FOM_targets[parameter_index][1][start:end],
        )
        differences = (targets[0] - Z_D, targets[1] - Z_V, D - targets[2], V - targets[3])
        if loss_type == "MSE":
            window_losses = [difference.square().mean() for difference in differences]
        else:
            window_losses = [difference.abs().mean() for difference in differences]
        expected_per_parameter[parameter_index].append(window_losses)
    expected = [
        sum(sum(windows[j] for windows in group) / len(group)
            for group in expected_per_parameter) for j in range(4)
    ]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1.0e-6, atol=1.0e-7)
    sum(actual).backward()
    assert all(state.grad is not None and torch.isfinite(state.grad).all()
               for pair in latent_states for state in pair)
