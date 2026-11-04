import os
import sys

import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Trainer.Trainer import Trainer


def _normalization_trainer(means: list[float], stds: list[float]) -> Trainer:
    """Construct the normalization-only portion of a Trainer for focused unit tests."""

    trainer = Trainer.__new__(Trainer)
    trainer.normalization_enabled = True
    trainer.data_mean = [torch.tensor(value, dtype=torch.float32) for value in means]
    trainer.data_std = [torch.tensor(value, dtype=torch.float32) for value in stds]
    return trainer


def test_second_order_normalization_uses_displacement_scale_without_velocity_centering():
    time = torch.arange(4, dtype=torch.float32)
    displacement = (5.0 + 3.0*time).reshape(-1, 1)
    velocity = torch.full_like(displacement, 3.0)
    data = [[displacement, velocity]]

    trainer = Trainer.__new__(Trainer)
    means, stds = Trainer._compute_mean_std_from_U(trainer, data)

    expected_scale = float(displacement.double().std(unbiased=False).item())
    assert means == pytest.approx([float(displacement.mean().item()), 0.0])
    assert stds == pytest.approx([expected_scale, expected_scale])

    trainer = _normalization_trainer(means, stds)
    normalized_displacement = trainer.normalize(displacement, 0)
    normalized_velocity = trainer.normalize(velocity, 1)

    normalized_displacement_derivative = normalized_displacement[1:] - normalized_displacement[:-1]
    assert torch.allclose(normalized_displacement_derivative, normalized_velocity[1:])
    assert torch.allclose(trainer.denormalize(normalized_displacement, 0), displacement)
    assert torch.allclose(trainer.denormalize(normalized_velocity, 1), velocity)


def test_first_order_normalization_retains_standard_centering_and_scaling():
    state = torch.tensor([[1.0], [2.0], [6.0]], dtype=torch.float32)
    trainer = Trainer.__new__(Trainer)

    means, stds = Trainer._compute_mean_std_from_U(trainer, [[state]])

    assert means == pytest.approx([float(state.mean().item())])
    assert stds == pytest.approx([float(state.double().std(unbiased=False).item())])


def test_jvp_create_graph_propagates_gradients_to_module_parameters():
    layer = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(2.0)

    inputs = torch.tensor([[1.0], [3.0]])
    tangent = torch.tensor([[4.0], [5.0]])
    jvp = torch.autograd.functional.jvp(
        func=layer,
        inputs=inputs,
        v=tangent,
        create_graph=True,
    )[1]
    loss = torch.sum(jvp**2)
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.item() == pytest.approx(4.0*(4.0**2 + 5.0**2))
