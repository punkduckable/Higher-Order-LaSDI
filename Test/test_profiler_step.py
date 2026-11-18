"""Focused tests for profiling one global training epoch across rounds."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Trainer.Trainer import Trainer


class MinimalTrainer(Trainer):
    """Exercise the base training loop without model or physics dependencies."""

    def Iterate(
        self,
        start_iter: int,
        end_iter: int,
        profiler: torch.profiler.profile | None = None,
    ) -> None:
        """Emit distinct profiler ranges for round setup and each training epoch."""

        self.rounds.append((start_iter, end_iter, profiler is not None))
        with self._profile_region("round_setup"):
            torch.ones(1) + 1
        if profiler is not None:
            profiler.step()

        for iteration in range(start_iter, end_iter):
            with self._profile_region(f"train_step_{iteration + 1}"):
                torch.ones(1) + 1
            if profiler is not None:
                profiler.step()

    def Load_Checkpoint(self) -> tuple[object, int]:
        """Return a stand-in checkpoint for the base loop to restore."""

        return self.encoder_decoder, self.restart_iter


@pytest.mark.parametrize("profiler_epoch", [None, 1, 2, 3, 4])
@pytest.mark.filterwarnings("ignore:Profiler won't be using warmup")
def test_train_profiles_only_selected_global_step(
    profiler_epoch: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the requested epoch is captured, even at the start of a later round."""

    traces: list[list[str]] = []

    def on_trace_ready(profiler: torch.profiler.profile) -> None:
        """Collect custom setup and epoch ranges from the completed trace."""

        traces.append([
            event.name for event in profiler.events()
            if event.name == "round_setup" or event.name.startswith("train_step_")
        ])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.profiler,
        "tensorboard_trace_handler",
        lambda **kwargs: on_trace_ready,
    )

    trainer = MinimalTrainer.__new__(MinimalTrainer)
    trainer.U_Train = [[]]
    trainer.param_space = SimpleNamespace(n_train=lambda: 1)
    trainer.noise_ratio = 0.0
    trainer.restart_iter = 0
    trainer.n_iter = 2
    trainer.max_iter = 4
    trainer.config = SimpleNamespace(profiler_epoch=profiler_epoch)
    trainer.metrics_path = str(tmp_path / "metrics.jsonl")
    trainer.results_dir = str(tmp_path)
    trainer.encoder_decoder = object()
    trainer.rounds = []

    trainer.train()
    trainer.train()

    assert trainer.restart_iter == 4
    assert trainer.rounds == [
        (0, 2, profiler_epoch in (1, 2)),
        (2, 4, profiler_epoch in (3, 4)),
    ]
    assert traces == ([] if profiler_epoch is None else [[f"train_step_{profiler_epoch}"]])
    summary_path = tmp_path / "HLaSDI_train_profile.txt"
    assert summary_path.exists() == (profiler_epoch is not None)
    if profiler_epoch is not None:
        summary = summary_path.read_text()
        assert f"train_step_{profiler_epoch}" in summary
        assert "round_setup" not in summary
        assert all(
            f"train_step_{step}" not in summary
            for step in range(1, 5) if step != profiler_epoch
        )
