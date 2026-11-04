import logging
import os
import sys

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Trainer.Trainer import Trainer


def _rollout_trainer(
    *,
    rollout_weight: float = 1.0,
    ic_rollout_weight: float = 1.0,
    warmup_epochs: int = 20,
) -> Trainer:
    """Build the minimal Trainer state required by the checkpoint-schedule helpers."""

    trainer = Trainer.__new__(Trainer)
    trainer.loss_weights = {
        "rollout": rollout_weight,
        "IC_rollout": ic_rollout_weight,
    }
    trainer.warmup_epochs = warmup_epochs

    trainer.p_rollout_init = 0.1
    trainer.max_p_rollout = 0.5
    trainer.dp_per_update = 0.1
    trainer.rollout_update_freq = 10

    trainer.p_IC_rollout_init = 0.2
    trainer.max_p_IC_rollout = 1.0
    trainer.IC_dp_per_update = 0.2
    trainer.IC_rollout_update_freq = 15
    return trainer


def test_checkpoint_schedule_waits_for_all_enabled_curricula():
    trainer = _rollout_trainer()

    schedule = trainer.prepare_checkpoint_schedule(start_iter=150, end_iter=250)

    assert schedule.warmup_ready_offset == 20
    assert schedule.rollout_ready_offset == 40
    assert schedule.IC_rollout_ready_offset == 60
    assert schedule.first_eligible_offset == 60
    assert schedule.n_eligible_epochs == 40
    assert schedule.allows_checkpoint(59) is False
    assert schedule.allows_checkpoint(60) is True
    assert schedule.allows_checkpoint(99) is True
    assert schedule.allows_checkpoint(100) is False


def test_disabled_curricula_do_not_delay_checkpointing():
    trainer = _rollout_trainer(
        rollout_weight=0.0,
        ic_rollout_weight=0.0,
        warmup_epochs=3,
    )

    schedule = trainer.prepare_checkpoint_schedule(start_iter=0, end_iter=10)

    assert schedule.rollout_ready_offset == 0
    assert schedule.IC_rollout_ready_offset == 0
    assert schedule.first_eligible_offset == 3
    assert schedule.n_eligible_epochs == 7


def test_curriculum_starting_at_maximum_does_not_delay_checkpointing():
    trainer = _rollout_trainer(ic_rollout_weight=0.0, warmup_epochs=0)
    trainer.p_rollout_init = trainer.max_p_rollout

    schedule = trainer.prepare_checkpoint_schedule(start_iter=0, end_iter=5)

    assert schedule.rollout_enabled is True
    assert schedule.rollout_ready_offset == 0
    assert schedule.first_eligible_offset == 0
    assert schedule.n_eligible_epochs == 5


def test_checkpoint_schedule_handles_exact_decimal_update_count():
    trainer = _rollout_trainer(ic_rollout_weight=0.0, warmup_epochs=0)
    trainer.p_rollout_init = 0.0
    trainer.max_p_rollout = 0.07
    trainer.dp_per_update = 0.005
    trainer.rollout_update_freq = 10

    # 14 updates are required, but 0.07 / 0.005 is represented as 14.000000000000002. The
    # schedule must not accidentally require a 15th update.
    schedule = trainer.prepare_checkpoint_schedule(start_iter=0, end_iter=141)

    assert schedule.rollout_ready_offset == 140
    assert schedule.first_eligible_offset == 140
    assert schedule.n_eligible_epochs == 1
    assert schedule.allows_checkpoint(139) is False
    assert schedule.allows_checkpoint(140) is True


def test_checkpoint_schedule_reports_round_and_global_epoch_ranges(caplog):
    trainer = _rollout_trainer()

    with caplog.at_level(logging.INFO, logger="HLaSDI.Trainer.Trainer"):
        schedule = trainer.prepare_checkpoint_schedule(start_iter=150, end_iter=250)

    assert schedule.first_eligible_offset == 60
    assert "Checkpoint curriculum schedule for this training round:" in caplog.text
    assert "Rollout reaches max 0.500000 at:      round epoch 41" in caplog.text
    assert "IC rollout reaches max 1.000000 at:   round epoch 61" in caplog.text
    assert "Checkpointing enabled for:            round epochs 61-100" in caplog.text
    assert "Corresponding global epochs:          211-250" in caplog.text
    assert "Number of checkpoint-eligible epochs: 40" in caplog.text


def test_checkpoint_schedule_rejects_round_without_eligible_epoch(caplog):
    trainer = _rollout_trainer(warmup_epochs=0)

    with caplog.at_level(logging.INFO, logger="HLaSDI.Trainer.Trainer"):
        with pytest.raises(ValueError, match="cannot become active in this 60-epoch round"):
            trainer.prepare_checkpoint_schedule(start_iter=0, end_iter=60)

    assert "Checkpointing enabled for:            none" in caplog.text
    assert "Number of checkpoint-eligible epochs: 0" in caplog.text
