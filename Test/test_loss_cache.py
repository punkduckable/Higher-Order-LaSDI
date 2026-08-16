import os
import sys
import json

import numpy
import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Trainer.Trainer import Trainer
from HLaSDI.LatentDynamics import LD_Loss_Container


class _LossCacheDummy:
    def __init__(self, loss_by_param_path):
        self._loss_cache = []
        self.loss_by_param_path = str(loss_by_param_path)

    def _cache_loss(self, *args, **kwargs):
        return Trainer._cache_loss(self, *args, **kwargs)


def test_cache_loss_writes_one_jsonl_row_and_preserves_keys(tmp_path):
    loss_path = tmp_path / "Dummy_loss_by_param.jsonl"
    dummy = _LossCacheDummy(loss_path)

    Trainer._cache_loss(dummy, "per_param", torch.tensor(2.0), (numpy.float64(0.25), numpy.int64(2)))
    Trainer._cache_loss(dummy, "total_loss", torch.tensor(3.0))

    flushed = Trainer._flush_loss_cache(dummy, 7)

    assert flushed[("per_param", (numpy.float64(0.25), numpy.int64(2)))] == 2.0
    assert flushed[("total_loss", "total")] == 3.0
    assert dummy._loss_cache == []

    rows = [json.loads(line) for line in loss_path.read_text().splitlines()]
    assert rows == [
        {
            "epoch": 7,
            "losses": [
                {"loss_name": "per_param", "param": [0.25, 2], "value": 2.0},
                {"loss_name": "total_loss", "param": None, "value": 3.0},
            ],
        }
    ]


def test_cache_loss_rejects_non_tuple_parameter_key():
    dummy = _LossCacheDummy("unused.jsonl")

    with pytest.raises(AssertionError, match="param_tuple must be a tuple or None"):
        Trainer._cache_loss(dummy, "bad_key", torch.tensor(1.0), [])


def test_flush_loss_cache_rejects_nonfinite_losses(tmp_path):
    dummy = _LossCacheDummy(tmp_path / "Dummy_loss_by_param.jsonl")
    Trainer._cache_loss(dummy, "nan_loss", torch.tensor(float("nan")))

    with pytest.raises(AssertionError, match="cached loss tensors must be finite"):
        Trainer._flush_loss_cache(dummy, 1)


def test_ld_loss_container_accepts_scalar_and_per_parameter_losses():
    params = numpy.array([[0.25], [0.75]])
    losses = LD_Loss_Container(
        losses={
            "LD": [torch.tensor(1.0), torch.tensor(2.0)],
            "coef": torch.tensor([3.0]),
        },
        weights={"LD": 1.0, "coef": 0.5},
        params=params,
    )

    assert losses.params is params
    assert torch.allclose(losses.losses["coef"], torch.tensor([3.0]))


def test_ld_loss_container_rejects_wrong_per_parameter_length():
    with pytest.raises(ValueError, match="must have length 2"):
        LD_Loss_Container(
            losses={"LD": [torch.tensor(1.0)]},
            weights={"LD": 1.0},
            params=numpy.array([[0.25], [0.75]]),
        )


def test_process_latent_dynamics_losses_uses_container_losses_and_weights(tmp_path):
    dummy = _LossCacheDummy(tmp_path / "Dummy_loss_by_param.jsonl")
    raw = LD_Loss_Container(
        losses={
            "LD": [torch.tensor(1.0), torch.tensor(2.0)],
            "coef": torch.tensor(4.0),
        },
        weights={"LD": 0.5, "coef": 0.0},
        params=numpy.array([[0.25], [0.75]]),
    )

    loss_dict, weighted_sum = Trainer._process_latent_dynamics_losses(
        dummy,
        raw_LD_Losses=raw,
        device=torch.device("cpu"),
    )

    assert set(loss_dict.keys()) == {"LD"}
    assert torch.allclose(loss_dict["LD"], torch.tensor(3.0))
    assert torch.allclose(weighted_sum, torch.tensor(1.5))
    assert [entry[0] for entry in dummy._loss_cache] == ["LD", "LD", "LD", "coef"]
