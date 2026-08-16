import os
import sys
import json

import numpy
import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Trainer.Trainer import Trainer


class _LossCacheDummy:
    def __init__(self, loss_by_param_path):
        self._loss_cache = []
        self.loss_by_param_path = str(loss_by_param_path)


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
