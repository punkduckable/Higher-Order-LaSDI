import os
import sys

import pytest
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from Trainer.Trainer import Trainer


class _LossCacheDummy:
    def __init__(self):
        self._loss_cache = []
        self.loss_by_param = {}


def test_cache_loss_distinguishes_total_from_empty_parameter_tuple():
    dummy = _LossCacheDummy()

    Trainer._cache_loss(dummy, "per_param", 1, torch.tensor(2.0), ())
    Trainer._cache_loss(dummy, "total_loss", 1, torch.tensor(3.0))

    flushed = Trainer._flush_loss_cache(dummy)

    assert flushed[("per_param", ())] == 2.0
    assert flushed[("total_loss", "total")] == 3.0
    assert dummy.loss_by_param["per_param"][()]["losses"] == [2.0]
    assert dummy.loss_by_param["total_loss"]["total"]["losses"] == [3.0]


def test_cache_loss_rejects_non_tuple_parameter_key():
    dummy = _LossCacheDummy()

    with pytest.raises(AssertionError, match="param_tuple must be a tuple or None"):
        Trainer._cache_loss(dummy, "bad_key", 1, torch.tensor(1.0), [])
