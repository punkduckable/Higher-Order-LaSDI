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


class _MetricsCacheDummy:
    def __init__(self, metrics_path):
        self._metrics_cache = []
        self.metrics_path = str(metrics_path)

    def _cache_metric(self, *args, **kwargs):
        return Trainer._cache_metric(self, *args, **kwargs)


def test_cache_metric_writes_one_jsonl_row_and_preserves_keys(tmp_path):
    metrics_path = tmp_path / "Dummy_metrics.jsonl"
    dummy = _MetricsCacheDummy(metrics_path)

    Trainer._cache_metric(dummy, "loss/per_param/[0.25 2]", torch.tensor(2.0))
    Trainer._cache_metric(dummy, "loss/total", torch.tensor(3.0))

    flushed = Trainer._flush_metrics_cache(dummy, 7)

    assert flushed["loss/per_param/[0.25 2]"] == 2.0
    assert flushed["loss/total"] == 3.0
    assert dummy._metrics_cache == []

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert rows == [
        {
            "epoch": 7,
            "metrics": [
                {"loss/per_param/[0.25 2]": 2.0},
                {"loss/total": 3.0},
            ],
        }
    ]


def test_cache_metric_accepts_python_scalar_timing(tmp_path):
    dummy = _MetricsCacheDummy(tmp_path / "Dummy_metrics.jsonl")

    Trainer._cache_metric(dummy, "time/step", 1.0)
    flushed = Trainer._flush_metrics_cache(dummy, 3)

    assert flushed["time/step"] == 1.0


def test_cache_metric_rejects_unsupported_metric_type():
    dummy = _MetricsCacheDummy("unused.jsonl")

    with pytest.raises(TypeError, match="detached scalar torch.Tensor or a Python scalar"):
        Trainer._cache_metric(dummy, "bad_metric", object())


def test_flush_metrics_cache_rejects_nonfinite_metrics(tmp_path):
    dummy = _MetricsCacheDummy(tmp_path / "Dummy_metrics.jsonl")
    Trainer._cache_metric(dummy, "nan_metric", torch.tensor(float("nan")))

    with pytest.raises(AssertionError, match="cached tensor metrics must be finite"):
        Trainer._flush_metrics_cache(dummy, 1)


def test_ld_loss_container_accepts_scalar_losses_and_metrics():
    params = numpy.array([[0.25], [0.75]])
    losses = LD_Loss_Container(
        losses={
            "LD": torch.tensor(3.0),
            "coef": torch.tensor([3.0]),
        },
        weights={"LD": 1.0, "coef": 0.5},
        params=params,
        metrics={"loss/LD/total": torch.tensor(1.0)},
    )

    assert losses.params is params
    assert torch.allclose(losses.losses["LD"], torch.tensor(3.0))


def test_ld_loss_container_rejects_per_parameter_loss_lists():
    with pytest.raises(ValueError, match="instance of Tensor"):
        LD_Loss_Container(
            losses={"LD": [torch.tensor(1.0)]},
            weights={"LD": 1.0},
            params=numpy.array([[0.25], [0.75]]),
        )
