import importlib
import os
import sys

import numpy
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Enums import NextStep
from HLaSDI.Sample import ROM_Discrepancy
from HLaSDI.Schemas import ROMDiscrepancySamplerConfig


class _ParamSpace:
    def __init__(self):
        self.train_space = numpy.array([[0.0], [10.0]])
        self.test_space = numpy.array([[0.0], [2.0], [6.0], [10.0], [20.0]])
        self.appended = None

    def n_train(self):
        return self.train_space.shape[0]

    def n_test(self):
        return self.test_space.shape[0]

    def appendTrainSpace(self, new_sample):
        self.appended = new_sample.copy()
        self.train_space = numpy.concatenate([self.train_space, new_sample], axis = 0)


class _EncoderDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append(args[0] if len(args) > 0 else kwargs.get("device"))
        return super().to(*args, **kwargs)


class _LatentDynamics:
    n_IC = 1
    n_z = 2

    def RHS(self, Z, t_Grid, params, sample = False):
        assert sample == False
        rhs = []
        for i in range(params.shape[0]):
            t = numpy.asarray(t_Grid[i]).reshape(-1)
            theta = float(params[i, 0])
            rhs.append(numpy.stack([theta*t, theta*numpy.ones_like(t)], axis = 1))
        return rhs


class _Trainer:
    def __init__(self):
        self.param_space = _ParamSpace()
        self.encoder_decoder = _EncoderDecoder()
        self.latent_dynamics = _LatentDynamics()
        self.physics = object()
        self.t_Train = [
            torch.tensor([0.0, 0.25, 1.0], dtype = torch.float64),
            numpy.array([0.0, 0.5, 1.5], dtype = numpy.float64),
        ]
        self.t_Test = []
        self.U_Test = []
        self.checked_train_coefficients = False
        self.restart_iter = 12
        self.cached_metrics = []
        self.flushed_epochs = []

    def _check_train_coefficients(self):
        self.checked_train_coefficients = True

    def _cache_metric(self, key, value):
        self.cached_metrics.append((key, value))

    def _flush_metrics_cache(self, epoch):
        self.flushed_epochs.append(epoch)
        return {}


def test_rom_discrepancy_samples_candidate_with_largest_minimum_rhs_discrepancy(monkeypatch):
    rom_module = importlib.import_module("HLaSDI.Sample.ROM_Discrepancy")
    rollout_calls = []

    def mean_rollout(encoder_decoder, physics, latent_dynamics, param_grid, t_Grid, trainer):
        rollout_calls.append((param_grid.copy(), t_Grid))
        Zis = []
        for j in range(param_grid.shape[0]):
            t = numpy.asarray(t_Grid[j]).reshape(-1)
            Zis.append([numpy.stack([t, t*t], axis = 1)])
        return Zis

    monkeypatch.setattr(rom_module, "Mean_Rollout", mean_rollout)

    config = ROMDiscrepancySamplerConfig.model_validate({"type": "ROM_Discrepancy"})
    sampler = ROM_Discrepancy(config)
    trainer = _Trainer()

    next_step = sampler.Sample(trainer)

    assert next_step == NextStep.RunSample
    assert trainer.checked_train_coefficients
    assert trainer.cached_metrics[0][0] == "time/new_sample"
    assert trainer.cached_metrics[0][1] >= 0.0
    assert trainer.flushed_epochs == [trainer.restart_iter]
    assert numpy.allclose(trainer.param_space.appended, numpy.array([[20.0]]))
    assert numpy.allclose(trainer.param_space.train_space[-1, :], numpy.array([20.0]))

    # Only the three non-training test points should be considered as candidates.
    assert len(rollout_calls) == 3
    assert numpy.allclose(rollout_calls[0][0], numpy.array([[2.0], [2.0]]))
    assert numpy.allclose(rollout_calls[1][0], numpy.array([[6.0], [6.0]]))
    assert numpy.allclose(rollout_calls[2][0], numpy.array([[20.0], [20.0]]))

    # The sampler should restore the encoder/decoder to the device it found at entry.
    assert trainer.encoder_decoder.to_calls[-1] == torch.device("cpu")


def test_rom_discrepancy_sampler_is_exported_and_registered():
    from HLaSDI.Initialize import sampler_dict

    assert sampler_dict["ROM_Discrepancy"] is ROM_Discrepancy
