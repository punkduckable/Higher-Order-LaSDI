import os
import sys

import numpy
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

import Plotting.Plot as Plot
from ParameterSpace import ParameterSpace
from Plotting.Plot import Plot_Heatmap


def _parameter_space(sample_sizes):
    return ParameterSpace(
        {
            "parameter_space": {
                "parameters": [
                    {
                        "name": f"p{i}",
                        "min": 0.0,
                        "max": float(sample_size - 1),
                        "test_space_type": "uniform",
                        "sample_size": sample_size,
                        "log_scale": False,
                    }
                    for i, sample_size in enumerate(sample_sizes)
                ],
                "test_space": {"type": "grid"},
            }
        }
    )


def test_plot_heatmap_writes_single_2d_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Plot, "Figures_Path", str(tmp_path))
    param_space = _parameter_space([2, 3])
    values = numpy.arange(6, dtype = numpy.float64).reshape(param_space.test_grid_sizes)

    Plot_Heatmap(
        values=values,
        param_space=param_space,
        save_file_name="heat2d.png",
        show_plot=False,
        annotate_cells=False,
    )

    assert (tmp_path / "heat2d.png").exists()


def test_plot_heatmap_writes_one_2d_slice_per_third_parameter(tmp_path, monkeypatch):
    monkeypatch.setattr(Plot, "Figures_Path", str(tmp_path))
    param_space = _parameter_space([2, 2, 3])
    values = numpy.arange(12, dtype = numpy.float64).reshape(param_space.test_grid_sizes)

    Plot_Heatmap(
        values=values,
        param_space=param_space,
        save_file_name="heat3d.png",
        show_plot=False,
        annotate_cells=False,
    )

    assert len(list(tmp_path.glob("heat3d__p2_*.png"))) == 3


def test_plot_heatmap_rejects_parameter_spaces_above_3d():
    param_space = _parameter_space([2, 2, 2, 2])
    values = numpy.zeros(param_space.test_grid_sizes)

    with pytest.raises(ValueError, match="only supports 2D or 3D"):
        Plot_Heatmap(values=values, param_space=param_space, show_plot=False)
