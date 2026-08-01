import os
import sys
from pathlib import Path


SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(SCRIPTS)

from cleanup_run import is_coefficient_heatmap, select_loss_by_param_file


def test_coefficient_heatmap_detects_sliced_mean_and_std_names():
    assert is_coefficient_heatmap(
        Path("ThermalCoefficient_0_mean__initial temp_2.98e+02.png")
    )
    assert is_coefficient_heatmap(
        Path("ThermalCoefficient_12_std__initial temp_3.78e+02.png")
    )


def test_coefficient_heatmap_ignores_non_coefficient_heatmaps():
    assert not is_coefficient_heatmap(
        Path("Thermal_U_STD_Heatmap__initial temp_2.98e+02.png")
    )
    assert not is_coefficient_heatmap(Path("Thermal_U_Recon_Rel_Error.png"))


def test_select_loss_by_param_uses_matching_save_prefix_even_if_filtered_out(tmp_path):
    save = tmp_path / "Thermal_07_30_2026_19_18.npy"
    loss = tmp_path / "Thermal_loss_by_param.pkl"
    other_loss = tmp_path / "OtherPhysics_loss_by_param.pkl"

    save.write_bytes(b"save")
    loss.write_bytes(b"loss")
    other_loss.write_bytes(b"other")

    os.utime(save, (200.0, 200.0))
    os.utime(loss, (50.0, 50.0))
    os.utime(other_loss, (300.0, 300.0))

    assert (
        select_loss_by_param_file(
            all_result_files=[save, loss, other_loss],
            filtered_result_files=[save],
            latest_save=save,
        )
        == loss
    )


def test_select_loss_by_param_prefers_longest_matching_prefix(tmp_path):
    save = tmp_path / "Thermal_Weak_07_30_2026_19_18.npy"
    broad_loss = tmp_path / "Thermal_loss_by_param.pkl"
    exact_loss = tmp_path / "Thermal_Weak_loss_by_param.pkl"

    save.write_bytes(b"save")
    broad_loss.write_bytes(b"broad")
    exact_loss.write_bytes(b"exact")

    os.utime(broad_loss, (300.0, 300.0))
    os.utime(exact_loss, (100.0, 100.0))

    assert (
        select_loss_by_param_file(
            all_result_files=[save, broad_loss, exact_loss],
            filtered_result_files=[save],
            latest_save=save,
        )
        == exact_loss
    )
