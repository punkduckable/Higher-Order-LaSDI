import os
import sys
from pathlib import Path


SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(SCRIPTS)

from cleanup_run import select_metrics_file


def test_select_metrics_uses_matching_save_prefix_even_if_filtered_out(tmp_path):
    save = tmp_path / "Thermal_07_30_2026_19_18.npy"
    metrics = tmp_path / "Thermal_metrics.jsonl"
    other_metrics = tmp_path / "OtherPhysics_metrics.jsonl"

    save.write_bytes(b"save")
    metrics.write_bytes(b"loss")
    other_metrics.write_bytes(b"other")

    os.utime(save, (200.0, 200.0))
    os.utime(metrics, (50.0, 50.0))
    os.utime(other_metrics, (300.0, 300.0))

    assert (
        select_metrics_file(
            all_result_files=[save, metrics, other_metrics],
            filtered_result_files=[save],
            latest_save=save,
        )
        == metrics
    )


def test_select_metrics_prefers_longest_matching_prefix(tmp_path):
    save = tmp_path / "Thermal_Weak_07_30_2026_19_18.npy"
    broad_metrics = tmp_path / "Thermal_metrics.jsonl"
    exact_metrics = tmp_path / "Thermal_Weak_metrics.jsonl"

    save.write_bytes(b"save")
    broad_metrics.write_bytes(b"broad")
    exact_metrics.write_bytes(b"exact")

    os.utime(broad_metrics, (300.0, 300.0))
    os.utime(exact_metrics, (100.0, 100.0))

    assert (
        select_metrics_file(
            all_result_files=[save, broad_metrics, exact_metrics],
            filtered_result_files=[save],
            latest_save=save,
        )
        == exact_metrics
    )
