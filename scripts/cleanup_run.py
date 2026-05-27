#!/usr/bin/env python3
"""Archive Higher-Order-LaSDI outputs after a workflow run.

The script creates a dated run directory under ``Figures`` named like
``May 25 - 1``, ``May 25 - 2``, etc. It then:

* moves requested stdout/stderr/log files into that directory if they exist,
* copies the example YAML config into that directory,
* copies the most recent result save from ``results`` that is not a
  ``*_loss_by_param.pkl`` file,
* copies the most recent ``*_loss_by_param.pkl`` file from ``results``, and
* moves top-level files in ``Figures`` whose modification time is later than
  the latest non-loss result save. Files ending in ``_mean.png`` or
  ``_std.png`` are moved into a ``Coefficient Heatmaps`` subdirectory.

Only direct children of ``Figures`` are moved; existing dated subdirectories
are never traversed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
from pathlib import Path


DEFAULT_LOG_FILES = (
    "output.txt",
    "stdout.txt",
    "stderr.txt",
    "ho_lasdi_stdout.txt",
    "ho_lasdi_stderr.txt",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Archive Higher-Order-LaSDI logs, configs, results, and figures."
    )
    parser.add_argument(
        "example",
        help=(
            "Example YAML file used for the run. Prefer a bare filename such as "
            "'Thermal.yml'; the file must exist in ./examples."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of this script's directory.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing example YAML files. Relative paths are "
            "interpreted relative to --repo-root. Defaults to ./examples "
            "or ./Examples if present."
        ),
    )
    parser.add_argument(
        "--stdout",
        type=Path,
        default=None,
        help="Optional stdout file to move into the run directory.",
    )
    parser.add_argument(
        "--stderr",
        type=Path,
        default=None,
        help="Optional stderr file to move into the run directory.",
    )
    parser.add_argument(
        "--output-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional output/log file to move into the run directory. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without copying or moving files.",
    )
    parser.add_argument(
        "--min-result-mtime",
        type=float,
        default=None,
        help=(
            "Only consider result files modified at or after this Unix timestamp. "
            "The SLURM deck passes the workflow start time to avoid archiving "
            "stale result files if a run fails before saving."
        ),
    )
    return parser.parse_args()


def resolve_example(
    repo_root: Path, example: str, examples_dir_arg: Path | None = None
) -> Path:
    """Return the validated example config path.

    The repository uses ``examples`` (lowercase). A capitalized ``Examples``
    directory is also accepted if present, but the config must be a direct
    child of that directory and must use the ``.yml`` extension.
    """

    if not example.endswith(".yml"):
        raise ValueError(f"Example must include the .yml extension: {example}")

    if examples_dir_arg is not None:
        examples_dir = (
            examples_dir_arg
            if examples_dir_arg.is_absolute()
            else repo_root / examples_dir_arg
        )
    else:
        examples_dir = repo_root / "examples"
        if not examples_dir.is_dir():
            examples_dir = repo_root / "Examples"

    candidate = Path(example)
    if candidate.is_absolute():
        config_path = candidate
    elif candidate.parent == Path("."):
        config_path = examples_dir / candidate.name
    else:
        config_path = repo_root / candidate

    config_path = config_path.resolve()
    examples_dir = examples_dir.resolve()

    if config_path.parent != examples_dir:
        raise ValueError(
            f"Example must be a direct child of {examples_dir}: {config_path}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Example config not found: {config_path}")
    return config_path


def next_run_directory(figures_dir: Path, today: dt.date) -> Path:
    """Create and return the next ``Month Day - N`` directory in ``Figures``."""

    date_prefix = f"{today.strftime('%B')} {today.day}"
    pattern = re.compile(rf"^{re.escape(date_prefix)} - (\d+)(?:\b| .*)")

    used_numbers: set[int] = set()
    if figures_dir.is_dir():
        for path in figures_dir.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                used_numbers.add(int(match.group(1)))

    run_number = 1
    while run_number in used_numbers:
        run_number += 1

    return figures_dir / f"{date_prefix} - {run_number}"


def latest_file(paths: list[Path]) -> Path | None:
    """Return the file with the newest modification time, or ``None``."""

    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def unique_destination(destination: Path) -> Path:
    """Avoid overwriting existing files by appending ``_N`` when needed."""

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_file(source: Path, destination_dir: Path, dry_run: bool) -> None:
    """Copy ``source`` into ``destination_dir`` without overwriting files."""

    destination = unique_destination(destination_dir / source.name)
    print(f"COPY {source} -> {destination}")
    if not dry_run:
        shutil.copy2(source, destination)


def move_file(source: Path, destination_dir: Path, dry_run: bool) -> None:
    """Move ``source`` into ``destination_dir`` without overwriting files."""

    destination = unique_destination(destination_dir / source.name)
    print(f"MOVE {source} -> {destination}")
    if not dry_run:
        shutil.move(str(source), str(destination))


def collect_log_files(repo_root: Path, args: argparse.Namespace) -> list[Path]:
    """Collect requested log files, preserving order and removing duplicates."""

    raw_paths: list[Path] = []
    raw_paths.extend(Path(name) for name in DEFAULT_LOG_FILES)
    if args.stdout is not None:
        raw_paths.append(args.stdout)
    if args.stderr is not None:
        raw_paths.append(args.stderr)
    raw_paths.extend(args.output_file)

    log_files: list[Path] = []
    seen: set[Path] = set()
    for path in raw_paths:
        resolved = path if path.is_absolute() else repo_root / path
        resolved = resolved.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            log_files.append(resolved)
        else:
            print(f"SKIP missing log file: {resolved}")
    return log_files


def top_level_figures_after(figures_dir: Path, timestamp: float) -> list[Path]:
    """Return direct child files of ``Figures`` modified after ``timestamp``."""

    figure_files: list[Path] = []
    for path in figures_dir.iterdir():
        if not path.is_file():
            continue
        if path.stat().st_mtime > timestamp:
            figure_files.append(path)
    return sorted(figure_files, key=lambda path: path.stat().st_mtime)


def is_coefficient_heatmap(path: Path) -> bool:
    """Return True for coefficient heatmap images named ``*_mean/std.png``."""

    lower_name = path.name.lower()
    return lower_name.endswith("_mean.png") or lower_name.endswith("_std.png")


def main() -> int:
    """Archive the run artifacts and return a process exit code."""

    # Fetch arguments, set up directory structure.
    args = parse_args()
    repo_root = args.repo_root.resolve()
    figures_dir = repo_root / "Figures"
    results_dir = repo_root / "results"

    # Get path to config file.
    config_path = resolve_example(repo_root, args.example, args.examples_dir)

    # Checks
    if not figures_dir.is_dir():
        raise FileNotFoundError(f"Figures directory not found: {figures_dir}")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")

    # Set up a directory (and coefficient heatmap sub-directory) to hold the files.
    run_dir = next_run_directory(figures_dir, dt.date.today())
    heatmap_dir = run_dir / "Coefficient Heatmaps"

    print(f"Archive directory: {run_dir}")
    if not args.dry_run:
        run_dir.mkdir(parents=False, exist_ok=False)

    # Logs are moved because they are run-specific scratch files in repo root.
    for log_file in collect_log_files(repo_root, args):
        move_file(log_file, run_dir, args.dry_run)

    # Configs/results are copied so canonical inputs and result saves remain
    # available in their standard repository locations.
    copy_file(config_path, run_dir, args.dry_run)

    # Fetch all files in results (or, if min_result_mtime is defined, then only files in results created after this))
    result_files = [path for path in results_dir.iterdir() if path.is_file()]
    if args.min_result_mtime is not None:
        result_files = [
            path
            for path in result_files
            if path.stat().st_mtime >= args.min_result_mtime
        ]

    # Fetch the save/loss_by_parm files by fetching the last file in results with 
    # that do/do not end with loss_by_parm.
    latest_save = latest_file(
        [path for path in result_files if not path.name.endswith("loss_by_param.pkl")]
    )
    latest_loss_by_param = latest_file(
        [path for path in result_files if path.name.endswith("loss_by_param.pkl")]
    )

    # Copy the save
    if latest_save is None:
        print("WARNING: no non-loss result save found in results; skipping figure move.")
    else:
        copy_file(latest_save, run_dir, args.dry_run)

    # Copy loss_by_parm
    if latest_loss_by_param is None:
        print("WARNING: no *_loss_by_param.pkl file found in results.")
    else:
        copy_file(latest_loss_by_param, run_dir, args.dry_run)

    # Now copy the figures created after the save.
    if latest_save is not None:
        save_mtime = latest_save.stat().st_mtime
        for figure_file in top_level_figures_after(figures_dir, save_mtime):
            if figure_file.resolve().is_relative_to(run_dir.resolve()):
                continue
            if is_coefficient_heatmap(figure_file):
                if not args.dry_run:
                    heatmap_dir.mkdir(exist_ok=True)
                move_file(figure_file, heatmap_dir, args.dry_run)
            else:
                move_file(figure_file, run_dir, args.dry_run)

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
