#!/usr/bin/env python3
"""Archive Higher-Order-LaSDI outputs after a workflow run.

The script creates a dated run directory under ``Figures`` named like
``May 25 - 1``, ``May 25 - 2``, etc. It then:

* moves requested stdout/stderr/log files into that directory if they exist,
* copies the example YAML config into that directory,
* copies the requested result save, or the most recent result save from
  ``results`` that is not a ``*_loss_by_param.jsonl`` file,
* copies the matching ``*_loss_by_param.jsonl`` file from ``results``, and
* moves top-level files in ``Figures`` whose modification time is later than
  the archived result save, or later than ``--min-figure-mtime`` when supplied.
  Coefficient mean/std heatmap files are moved into a
  ``Coefficient Heatmaps`` subdirectory.

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

LOSS_BY_PARAM_SUFFIX = "_loss_by_param.jsonl"


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
    parser.add_argument(
        "--result-save",
        "--artifact",
        dest="result_save",
        type=Path,
        default=None,
        help=(
            "Explicit serialized experiment artifact to copy into the archive. "
            "Relative paths are interpreted relative to --repo-root, then ./results."
        ),
    )
    parser.add_argument(
        "--min-figure-mtime",
        type=float,
        default=None,
        help=(
            "Only move figures modified after this Unix timestamp. If omitted, "
            "figures are moved only if they are newer than the archived result save."
        ),
    )
    return parser.parse_args()


def resolve_example(
    repo_root: Path, example: str, examples_dir_arg: Path | None = None
) -> Path:
    """Return the resolved example config path.

    The repository uses ``examples`` (lowercase). A capitalized ``Examples``
    directory is also accepted if present, but the config must be a direct
    child of that directory and must use the ``.yml`` extension. The file is
    not required to exist; missing files are skipped later with a warning so
    cleanup can still archive whatever artifacts are present.
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
    """Return the existing file with the newest modification time, or ``None``."""

    existing_paths = [path for path in paths if path.is_file()]
    if not existing_paths:
        return None
    return max(existing_paths, key=lambda path: path.stat().st_mtime)


def resolve_result_save(repo_root: Path, result_save: Path) -> Path:
    """Resolve an explicit serialized experiment artifact path."""

    candidates: list[Path] = []
    if result_save.is_absolute():
        candidates.append(result_save)
    else:
        candidates.append(repo_root / result_save)
        candidates.append(repo_root / "results" / result_save)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(f"result save does not exist: {result_save}")


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

    if not source.is_file():
        print(f"SKIP missing file: {source}")
        return

    destination = unique_destination(destination_dir / source.name)
    print(f"COPY {source} -> {destination}")
    if not dry_run:
        shutil.copy2(source, destination)


def move_file(source: Path, destination_dir: Path, dry_run: bool) -> None:
    """Move ``source`` into ``destination_dir`` without overwriting files."""

    if not source.is_file():
        print(f"SKIP missing file: {source}")
        return

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


def is_loss_by_param_file(path: Path) -> bool:
    """Return True for per-parameter loss JSONL files."""

    return path.name.endswith(LOSS_BY_PARAM_SUFFIX)


def loss_by_param_prefix(path: Path) -> str:
    """Return the physics/result prefix for a per-parameter loss file."""

    return path.name[: -len(LOSS_BY_PARAM_SUFFIX)]


def loss_by_param_matches_save(loss_file: Path, result_save: Path) -> bool:
    """Return True when ``loss_file`` appears to belong to ``result_save``.

    Result saves include a timestamp (for example ``Thermal_07_30_2026_19_18.npy``),
    while loss files are overwritten under the physics-type prefix (for example
    ``Thermal_loss_by_param.jsonl``). Match by that prefix instead of modification
    time only, because restart/resume workflows can leave the loss JSONL file older
    than ``--min-result-mtime`` even though it is the companion diagnostics file.
    """

    if not is_loss_by_param_file(loss_file):
        return False

    loss_prefix = loss_by_param_prefix(loss_file)
    save_stem = result_save.stem
    return save_stem == loss_prefix or save_stem.startswith(f"{loss_prefix}_")


def select_loss_by_param_file(
    all_result_files: list[Path],
    filtered_result_files: list[Path],
    latest_save: Path | None,
) -> Path | None:
    """Select the best per-parameter loss file to archive."""

    if latest_save is None:
        return latest_file(
            [path for path in filtered_result_files if is_loss_by_param_file(path)]
        )

    matching_loss_files = [
        path
        for path in all_result_files
        if loss_by_param_matches_save(path, latest_save)
    ]
    if matching_loss_files:
        longest_prefix_length = max(
            len(loss_by_param_prefix(path)) for path in matching_loss_files
        )
        return latest_file(
            [
                path
                for path in matching_loss_files
                if len(loss_by_param_prefix(path)) == longest_prefix_length
            ]
        )

    return latest_file(
        [path for path in filtered_result_files if is_loss_by_param_file(path)]
    )


def is_coefficient_heatmap(path: Path) -> bool:
    """Return True for coefficient mean/std heatmap images."""

    lower_name = path.name.lower()
    if not lower_name.endswith(".png"):
        return False

    return (
        re.search(r"coefficient_\d+_(?:mean|std)(?:__.*)?\.png$", lower_name)
        is not None
    )


def main() -> int:
    """Archive the run artifacts and return a process exit code."""

    # Fetch arguments, set up directory structure.
    args = parse_args()
    repo_root = args.repo_root.resolve()
    figures_dir = repo_root / "Figures"
    results_dir = repo_root / "results"

    # Get path to config file.
    config_path = resolve_example(repo_root, args.example, args.examples_dir)

    # Resolve an explicitly requested serialized artifact before any filesystem mutations. This
    # avoids creating a run directory or moving logs when the artifact path is misspelled.
    explicit_result_save = (
        resolve_result_save(repo_root, args.result_save)
        if args.result_save is not None
        else None
    )

    # Ensure output directories exist. If a path exists but is not a directory,
    # fail fast rather than silently writing somewhere unexpected.
    for directory in (figures_dir, results_dir):
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(f"expected directory path: {directory}")
        if not directory.exists():
            print(f"CREATE directory: {directory}")
            if not args.dry_run:
                directory.mkdir(parents=True, exist_ok=True)

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

    # Fetch all files in results (or, if min_result_mtime is defined, then only
    # files in results created after this).
    # In dry-run mode the directory may not actually have been created above.
    all_result_files = (
        [path for path in results_dir.iterdir() if path.is_file()]
        if results_dir.is_dir()
        else []
    )
    result_files = all_result_files
    if args.min_result_mtime is not None:
        result_files = [
            path
            for path in result_files
            if path.stat().st_mtime >= args.min_result_mtime
        ]

    # Fetch the save/loss_by_param files. Prefer an explicit serialized artifact when supplied;
    # otherwise keep the legacy behavior of selecting the most recent non-loss result file.
    if args.result_save is None:
        latest_save = latest_file(
            [path for path in result_files if not is_loss_by_param_file(path)]
        )
    else:
        latest_save = explicit_result_save

    latest_loss_by_param = select_loss_by_param_file(
        all_result_files,
        result_files,
        latest_save,
    )

    # Copy the save
    if latest_save is None:
        print("WARNING: no non-loss result save found in results; skipping figure move.")
    else:
        copy_file(latest_save, run_dir, args.dry_run)

    # Copy loss_by_param
    if latest_loss_by_param is None:
        print("WARNING: no *_loss_by_param.jsonl file found in results.")
    else:
        if (
            args.min_result_mtime is not None
            and latest_loss_by_param.stat().st_mtime < args.min_result_mtime
        ):
            print(
                "WARNING: copying matching loss_by_param file older than "
                "--min-result-mtime: "
                f"{latest_loss_by_param}"
            )
        copy_file(latest_loss_by_param, run_dir, args.dry_run)

    # Now move the figures created after the save, or after the explicit figure timestamp if one
    # was provided. This supports the split train/analyze workflow, where analysis may be run as a
    # separate job after the serialized artifact already exists.
    if latest_save is not None:
        figure_mtime = (
            args.min_figure_mtime
            if args.min_figure_mtime is not None
            else latest_save.stat().st_mtime
        )
        for figure_file in top_level_figures_after(figures_dir, figure_mtime):
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
