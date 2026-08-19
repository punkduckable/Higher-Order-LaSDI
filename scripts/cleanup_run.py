#!/usr/bin/env python3
"""Gather Higher-Order-LaSDI non-figure outputs into a run-specific results directory.

The run-specific directory is the trainer's ``results_dir``:

    results/<trainer type>_<date/time>_<pid>/

``run_experiment.py`` writes the serialized artifact, metrics JSONL, and a copy of the launch
config there. This cleanup script moves requested stdout/stderr/log files into the same directory
and copies the example config if it was not already backed up. Figures are intentionally left in
the separate run-specific ``Figures/<run_ID>`` directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_LOG_FILES = (
    "output.txt",
    "stdout.txt",
    "stderr.txt",
    "ho_lasdi_stdout.txt",
    "ho_lasdi_stderr.txt",
)

METRICS_SUFFIX = "_metrics.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Archive Higher-Order-LaSDI logs, configs, and results."
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
            "Relative paths are interpreted relative to --repo-root, --run-dir, then ./results."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Run-specific results directory (trainer.results_dir). Relative paths are "
            "interpreted relative to --repo-root."
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


def latest_file(paths: list[Path]) -> Path | None:
    """Return the existing file with the newest modification time, or ``None``."""

    existing_paths = [path for path in paths if path.is_file()]
    if not existing_paths:
        return None
    return max(existing_paths, key=lambda path: path.stat().st_mtime)


def resolve_result_save(repo_root: Path, result_save: Path, run_dir: Path) -> Path:
    """Resolve an explicit serialized experiment artifact path."""

    candidates: list[Path] = []
    if result_save.is_absolute():
        candidates.append(result_save)
    else:
        candidates.append(repo_root / result_save)
        candidates.append(run_dir / result_save)
        candidates.append(repo_root / "results" / result_save)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(f"result save does not exist: {result_save}")


def resolve_run_dir(repo_root: Path, run_dir: Path) -> Path:
    """Resolve the explicit run-specific results directory."""

    if run_dir.is_absolute():
        return run_dir.resolve()
    return (repo_root / run_dir).resolve()


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

    if source.resolve() == (destination_dir / source.name).resolve():
        print(f"SKIP already in archive: {source}")
        return

    destination = unique_destination(destination_dir / source.name)
    print(f"COPY {source} -> {destination}")
    if not dry_run:
        shutil.copy2(source, destination)


def copy_file_if_missing(source: Path, destination_dir: Path, dry_run: bool) -> None:
    """Copy ``source`` into ``destination_dir`` only when that filename is absent."""

    if not source.is_file():
        print(f"SKIP missing file: {source}")
        return

    destination = destination_dir / source.name
    if destination.exists():
        print(f"SKIP existing file: {destination}")
        return

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


def is_metrics_file(path: Path) -> bool:
    """Return True for metric JSONL files."""

    return path.name.endswith(METRICS_SUFFIX)


def is_result_save(path: Path) -> bool:
    """Return True for serialized experiment artifacts."""

    return path.is_file() and path.suffix == ".npy"


def metrics_prefix(path: Path) -> str:
    """Return the physics/result prefix for a metric file."""

    return path.name[: -len(METRICS_SUFFIX)]


def metrics_matches_save(metrics_file: Path, result_save: Path) -> bool:
    """Return True when ``metrics_file`` appears to belong to ``result_save``.

    Result saves include a timestamp (for example ``Thermal_07_30_2026_19_18.npy``),
    while metrics files are overwritten under the physics-type prefix (for example
    ``Thermal_metrics.jsonl``). Match by that prefix instead of modification
    time only, because restart/resume workflows can leave the metrics JSONL file older
    than ``--min-result-mtime`` even though it is the companion diagnostics file.
    """

    if not is_metrics_file(metrics_file):
        return False

    metrics_file_prefix = metrics_prefix(metrics_file)
    save_stem = result_save.stem
    return save_stem == metrics_file_prefix or save_stem.startswith(f"{metrics_file_prefix}_")


def select_metrics_file(
    all_result_files: list[Path],
    filtered_result_files: list[Path],
    latest_save: Path | None,
) -> Path | None:
    """Select the best metric file to archive."""

    if latest_save is None:
        return latest_file(
            [path for path in filtered_result_files if is_metrics_file(path)]
        )

    matching_metrics_files = [
        path
        for path in all_result_files
        if metrics_matches_save(path, latest_save)
    ]
    if matching_metrics_files:
        longest_prefix_length = max(
            len(metrics_prefix(path)) for path in matching_metrics_files
        )
        return latest_file(
            [
                path
                for path in matching_metrics_files
                if len(metrics_prefix(path)) == longest_prefix_length
            ]
        )

    return latest_file(
        [path for path in filtered_result_files if is_metrics_file(path)]
    )


def main() -> int:
    """Archive the run artifacts and return a process exit code."""

    # Fetch arguments, set up directory structure.
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = resolve_run_dir(repo_root, args.run_dir)

    # Get path to config file.
    config_path = resolve_example(repo_root, args.example, args.examples_dir)

    # Ensure the run-specific results directory exists. If a path exists but is not a directory,
    # fail fast rather than silently writing somewhere unexpected.
    if run_dir.exists() and not run_dir.is_dir():
        raise NotADirectoryError(f"expected run directory path: {run_dir}")
    if not run_dir.exists():
        print(f"CREATE directory: {run_dir}")
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve an explicitly requested serialized artifact before any filesystem mutations. This
    # avoids creating a run directory or moving logs when the artifact path is misspelled.
    explicit_result_save = (
        resolve_result_save(repo_root, args.result_save, run_dir)
        if args.result_save is not None
        else None
    )

    print(f"Archive directory: {run_dir}")

    # Logs are moved because they are run-specific scratch files in repo root.
    for log_file in collect_log_files(repo_root, args):
        move_file(log_file, run_dir, args.dry_run)

    # run_experiment.py copies this config at trainer initialization time. If cleanup is called
    # separately and the config is missing from the run directory, copy it now without making a
    # duplicate.
    copy_file_if_missing(config_path, run_dir, args.dry_run)

    # Fetch all files in the run-specific results directory (or, if min_result_mtime is defined,
    # only files created after this).
    # In dry-run mode the directory may not actually have been created above.
    all_result_files = [path for path in run_dir.iterdir() if path.is_file()] if run_dir.is_dir() else []
    result_files = all_result_files
    if args.min_result_mtime is not None:
        result_files = [
            path
            for path in result_files
            if path.stat().st_mtime >= args.min_result_mtime
        ]

    # Fetch the save/metrics files. Prefer an explicit serialized artifact when supplied;
    # otherwise select the most recent serialized experiment artifact in this run directory.
    if args.result_save is None:
        latest_save = latest_file(
            [path for path in result_files if is_result_save(path)]
        )
    else:
        latest_save = explicit_result_save

    latest_metrics = select_metrics_file(
        all_result_files,
        result_files,
        latest_save,
    )

    # Copy the save
    if latest_save is None:
        print("WARNING: no serialized result save found in run results directory.")
    else:
        copy_file(latest_save, run_dir, args.dry_run)

    # Copy metrics
    if latest_metrics is None:
        print("WARNING: no *_metrics.jsonl file found in run results directory.")
    else:
        if (
            args.min_result_mtime is not None
            and latest_metrics.stat().st_mtime < args.min_result_mtime
        ):
            print(
                "WARNING: copying matching metrics file older than "
                "--min-result-mtime: "
                f"{latest_metrics}"
            )
        copy_file(latest_metrics, run_dir, args.dry_run)

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
