#!/usr/bin/env python3
"""Convert LaSDI per-epoch JSONL metrics to TensorBoard event files.

The trainers intentionally write package-agnostic JSON Lines files during training. This script is
post-processing only: it reads a completed ``*_metrics.jsonl`` file and writes TensorBoard
scalar events that can be viewed with ``tensorboard --logdir ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


METRICS_SUFFIX = "_metrics"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Convert a LaSDI *_metrics.jsonl file to TensorBoard scalar events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to the JSON Lines metrics file produced by a trainer.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help=(
            "Directory where TensorBoard event files will be written. If omitted, uses "
            "tb_runs/<jsonl stem without _metrics>."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional subdirectory name under --logdir. Use this when converting multiple JSONL "
            "files into the same parent logdir."
        ),
    )
    parser.add_argument(
        "--param-names",
        type=str,
        default=None,
        help=(
            "Deprecated no-op retained for CLI compatibility with old commands; metric keys are "
            "already serialized by the trainer."
        ),
    )
    parser.add_argument(
        "--totals-only",
        action="store_true",
        help="Only export metric keys ending in /total.",
    )
    return parser.parse_args()


def default_logdir(jsonl_path: Path) -> Path:
    """Return the default TensorBoard output directory for a JSONL metrics file."""

    stem = jsonl_path.stem
    if stem.endswith(METRICS_SUFFIX):
        stem = stem[: -len(METRICS_SUFFIX)]
    return Path("tb_runs") / stem


def sanitize_tag_part(value: str) -> str:
    """Make one component of a TensorBoard tag stable and readable."""

    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_.=+\-]+", "_", value)
    value = value.strip("_")
    return value if value else "unnamed"


def format_param_value(value: Any) -> str:
    """Format one JSON parameter value for use in a TensorBoard tag."""

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return "%.8e" % value
    if isinstance(value, str):
        return sanitize_tag_part(value)
    if value is None:
        return "None"
    raise TypeError("parameter values must be JSON scalars, got %s" % type(value))


def parse_param_names(param_names_arg: str | None) -> list[str] | None:
    """Parse an optional comma-separated parameter-name list."""

    if param_names_arg is None:
        return None
    param_names = [sanitize_tag_part(part) for part in param_names_arg.split(",")]
    if any(name == "unnamed" for name in param_names):
        raise ValueError("--param-names must not contain empty names")
    return param_names


def tensorboard_tag(loss_name: str, param: list[Any] | None, param_names: list[str] | None) -> str:
    """Build the TensorBoard scalar tag for one loss record."""

    loss_name = sanitize_tag_part(loss_name)
    if param is None:
        return "loss/%s/total" % loss_name

    if param_names is not None and len(param_names) != len(param):
        raise ValueError(
            "--param-names has %d entries, but encountered param with %d values: %s"
            % (len(param_names), len(param), str(param))
        )

    names = param_names if param_names is not None else ["p%d" % i for i in range(len(param))]
    param_label = "__".join(
        "%s=%s" % (sanitize_tag_part(name), format_param_value(value))
        for name, value in zip(names, param)
    )
    return "loss/%s/by_param/%s" % (loss_name, param_label)


def read_jsonl_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read and validate JSONL rows from ``jsonl_path``."""

    if not jsonl_path.is_file():
        raise FileNotFoundError("JSONL metrics file does not exist: %s" % jsonl_path)

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped == "":
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError("line %d must contain a JSON object" % line_number)
            if "epoch" not in row:
                raise ValueError("line %d is missing required key 'epoch'" % line_number)
            if "metrics" not in row:
                raise ValueError("line %d is missing required key 'metrics'" % line_number)
            if not isinstance(row["metrics"], list):
                raise ValueError("line %d key 'metrics' must be a list" % line_number)
            rows.append(row)

    if len(rows) == 0:
        raise ValueError("JSONL metrics file contains no rows: %s" % jsonl_path)
    return rows


def convert_jsonl_to_tensorboard(
    jsonl_path: Path,
    logdir: Path,
    param_names: list[str] | None,
    totals_only: bool,
) -> int:
    """Convert one JSONL metrics file into TensorBoard scalar events.

    Returns the number of scalar records written.
    """

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise SystemExit(
            "TensorBoard support is not installed. Install the optional visualization "
            "dependencies with `uv sync --extra viz`, then rerun this command."
        ) from error

    rows = read_jsonl_rows(jsonl_path)
    logdir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    writer = SummaryWriter(log_dir=str(logdir))
    try:
        for row in rows:
            epoch = int(row["epoch"])
            for record in row["metrics"]:
                if not isinstance(record, dict):
                    raise ValueError("metric records must be JSON objects, got %s" % type(record))
                if len(record) != 1:
                    raise ValueError("metric records must contain exactly one key/value pair")

                tag, value_raw = next(iter(record.items()))
                value = float(value_raw)

                if not isinstance(tag, str):
                    raise ValueError("metric key must be a string, got %s" % type(tag))
                if not math.isfinite(value):
                    raise ValueError("metric value must be finite, got %s" % value)
                if totals_only and not tag.endswith("/total"):
                    continue

                writer.add_scalar(tag, value, epoch)
                n_written += 1
    finally:
        writer.close()

    return n_written


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    jsonl_path = args.jsonl_path.expanduser().resolve()
    logdir = args.logdir if args.logdir is not None else default_logdir(jsonl_path)
    logdir = logdir.expanduser()
    if args.run_name is not None:
        logdir = logdir / sanitize_tag_part(args.run_name)
    logdir = logdir.resolve()

    param_names = parse_param_names(args.param_names)
    n_written = convert_jsonl_to_tensorboard(
        jsonl_path=jsonl_path,
        logdir=logdir,
        param_names=param_names,
        totals_only=args.totals_only,
    )
    print("Wrote %d TensorBoard scalar events to %s" % (n_written, logdir))
    print("View with: tensorboard --logdir %s" % logdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
