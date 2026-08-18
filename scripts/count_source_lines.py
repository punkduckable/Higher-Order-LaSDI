#!/usr/bin/env python3
"""Count lines in Python source files under the main repo code directories."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = ("scripts", "Test", "src")


def count_lines(path: Path) -> int:
    """Return the number of physical lines in a text file."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_python_files() -> list[Path]:
    """Return sorted .py files from the configured target directories."""
    files: list[Path] = []
    for dirname in TARGET_DIRS:
        directory = ROOT / dirname
        if directory.exists():
            files.extend(directory.rglob("*.py"))
    return sorted(files)


def main() -> None:
    total = 0
    directory_totals = {dirname: 0 for dirname in TARGET_DIRS}
    directory_file_counts = {dirname: 0 for dirname in TARGET_DIRS}
    sub_dir_totals = {dirname: {} for dirname in TARGET_DIRS}
    sub_dir_file_counts = {dirname: {} for dirname in TARGET_DIRS}
    files = iter_python_files()

    for path in files:
        lines = count_lines(path)
        relative_path = path.relative_to(ROOT)

        # Fetch the top level directory name.
        dirname = relative_path.parts[0]

        # Update sub-directory totals, file counts
        # If the number of parts is > 2, then there must be a sub-directory structure
        if len(relative_path.parts) > 2:
            # Get the sub-dir name
            sub_dir_name: str = str(relative_path.parts[1])
            for part in relative_path.parts[2:-1]:
                sub_dir_name += "/" + str(part)

            # Update totals, file counts
            sub_dir_totals[dirname][sub_dir_name] = (
                sub_dir_totals[dirname].get(sub_dir_name, 0) + lines
            )
            sub_dir_file_counts[dirname][sub_dir_name] = (
                sub_dir_file_counts[dirname].get(sub_dir_name, 0) + 1
            )

        # Update directory totals, file counts
        directory_totals[dirname] += lines
        directory_file_counts[dirname] += 1

        # Update total lines counted
        total += lines

        # Report!
        print(f"{lines:7d}  {relative_path}")

    print("-" * 30)
    for dirname in TARGET_DIRS:
        # Print total for this top-level directory
        print(
            f"{directory_totals[dirname]:7d}  {dirname}/  ({directory_file_counts[dirname]} files)"
        )

        # Print totals for sub-directories in this top-level directory
        for sub_dir_name, totals in sub_dir_totals[dirname].items():
            print(
                f"{totals:7d}  "
                f"  {sub_dir_name}/  ({sub_dir_file_counts[dirname][sub_dir_name]} files)"
            )

    print("-" * 30)
    print(f"{total:7d}  total ({len(files)} files)")


if __name__ == "__main__":
    main()
