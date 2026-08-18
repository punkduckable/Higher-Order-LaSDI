#!/bin/bash
#
# SLURM input deck for running Higher-Order-LaSDI from the queue.
#
# Submit with:
#   sbatch run_higher_order_lasdi.slurm
#
# Edit the "User settings" section below before submitting. In particular,
# set Example to the name of a .yml file that exists in ./examples.

# ----------------------------- SLURM settings -----------------------------
# Adjust these defaults for your machine/partition/account as needed.
#SBATCH --job-name=HigherOrderLaSDI
#SBATCH --partition=pbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=iffmodel
#SBATCH --exclusive
#SBATCH --mail-type=all
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#
# Keep SLURM's own wrapper output separate from the Python stdout/stderr logs.
# The Python logs are written below and then moved by scripts/cleanup_run.py.
#SBATCH --output=slurm-%x.log
#SBATCH --error=slurm-%x.err

set -u

# ----------------------------- User settings ------------------------------
# Name of the example file in ./examples. Include the .yml extension.
Example="Advection.yml"

# Repository root. Leave empty to auto-detect. If auto-detection fails on your
# system, set this to the absolute path of the Higher-Order-LaSDI checkout, e.g.
# RepositoryRoot="/g/g20/robertrs/lustre1/AI4NS/Higher-Order-LaSDI"
RepositoryRoot=""

# Directory containing example YAML files. Relative paths are interpreted
# relative to RepositoryRoot.
ExamplesDir="examples"

# Python stdout/stderr files. These are created in the repository root and
# moved into the dated Figures subdirectory by scripts/cleanup_run.py.
STDOUT_FILE="HLaSDI_${SLURM_JOB_ID:-manual}_stdout.txt"
STDERR_FILE="HLaSDI_${SLURM_JOB_ID:-manual}_stderr.txt"

# LC module environment used when PyMFEM/mpi4py were built. Keep these modules
# consistent with the install environment, especially for PyMFEM examples.
LoadLCModules="true"
LCCompilerModule="intel-classic/2021.6.0"
LCMPIModule="mvapich2/2.3.7"


# ----------------------------- Run workflow -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SLURM may execute a copied version of this script from a spool directory, so
# "${BASH_SOURCE[0]}" is not always the repository path. Prefer an explicit
# RepositoryRoot if provided, then the sbatch submission directory, then the
# script directory for interactive/manual runs.
candidate_roots=()
if [[ -n "$RepositoryRoot" ]]; then
    candidate_roots+=("$RepositoryRoot")
fi
if [[ -n "${HLASDI_REPO_ROOT:-}" ]]; then
    candidate_roots+=("$HLASDI_REPO_ROOT")
fi
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    candidate_roots+=("$SLURM_SUBMIT_DIR")
fi
candidate_roots+=("$SCRIPT_DIR")

REPO_ROOT=""
for candidate in "${candidate_roots[@]}"; do
    if [[ -f "${candidate}/scripts/run_experiment.py" ]]; then
        REPO_ROOT="$(cd "$candidate" && pwd)"
        break
    fi
done

if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: could not find the Higher-Order-LaSDI repository root." >&2
    echo "Set RepositoryRoot in this script to the absolute repository path." >&2
    exit 2
fi

cd "$REPO_ROOT" || exit 1

if [[ "$LoadLCModules" == "true" ]]; then
    # Batch shells may not initialize the LC module command by default.
    if ! command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source /etc/profile || true
    fi

    if ! command -v module >/dev/null 2>&1; then
        echo "ERROR: environment modules are not available in this batch shell." >&2
        exit 2
    fi

    module --force purge
    module load StdEnv
    module load "$LCCompilerModule"
    module load "$LCMPIModule"

    # Suppress benign MVAPICH2 import warnings in batch logs.
    export MV2_USE_ALIGNED_ALLOC=1
    export MV2_USE_THREAD_WARNING=0
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ "$ExamplesDir" == /* ]]; then
    EXAMPLES_PATH="$ExamplesDir"
else
    EXAMPLES_PATH="${REPO_ROOT}/${ExamplesDir}"
fi

if [[ ! "$Example" == *.yml ]]; then
    echo "ERROR: Example must include the .yml extension: ${Example}" >&2
    exit 2
fi

if [[ ! -d "$EXAMPLES_PATH" ]]; then
    echo "ERROR: examples directory not found: ${EXAMPLES_PATH}" >&2
    echo "Set ExamplesDir in this script to the directory containing YAML examples." >&2
    exit 2
fi

CONFIG_FILE="${EXAMPLES_PATH}/${Example}"

if [[ "$Example" == */* || ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Example must be a file in ${EXAMPLES_PATH}: ${Example}" >&2
    echo "Available examples:" >&2
    find "$EXAMPLES_PATH" -maxdepth 1 -type f -name '*.yml' -printf '  %f\n' | sort >&2
    exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found at $PYTHON" >&2
    echo "Run 'uv sync --python 3.11 --extra pymfem' in the repository root before submitting this job." >&2
    exit 2
fi

# Activate for subprocesses that inspect VIRTUAL_ENV/PATH, but invoke $PYTHON
# explicitly below so the batch job uses the same interpreter that was tested
# interactively.
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

# Preflight the exact Python environment used by the job. This catches PyMFEM
# environment mismatches before launching the full workflow.
"$PYTHON" -c 'import sys; print("Python:", sys.executable); import mfem; print("mfem:", mfem.__file__)'

echo "Starting Higher-Order-LaSDI job at $(date)"
echo "Repository: ${REPO_ROOT}"
echo "Config:     ${CONFIG_FILE}"
echo "Python:     ${PYTHON}"
echo "Stdout:     ${STDOUT_FILE}"
echo "Stderr:     ${STDERR_FILE}"

# Use a non-interactive Matplotlib backend for batch jobs.
export MPLBACKEND=Agg

RUN_START_EPOCH="$(date +%s)"

echo "Launching training workflow (LaSDI) at $(date)"
"$PYTHON" scripts/run_experiment.py --config "$CONFIG_FILE" > "$STDOUT_FILE" 2> "$STDERR_FILE"
workflow_status=$?

echo "Training workflow finished at $(date) with exit code ${workflow_status}"

ARTIFACT_FILE=""
analysis_status=0
if [[ "$workflow_status" -eq 0 ]]; then
    # Fetch the newest serialized experiment artifact created by this job. This excludes
    # per-parameter loss JSONL files and avoids accidentally analyzing a stale artifact.
    ARTIFACT_FILE="$(
        "$PYTHON" -c 'from pathlib import Path; import sys
start = float(sys.argv[1])
results = Path("results")
files = [p for p in results.iterdir() if p.is_file() and not p.name.endswith("loss_by_param.jsonl") and p.stat().st_mtime >= start] if results.is_dir() else []
print(max(files, key=lambda p: p.stat().st_mtime).resolve() if files else "")' "$RUN_START_EPOCH"
    )"

    if [[ -n "$ARTIFACT_FILE" && -f "$ARTIFACT_FILE" ]]; then
        echo "Launching analysis workflow at $(date)"
        echo "Artifact: ${ARTIFACT_FILE}"
        "$PYTHON" scripts/analyze_experiment.py --artifact "$ARTIFACT_FILE" >> "$STDOUT_FILE" 2>> "$STDERR_FILE"
        analysis_status=$?
        echo "Analysis workflow finished at $(date) with exit code ${analysis_status}"
    else
        echo "ERROR: training succeeded, but no new serialized experiment artifact was found in results/." >&2
        analysis_status=2
    fi
else
    echo "Skipping analysis because training failed." >&2
fi

echo "Running cleanup..."

# Cleanup is intentionally run even if training fails, so partial logs/configs
# are still archived for debugging. It copies result/config files and moves
# top-level Figures outputs generated by this job.
cleanup_args=(
    "$Example"
    --repo-root "$REPO_ROOT"
    --examples-dir "$EXAMPLES_PATH"
    --stdout "$STDOUT_FILE"
    --stderr "$STDERR_FILE"
    --min-result-mtime "$RUN_START_EPOCH"
)
if [[ -n "$ARTIFACT_FILE" && -f "$ARTIFACT_FILE" ]]; then
    cleanup_args+=(--artifact "$ARTIFACT_FILE" --min-figure-mtime "$RUN_START_EPOCH")
fi

cleanup_output="$("$PYTHON" scripts/cleanup_run.py "${cleanup_args[@]}" 2>&1)"
cleanup_status=$?
printf '%s\n' "$cleanup_output"

if [[ "$cleanup_status" -ne 0 ]]; then
    echo "WARNING: cleanup_run.py exited with code ${cleanup_status}" >&2
else
    ARCHIVE_DIR="$(printf '%s\n' "$cleanup_output" | sed -n 's/^Archive directory: //p' | tail -n 1)"
    if [[ -n "$ARCHIVE_DIR" && -d "$ARCHIVE_DIR" ]]; then
        METRICS_FILE="$(
            "$PYTHON" -c 'from pathlib import Path; import sys
run_dir = Path(sys.argv[1])
metrics_files = sorted(
    run_dir.glob("*_metrics.jsonl"),
    key=lambda path: path.stat().st_mtime,
)
print(metrics_files[-1].resolve() if metrics_files else "")' "$ARCHIVE_DIR"
        )"

        if [[ -n "$METRICS_FILE" && -f "$METRICS_FILE" ]]; then
            TB_LOGDIR="${ARCHIVE_DIR}/tb_runs"
            echo "Building TensorBoard files at $(date)"
            echo "Metrics: ${METRICS_FILE}"
            echo "Logdir:  ${TB_LOGDIR}"
            "$PYTHON" scripts/jsonl_to_tensorboard.py "$METRICS_FILE" --logdir "$TB_LOGDIR"
            tensorboard_status=$?
            if [[ "$tensorboard_status" -ne 0 ]]; then
                echo "WARNING: jsonl_to_tensorboard.py exited with code ${tensorboard_status}" >&2
            fi
        else
            echo "WARNING: no archived *_metrics.jsonl file found for TensorBoard conversion." >&2
        fi
    else
        echo "WARNING: could not determine archive directory for TensorBoard conversion." >&2
    fi
fi

if [[ "$workflow_status" -ne 0 ]]; then
    exit "$workflow_status"
fi
exit "$analysis_status"
