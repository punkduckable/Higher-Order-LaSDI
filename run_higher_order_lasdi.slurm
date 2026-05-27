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
#SBATCH --partition=pdebug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=iffmodel
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#
# Keep SLURM's own wrapper output separate from the Python stdout/stderr logs.
# The Python logs are written below and then moved by scripts/cleanup_run.py.
#SBATCH --output=slurm-%x-%j.log
#SBATCH --error=slurm-%x-%j.err

set -u

# ----------------------------- User settings ------------------------------
# Name of the example file in ./examples. Include the .yml extension.
Example="Thermal.yml"

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
    if [[ -f "${candidate}/src/Workflow.py" ]]; then
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

if [[ ! -f ./venv/bin/activate ]]; then
    echo "ERROR: virtual environment not found at ./venv/bin/activate" >&2
    exit 2
fi

echo "Starting Higher-Order-LaSDI job at $(date)"
echo "Repository: ${REPO_ROOT}"
echo "Config:     ${CONFIG_FILE}"
echo "Stdout:     ${STDOUT_FILE}"
echo "Stderr:     ${STDERR_FILE}"

# shellcheck source=/dev/null
source ./venv/bin/activate

# Use a non-interactive Matplotlib backend for batch jobs.
export MPLBACKEND=Agg

RUN_START_EPOCH="$(date +%s)"

python src/Workflow.py --config "$CONFIG_FILE" > "$STDOUT_FILE" 2> "$STDERR_FILE"
workflow_status=$?

echo "Workflow finished at $(date) with exit code ${workflow_status}"
echo "Running cleanup..."

# Cleanup is intentionally run even if training fails, so partial logs/configs
# are still archived for debugging. It copies result/config files and moves
# top-level Figures outputs generated after the latest save file.
python scripts/cleanup_run.py "$Example" \
    --repo-root "$REPO_ROOT" \
    --examples-dir "$EXAMPLES_PATH" \
    --stdout "$STDOUT_FILE" \
    --stderr "$STDERR_FILE" \
    --min-result-mtime "$RUN_START_EPOCH"
cleanup_status=$?

if [[ "$cleanup_status" -ne 0 ]]; then
    echo "WARNING: cleanup_run.py exited with code ${cleanup_status}" >&2
fi

exit "$workflow_status"
