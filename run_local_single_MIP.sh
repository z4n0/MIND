#!/usr/bin/env bash
set -euo pipefail

# ── 0) repo root & logs ─────────────────────────────────────────────────────
# Equivalent of $SLURM_SUBMIT_DIR
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

mkdir -p logs
ts="$(date +'%Y-%m-%d_%H-%M-%S')"
# --- EDIT THIS for different runs ---
job_name="d121_3c_local"
final_log="logs/${ts}_${job_name}.out"

# Tee all stdout/stderr to a timestamped log file
exec > >(tee -a "$final_log") 2>&1

# ── 1) software stack & venv ────────────────────────────────────────────────
# Activate your local Python environment.
# Adjust the path if your venv is located elsewhere.
VENV_PATH="$REPO_DIR/.venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

# Sanity check: CUDA & torch should be visible
echo "Logging to:   $final_log"
echo "Running on:   $(hostname)"
python -c "import torch, sys; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU Available: {torch.cuda.is_available()}'); sys.exit(0 if torch.cuda.is_available() else 1)"

# ── 2) project-specific environment ─────────────────────────────────────────
# Map cluster paths to your local paths.
# --- IMPORTANT: SET YOUR LOCAL DATA PATH HERE ---
export DATA_ROOT="${DATA_ROOT:-$REPO_DIR/data}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:$REPO_DIR/mlruns}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-SL_Single_MIP}"

# Make the repository importable
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Pin to a specific GPU if you have multiple (e.g., 0, 1, ...).
# You can override this from the command line: CUDA_VISIBLE_DEVICES=1 ./run_local.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "──────────────────────────────────────────────"
echo "Started at:   $(date)"
echo "Repo dir:     $REPO_DIR"
echo "Data root:    $DATA_ROOT"
echo "MLflow URI:   $MLFLOW_TRACKING_URI"
echo "Using GPU:    ${CUDA_VISIBLE_DEVICES}"
echo "──────────────────────────────────────────────"

# ── 3) launch training ──────────────────────────────────────────────────────
# This is the command that runs your training.
# Change the python script and --yaml argument for different experiments.
python train_3c.py --yaml configs/3c/densenet121.yaml

# ── 4) end ──────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "Finished at:  $(date)"
echo "Log saved to $final_log"