#!/usr/bin/env bash
set -euo pipefail

# ── 0) repo root & logs ─────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs
ts="$(date +'%Y-%m-%d_%H-%M-%S')"
job_name="vit_pretrained_local"
final_log="logs/${ts}_${job_name}.out"

exec > >(tee -a "$final_log") 2>&1

# ── 1) software stack & venv ────────────────────────────────────────────────
VENV_PATH="$REPO_DIR/.venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

echo "Logging to:   $final_log"
echo "Running on:   $(hostname)"
python -c "import torch, sys; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU Available: {torch.cuda.is_available()}'); sys.exit(0 if torch.cuda.is_available() else 1)"

# ── 2) project-specific environment ─────────────────────────────────────────
export DATA_ROOT="${DATA_ROOT:-$REPO_DIR/data}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:$REPO_DIR/mlruns}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-SL_Pretrained_ViT}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PRETRAINED_WEIGHTS_DIR="${PRETRAINED_WEIGHTS_DIR:-$REPO_DIR/pretrained_weights}"

echo "──────────────────────────────────────────────"
echo "Started at:   $(date)"
echo "Repo dir:     $REPO_DIR"
echo "Data root:    $DATA_ROOT"
echo "MLflow URI:   $MLFLOW_TRACKING_URI"
echo "Using GPU:    ${CUDA_VISIBLE_DEVICES}"
echo "──────────────────────────────────────────────"

# ── 3) launch training ──────────────────────────────────────────────────────
python train_vit.py --yaml configs/pretrained/vit.yaml

# ── 4) end ──────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "Finished at:  $(date)"
echo "Log saved to $final_log"
