#!/usr/bin/env bash
set -euo pipefail

# ── 0) repo root & logs ─────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs
ts="$(date +'%Y-%m-%d_%H-%M-%S')"
job_name="sequential_3c_runs"
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
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-DS1_3c}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "──────────────────────────────────────────────"
echo "Started at:   $(date)"
echo "Repo dir:     $REPO_DIR"
echo "Data root:    $DATA_ROOT"
echo "MLflow URI:   $MLFLOW_TRACKING_URI"
echo "Using GPU:    ${CUDA_VISIBLE_DEVICES}"
echo "──────────────────────────────────────────────"

# ── 3) define scripts to run ────────────────────────────────────────────────
SCRIPTS_DIR="local_run_scripts_DS1/3c"
scripts=(
    # "run_local_densenet121_3c.sh"
    "run_local_densenet169_3c.sh"
    # "run_local_efficientnetb0_3c.sh"
    # "run_local_efficientnetb3_3c.sh"
    "run_local_resnet18_3c.sh"
    # "run_local_resnet50_3c.sh"
    "run_local_vit_3c.sh"
)

# ── 4) run scripts sequentially ─────────────────────────────────────────────
total_scripts=${#scripts[@]}
successful_runs=0
failed_runs=0

echo "Found $total_scripts scripts to run sequentially"
echo "=================================================="

for i in "${!scripts[@]}"; do
    script_name="${scripts[$i]}"
    script_path="$SCRIPTS_DIR/$script_name"
    script_num=$((i + 1))
    
    echo ""
    echo "──────────────────────────────────────────────"
    echo "Running script $script_num/$total_scripts: $script_name"
    echo "Started at: $(date)"
    echo "──────────────────────────────────────────────"
    
    if [ ! -f "$script_path" ]; then
        echo "ERROR: Script not found: $script_path"
        ((failed_runs += 1))
        continue
    fi
    
    if [ ! -x "$script_path" ]; then
        echo "Making script executable: $script_path"
        chmod +x "$script_path"
    fi
    
    # Run the script and capture its exit code
    if bash "$script_path"; then
        echo "✅ SUCCESS: $script_name completed successfully"
        ((successful_runs += 1))
    else
        echo "❌ FAILED: $script_name failed with exit code $?"
        ((failed_runs += 1))
    fi
    
    echo "Finished at: $(date)"
    echo "──────────────────────────────────────────────"
done

# ── 5) summary ──────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "SEQUENTIAL RUN SUMMARY"
echo "=================================================="
echo "Total scripts:     $total_scripts"
echo "Successful runs:   $successful_runs"
echo "Failed runs:       $failed_runs"
echo "Success rate:      $(( (successful_runs * 100) / total_scripts ))%"
echo "Finished at:       $(date)"
echo "Log saved to:      $final_log"
echo "=================================================="

# Exit with error code if any script failed
if [ $failed_runs -gt 0 ]; then
    echo "⚠️  Some scripts failed. Check the log for details."
    exit 1
else
    echo "�� All scripts completed successfully!"
    exit 0
fi
