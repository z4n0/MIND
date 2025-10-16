#!/usr/bin/env bash
set -euo pipefail

# ── 0) repo root & logs ─────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs
ts="$(date +'%Y-%m-%d_%H-%M-%S')"
job_name="sequential_pretrained_4c_runs"
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
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-DS1_pretrained_4C}"
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

# ── 3) define configs to run ────────────────────────────────────────────────
# All 4-channel pretrained models using timm library with ImageNet weights
configs=(
    "configs/pretrained_timm_4c/resnet18.yaml"
    # "configs/pretrained_timm_4c/resnet50.yaml"
    "configs/pretrained_timm_4c/densenet121.yaml"
    "configs/pretrained_timm_4c/densenet169.yaml"
    "configs/pretrained_timm_4c/vit_small_patch16_224.yaml"
    # "configs/pretrained_timm_4c/vit_base_patch16_224.yaml"
    # "configs/pretrained_timm_4c/deit_small_patch16_224.yaml"
    # "configs/pretrained_timm_4c/deit_base_patch16_224.yaml"
    # "configs/pretrained_timm_4c/vit_base_patch16_384.yaml"
)

# ── 4) run configs sequentially ─────────────────────────────────────────────
total_configs=${#configs[@]}
successful_runs=0
failed_runs=0

echo "Found $total_configs 4-channel pretrained configs to run sequentially"
echo "=================================================="

for i in "${!configs[@]}"; do
    config_path="${configs[$i]}"
    config_name=$(basename "$config_path" .yaml)
    config_num=$((i + 1))
    
    echo ""
    echo "──────────────────────────────────────────────"
    echo "Running config $config_num/$total_configs: $config_name"
    echo "Config path: $config_path"
    echo "Started at: $(date)"
    echo "──────────────────────────────────────────────"
    
    if [ ! -f "$config_path" ]; then
        echo "ERROR: Config not found: $config_path"
        ((failed_runs += 1))
        continue
    fi
    
    # Run the training and capture its exit code
    if python train_pretrained.py --yaml "$config_path"; then
        echo "✅ SUCCESS: $config_name completed successfully"
        ((successful_runs += 1))
    else
        echo "❌ FAILED: $config_name failed with exit code $?"
        ((failed_runs += 1))
    fi
    
    echo "Finished at: $(date)"
    echo "──────────────────────────────────────────────"
done

# ── 5) summary ──────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "SEQUENTIAL RUN SUMMARY - 4-CHANNEL PRETRAINED"
echo "=================================================="
echo "Total configs:     $total_configs"
echo "Successful runs:   $successful_runs"
echo "Failed runs:       $failed_runs"
echo "Success rate:      $(( (successful_runs * 100) / total_configs ))%"
echo "Finished at:       $(date)"
echo "Log saved to:      $final_log"
echo "=================================================="
echo ""
echo "Models tested:"
echo "  - CNNs: ResNet18, ResNet50, DenseNet121, DenseNet169"
echo "  - ViTs: ViT-Small, ViT-Base (224 & 384), DeiT-Small, DeiT-Base"
echo "  - All with 4-channel ImageNet-pretrained weights via timm"
echo "=================================================="

# Exit with error code if any config failed
if [ $failed_runs -gt 0 ]; then
    echo "⚠️  Some configs failed. Check the log for details."
    exit 1
else
    echo "✅ All 4-channel pretrained configs completed successfully!"
    exit 0
fi

