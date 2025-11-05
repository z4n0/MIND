#!/usr/bin/env bash
set -euo pipefail

# ── 0) repo root & logs ─────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs
ts="$(date +'%Y-%m-%d_%H-%M-%S')"
job_name="sequential_all_runs"
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
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "──────────────────────────────────────────────"
echo "Started at:   $(date)"
echo "Repo dir:     $REPO_DIR"
echo "Data root:    $DATA_ROOT"
echo "MLflow URI:   $MLFLOW_TRACKING_URI"
echo "Using GPU:    ${CUDA_VISIBLE_DEVICES}"
echo "──────────────────────────────────────────────"

# ── 3) define function to run a script group ────────────────────────────────
run_script_group() {
    set +e 
    local group_name="$1"
    local scripts_dir="$2"
    local mlflow_experiment="$3"
    shift 3
    local scripts=("$@")
    
    export MLFLOW_EXPERIMENT_NAME="$mlflow_experiment"
    
    local total_scripts=${#scripts[@]}
    local successful_runs=0
    local failed_runs=0
    
    echo ""
    echo "=================================================="
    echo "GROUP: $group_name"
    echo "=================================================="
    echo "MLflow Experiment: $mlflow_experiment"
    echo "Found $total_scripts scripts to run sequentially"
    echo "=================================================="
    
    for i in "${!scripts[@]}"; do
        local script_name="${scripts[$i]}"
        local script_path="$scripts_dir/$script_name"
        local script_num=$((i + 1))
        
        echo ""
        echo "──────────────────────────────────────────────"
        echo "[$group_name] Running script $script_num/$total_scripts: $script_name"
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
    
    echo ""
    echo "=================================================="
    echo "[$group_name] SUMMARY"
    echo "=================================================="
    echo "Total scripts:     $total_scripts"
    echo "Successful runs:   $successful_runs"
    echo "Failed runs:       $failed_runs"
    echo "Success rate:      $(( (successful_runs * 100) / total_scripts ))%"
    echo "=================================================="
    
    # Return success/failure counts via global variables
    eval "${group_name}_successful=$successful_runs"
    eval "${group_name}_failed=$failed_runs"
    eval "${group_name}_total=$total_scripts"
    set -e
}

# ── 4) run all script groups sequentially ──────────────────────────────────
overall_successful=0
overall_failed=0
overall_total=0

# Group 1: 3-channel experiments
SCRIPTS_DIR_3C="local_run_scripts_DS1/3c"
scripts_3c=(
    # "run_local_densenet121_3c.sh"
    # "run_local_densenet169_3c.sh"
    # "run_local_resnet18_3c.sh"
    # "run_local_vit_3c.sh"
)
run_script_group "group3C" "$SCRIPTS_DIR_3C" "DS1_3c" "${scripts_3c[@]}"
overall_successful=$((overall_successful + ${group3C_successful:-0}))
overall_failed=$((overall_failed + ${group3C_failed:-0}))
overall_total=$((overall_total + ${group3C_total:-0}))

# Group 2: 4-channel experiments
SCRIPTS_DIR_4C="local_run_scripts_DS1/4c"
scripts_4c=(
    # "run_local_densenet121_4c.sh"
    # "run_local_densenet169_4c.sh"
    # "run_local_resnet18_4c.sh"
    # "run_local_vit_4c.sh"
)
run_script_group "group4C" "$SCRIPTS_DIR_4C" "DS1_4c" "${scripts_4c[@]}"
overall_successful=$((overall_successful + ${group4C_successful:-0}))
overall_failed=$((overall_failed + ${group4C_failed:-0}))
overall_total=$((overall_total + ${group4C_total:-0}))

# Group 3: Pretrained experiments
SCRIPTS_DIR_PRETRAINED="local_run_scripts_DS1/pretrained"
scripts_pretrained=(
    "run_local_vit_pretrained_3c.sh" 
    "run_local_resnet18_pretrained.sh"
    "run_local_densenet121_pretrained.sh"
    "run_local_densenet169_pretrained.sh"
    # run_local_vit_small_patch16_224_pretrained.sh
    # run_local_deit_small_patch16_224_pretrained.sh
    # run_local_deit_base_patch16_224_pretrained.sh
    # run_local_vit_base_patch16_224_pretrained.sh
    # run_local_vit_base_patch16_384_pretrained.sh
)
run_script_group "groupPRETRAINED" "$SCRIPTS_DIR_PRETRAINED" "DS1_pretrained" "${scripts_pretrained[@]}"
overall_successful=$((overall_successful + ${groupPRETRAINED_successful:-0}))
overall_failed=$((overall_failed + ${groupPRETRAINED_failed:-0}))
overall_total=$((overall_total + ${groupPRETRAINED_total:-0}))

# ── 5) final summary ────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "OVERALL SEQUENTIAL RUN SUMMARY"
echo "=================================================="
echo "Total scripts (all groups):     $overall_total"
echo "Successful runs:                $overall_successful"
echo "Failed runs:                    $overall_failed"
if [ $overall_total -gt 0 ]; then
    echo "Success rate:                   $(( (overall_successful * 100) / overall_total ))%"
else
    echo "Success rate:                   0%"
fi
echo "Finished at:                    $(date)"
echo "Log saved to:                   $final_log"
echo "=================================================="

# Exit with error code if any script failed
if [ $overall_failed -gt 0 ]; then
    echo "⚠️  Some scripts failed. Check the log for details."
    exit 1
else
    echo "✅ All scripts completed successfully!"
    exit 0
fi