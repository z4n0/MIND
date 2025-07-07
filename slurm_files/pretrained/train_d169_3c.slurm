#!/bin/bash
#SBATCH --job-name=densenet169_3c_tesi
#SBATCH --account=pMI24_EleBr_1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:55:00

# --- 1) temporary log file -----------------------------------------------
# At submission time SLURM doesn’t know the date, so we log to /tmp first
#SBATCH --output=/tmp/%x-%j.tmp.out

# ───────────────────────────────────────────────────────────────────────────
# 2) software stack & venv
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$HOME/venvs/tesi/bin/activate"

# 3) project-specific environment
export DATA_ROOT="$WORK/lzanotto/FOLDER_CINECA/data"
export MLFLOW_TRACKING_URI="file:$WORK/lzanotto/FOLDER_CINECA/mlruns"
export MLFLOW_EXPERIMENT_NAME="tesi_cv"

# make your code (FOLDER_CINECA) importable
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# --- 4) rename log with timestamp -----------------------------------------
ts=$(date +'%Y-%m-%d_%H-%M-%S')                  # e.g. 2025-06-19_14-42-33
log_dir="$SLURM_SUBMIT_DIR/logs"
mkdir -p "$log_dir"

tmp_log="/tmp/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.tmp.out"
final_log="$log_dir/${SLURM_JOB_NAME}_${ts}_${SLURM_JOB_ID}.out"

# move the file that SLURM already opened for us
mv "$tmp_log" "$final_log" 2>/dev/null || true

# optionally keep writing into the renamed file
exec > >(tee -a "$final_log") 2>&1

echo "Logging to:   $final_log"
echo "Running on:   $(hostname)"
echo "CUDA visible: $CUDA_VISIBLE_DEVICES"
echo "Started at:   $(date)"
echo "──────────────────────────────────────────────"

# --- 5) launch training ----------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
srun python train_pretrained.py --yaml configs/pretrained/densenet169.yaml
