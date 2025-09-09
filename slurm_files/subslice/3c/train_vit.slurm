#!/bin/bash
#SBATCH --job-name=vit_3c_subslice
#SBATCH --account=pMI24_EleBr_1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# --- 1) temporary log file -----------------------------------------------
# At submission time SLURM doesn’t know the date, so we log to /tmp first
#SBATCH --output=/tmp/%x-%j.tmp.out

# ───────────────────────────────────────────────────────────────────────────
# 2) software stack & venv
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$HOME/venvs/tesi/bin/activate"

# 3) project-specific environment
export DATA_ROOT="$WORK/lzanotto/FOLDER_CINECA/data/SUBSLICE_MIPS/"
export MLFLOW_TRACKING_URI="file:$WORK/lzanotto/FOLDER_CINECA/mlruns"
export MLFLOW_EXPERIMENT_NAME="SL_ViT_MIP"

# make your code (FOLDER_CINECA) importable
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# --- 4) rename log with timestamp -----------------------------------------
ts=$(date +'%Y-%m-%d_%H-%M-%S')
log_dir="$SLURM_SUBMIT_DIR/logs"
mkdir -p "$log_dir"

tmp_log="/tmp/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.tmp.out"
final_log="$log_dir/${SLURM_JOB_NAME}_${ts}_${SLURM_JOB_ID}.out"

# move the file that SLURM already opened for us
mv "$tmp_log" "$final_log" 2>/dev/null || true
# Re-establish the link to the new log file for all subsequent output
exec > >(tee -a "$final_log") 2>&1

# --- 5) launch training ---------------------------------------------------
echo "--- START ---"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Submitted from: $SLURM_SUBMIT_DIR"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"
echo "--------------------"

srun python "$SLURM_SUBMIT_DIR/train_vit.py" --yaml "configs/3c/vit.yaml"

echo "--- END ---"