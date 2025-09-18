#!/bin/bash
###############################################################################
#  SimCLR self-supervised pre-training  (ResNet-18, 3-channel MIPs)
#  $ sbatch simclr_resnet18.slurm
###############################################################################
#SBATCH --job-name=ssl_resnet18_3c      # appears in squeue / log filename
#SBATCH --account=pMI24_EleBr_1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1                    # 1 × A100 on "boost"
#SBATCH --cpus-per-task=8               # dataloader / augmentations
#SBATCH --mem=32G                       # system RAM
#SBATCH --time=01:55:00                 # hh:mm:ss
# ––– temporary log file (renamed below) ––––––––––––––––––––––––––––––––––
#SBATCH --output=/tmp/%x-%j.tmp.out

# ───────────────────── 1) software stack & venv ───────────────────────────
module load profile/deeplrn
module load cineca-ai/4.3.0             # CUDA 12 + PyTorch 2.2 stack
source "$HOME/venvs/tesi/bin/activate"  # adjust if you use another env

# ───────────────────── 2) project-specific env vars ───────────────────────
export DATA_ROOT="$WORK/lzanotto/FOLDER_CINECA/data"
echo "DATA_ROOT will be: $WORK/lzanotto/FOLDER_CINECA/data"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"    # make code importable

# ───────────────────── 3) rename Slurm log with timestamp ─────────────────
ts=$(date +'%Y-%m-%d_%H-%M-%S')                 # e.g. 2025-06-30_21-18-05
log_dir="$SLURM_SUBMIT_DIR/logs"
mkdir -p "$log_dir"

tmp_log="/tmp/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.tmp.out"
final_log="$log_dir/${ts}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"

# move the file that Slurm already opened for us
mv "$tmp_log" "$final_log" 2>/dev/null || true

# keep appending to the renamed file
exec > >(tee -a "$final_log") 2>&1

echo "WORK variable: $WORK"
echo "DATA_ROOT will be: $WORK/lzanotto/FOLDER_CINECA/data"
echo "Logging to:   $final_log"
echo "Running on:   $(hostname)"
echo "CUDA visible: $CUDA_VISIBLE_DEVICES"
echo "Started at:   $(date)"
echo "──────────────────────────────────────────────────────────────"

# ───────────────────── 4) launch SimCLR pre-training  ─────────────────────
cd "$SLURM_SUBMIT_DIR"          # project root (FOLDER_CINECA)

YAML=configs_cineca/ssl/simclr_resnet18_3c.yaml   # YAML with ResNet18 SimCLR options

# srun inherits the GPU/CPU allocation granted by Slurm
srun python simclr_pretraining.py --yaml "$YAML"

echo "Finished at:  $(date)"
