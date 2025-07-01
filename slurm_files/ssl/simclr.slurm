#!/bin/bash
###############################################################################
#  SimCLR self-supervised pre-training (DenseNet-121, 3-channel MIPs)
#  $ sbatch slurm_simclr_3c.slurm
###############################################################################
#SBATCH --job-name=ssl_d121_3c          # appears in squeue / *.out filename
#SBATCH --account=pMI24_EleBr_1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1                    # 1× A100 on “boost” queue
#SBATCH --cpus-per-task=4               # dataloader workers / augmentations
#SBATCH --mem=32G                       # system RAM
#SBATCH --time=00:59:00                 # hh:mm:ss  (adjust as needed)
#SBATCH --output=%x-%j.out              # e.g. ssl_d121_3c-1234567.out
#-----------------------------------------------------------------------------#
# 1) software modules
module load profile/deeplrn
module load cineca-ai/4.3.0             # CUDA 12  + PyTorch 2.2 stack

# 2) private virtual-env
source "$HOME/venvs/tesi/bin/activate"  # ← change if you use another env

# 3) project-specific environment vars
export DATA_ROOT=$WORK/lzanotto/data                # 3c_MIP / 4c_MIP live here
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH     # make code import-able

# 4) launch SimCLR pre-training
cd "$SLURM_SUBMIT_DIR"                              # project root (FOLDER_CINECA)

# YAML with **only** SimCLR options (no supervised-only keys)
YAML=configs/simclr_densenet_3c.yaml

# srun ensures the job inherits the GPU/CPU allocation from SLURM
srun python simclr_pretrain.py --yaml "$YAML"
