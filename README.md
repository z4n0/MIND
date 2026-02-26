# MIND – Medical Imaging for NeuroDegeneration

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-red)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![Platform](https://img.shields.io/badge/Platform-CINECA%20HPC%20%7C%20Local-green)

## Project Overview

This repository contains the research codebase for **Parkinson’s Disease (PD) vs. Multiple System Atrophy (MSA) classification** using CNN models on multi-channel confocal microscopy images of skin biopsies (sweat glands). 

The project implements a complete deep learning pipeline designed for biomedical imaging, incorporating:
- **Supervised Classification:** Baselines using CNNs (DenseNet, ResNet, EfficientNet, ...) and Vision Transformers (ViT).
- **Self-Supervised Learning (SSL):** Pretraining via BYOL, SimSiam, and SimCLR algorithms to leverage unlabeled data to overcome data-scarcity.

- **HPC Integration:** Native support for CINECA clusters via SLURM scheduling.

## Repository Structure

The project follows a modular Object-Oriented architecture compliant with PEP 8 standards.

```text
FOLDER_CINECA/
├── classes/               # Core logic: DatasetFactory, ModelManager, Reader, Trainer
├── clinica/               # Clinical data analysis: Notebooks for EDA and cleaned metadata (CSV)
├── configs/               # YAML configurations containing hyper-parameters for local development
├── configs_cineca/        # YAML configurations containing hyper-parameters optimized for HPC (SLURM)
├── data/                  # Data mount point (Git-ignored)
├── local_run_scripts_DS1/ # Bash scripts for local batch execution (DS1 - Full MIPs)
├── local_run_scripts_DS2/ # Bash scripts for local batch execution (DS2 - Subslices)
├── slurm_files/           # Production SLURM launch scripts
│   ├── 3c/                # 3-Channel Supervised launchers
│   ├── 4c/                # 4-Channel Supervised launchers
│   ├── ssl_pretraining/   # Self-Supervised Learning launchers
│   └── subslice/          # DS2 (Subslice) launchers
├── utils/                 # Utility functions: Transforms, Logging, Visualizations, Metrics
├── train.py               # Entry point for supervised training
├── byol_simsiam_pretraining.py # Entry point for SSL
├── simclr_pretraining.py  # Entry point for SSL
└── README.md              # Project documentation
```

## Setup & Installation

### Prerequisites
- Linux environment (Ubuntu/CentOS)
- NVIDIA GPU with CUDA drivers with at least 12gb VRAM

### Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate
# Install dependencies
pip install -r txt_files/requirements.txt
```

### Environment Variables
Configure the system paths before running experiments:
```bash
export DATA_ROOT="/path/to/data_root"       # Parent directory of 3c_MIP, 4c_MIP
export MLFLOW_TRACKING_URI="file:./mlruns"  # Or remote MLflow server
```

## Datasets & Data Layout

The project utilizes two distinct dataset configurations to assess model robustness:
- **DS1 (Full MIPs):** Whole-image Maximum Intensity Projections. Represents the standard clinical view.
- **DS2 (Subslices):** Cropped sub-regions (tiles) of the original images. Used to increase effective sample size and focus on local textural details.

The pipeline expects data organized as follows under `$DATA_ROOT`:
- **3c_MIP / 4c_MIP**: Labeled data for **DS1** (Class folders: `MSA-P`, `MSA-C`, `PD`, `MSA`).
  - *Note:* 3-channel images exclude the DAPI (gray) channel.
- **SUBSLICE_MIPS**: Labeled data for **DS2** (Subslices).
- **PRETRAINING_***: Unlabeled pools for SSL pretraining.

## Usage

Configuration is entirely driven by **YAML files**, ensuring experiment reproducibility without code modification.

### 1. Supervised Learning (Baseline)
Train a DenseNet121 on 4-channel images:
```bash
python train.py --yaml configs/4c/densenet121.yaml
```
*Features:* Automatic Stratified K-Fold, MLflow logging, Early Stopping.

### 2. Self-Supervised Pretraining (SSL)
Pretrain a ResNet18 encoder using BYOL or SimSiam:
```bash
python byol_simsiam_pretraining.py --yaml configs/ssl/byol_resnet18_3c.yaml
```
*Outputs:* Pretrained encoder weights saved to `pretrained_encoders/`.

### 3. Downstream Fine-Tuning
Fine-tune a pretrained SSL encoder for the classification task:
```bash
python downstream_supervised_fine_tuning.py \
    --yaml configs/ssl_ft/byol_resnet18.yaml \
    --encoder_path pretrained_encoders/encoder_weights.pth \
    --freeze_encoder
```

### 4. Running on HPC (via Slurm)
The `slurm_files/` directory contains production-ready scripts.
```bash
# Submit a job to the cluster
sbatch slurm_files/3c/train_d121_3c.slurm
```
This handles module loading, environment setup, and GPU allocation automatically.

### 5. Local Batch Execution
To streamline local development, the `local_run_scripts_DS1/` and `local_run_scripts_DS2/` directories contain bash scripts (e.g., `run_all_3c.sh`) to execute multiple experiments sequentially.
```bash
# Example: Run all 3-channel baseline models for DS1 sequence
bash local_run_scripts_DS1/run_all_3c.sh
```

## Clinical Data Integration

The `clinica/` directory contains tools for analyzing patient metadata (`Dati clinici finali cleaned.csv`):
- **Data Exploration:** Jupyter notebooks (`clinica_data_exploration.ipynb`) for visualizing demographic distributions (Age, Gender, UPDRS scores).
- **Correlation Analysis:** Scripts to correlate deep learning model predictions with clinical metrics.

## Methodology & Scientific Rigor

- **Data Integrity:** Stratification is performed strictly by **Patient ID** to ensure biological independence between train/validation sets.
- **Augmentation:** Biologically plausible transformations (affine, intensity) implemented via MONAI, preserving biomarker relationships.
- **Metrics:** AUROC, Accuracy, Sensitivity, and Specificity logged per fold and aggregated.
