# utils/directory_functions.py
import os
from pathlib import Path
from typing import Tuple

# !NOTE remeber to export these variables in your SLURM script!
# export DATA_ROOT=$WORK/lzanotto/data
# export PROJ_ROOT=$SLURM_SUBMIT_DIR         # dove si trova train.py
# export MLFLOW_TRACKING_URI=file:$WORK/lzanotto/mlruns


def _check_env(var: str) -> str:
    """Return the env-variable or raise a readable error."""
    v = os.getenv(var)
    if not v:
        raise RuntimeError(
            f"{var} is not defined. Export it in your SLURM script, e.g.\n"
            f'  export {var}=$WORK/lzanotto/data'
        )
    return v

# ----------------------------------------------------------------------------
#  MAIN UTILITIES (CINECA-only)
# ----------------------------------------------------------------------------
def get_data_directory(num_input_channels: int = 3) -> Path:
    """
    Return the directory that contains the raw images on CINECA.

    DATA_ROOT/
        ├─ 3c_MIP/   (for 3-channel)
        └─ 4c_MIP/   (for 4-channel)
    """
    root = Path(_check_env("DATA_ROOT"))
    sub  = "3c_MIP" if num_input_channels == 3 else "4c_MIP"
    data_dir = root / sub
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Expected dataset dir {data_dir} not found")
    return data_dir


def get_base_directory() -> Path:
    """
    Base directory for configs, checkpoints… (= project root on CINECA)
    """
    return Path(_check_env("PROJ_ROOT"))   # set in SLURM: export PROJ_ROOT=$SLURM_SUBMIT_DIR


def get_tracking_uri() -> str:
    """
    Return the MLflow tracking URI (file-store).
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI not defined – export it in the SLURM script, e.g.\n"
            "  export MLFLOW_TRACKING_URI=file:$WORK/lzanotto/mlruns"
        )
    return uri


# Helper that returns both at once (if you still need it)
def get_data_and_base_directory(num_input_channels: int = 3) -> Tuple[Path, Path]:
    return get_data_directory(num_input_channels), get_base_directory()

def get_yaml_path(num_input_channels: int = 3, model_name: str = "densenet121") -> Path:
    """
    Return the path to the YAML config file.
    """
    return get_base_directory() / "configs" / f"{num_input_channels}c" / f"{model_name.lower()}.yaml"
