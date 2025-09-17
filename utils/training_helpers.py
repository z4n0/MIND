import argparse
import re
import numpy as np
import shutil
from pathlib import Path
import os

# ───────────────────── CLI ────────────────────────────────────────────────

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="path inside configs/")
    return p.parse_args()

# ───────────────────── helpers ─────────────────────────────────────────────
def extract_patient_id(path: str) -> str:
    m = re.search(r'(\d{4})', path)
    return m.group(1) if m else "UNKNOWN"

def best_fold_idx(results, metric="test_balanced_acc") -> int:
    return int(np.argmax([r[metric] for r in results]))

def best_fold_idx_patient(results, metric="patient_major_bal_acc") -> int:
    return int(np.argmax([r[metric] for r in results]))

def get_data_directory(num_input_channels: int) -> Path:
    """
    Restituisce la cartella che contiene le immagini a seconda
    del numero di canali (3 → 3c_MIP, 4 → 4c_MIP).

    Richiede che la variabile d'ambiente DATA_ROOT sia già definita.
    """
    base = Path( os.environ.get("DATA_ROOT", "") )
    if not base.exists():
        raise EnvironmentError(
            "DATA_ROOT non impostata o path inesistente. "
            "Esegui `export DATA_ROOT=...` oppure modifica il tuo SLURM."
        )

    sub = {3: "3c_MIP", 4: "4c_MIP"}.get(num_input_channels)
    if sub is None:
        raise ValueError("num_input_channels deve essere 3 o 4")

    data_dir = base / sub
    if not data_dir.exists():
        raise FileNotFoundError(f"Cartella dati non trovata: {data_dir}")

    return data_dir

# ───────────────────── cleanup helper ──────────────────────────────────────
def cleanup_run_dir(run_dir: Path) -> None:
    """Remove the given run directory and ignore errors.

    Useful to auto-delete per-run artifacts after MLflow logging.
    """
    try:
        print(f"Cleaning up run directory: {run_dir}")
        shutil.rmtree(run_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: failed to remove {run_dir}: {e}")