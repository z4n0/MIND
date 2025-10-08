import random, os, numpy as np, torch
from monai.utils.misc import set_determinism
import torch.nn as nn
import hashlib
from typing import Tuple

def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, NumPy and PyTorch (CPU & all GPUs)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)         # for multi-GPU

    if deterministic:                         # make CUDA kernels repeatable
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True, warn_only=True) # this was not used, it's probably an overkill, the variations created by non deterministic algorithms are usually very small
        # algorithms are negligibly different from the deterministic ones, but they are not exactly the same
        
# ───────────────────────────────────────────────────────────────────────────────
# 1.  Bit-wise equality (strict)
# ───────────────────────────────────────────────────────────────────────────────
def models_equal_strict(model_a: nn.Module, model_b: nn.Module) -> bool:
    """
    Return True iff every parameter tensor is *bit-wise* identical.

    Raises
    ------
    ValueError
        If the state_dict keys differ (i.e. architectures not matching).
    """
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()

    if sd_a.keys() != sd_b.keys():
        raise ValueError("Models have different parameter sets.")

    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a.keys())


# ───────────────────────────────────────────────────────────────────────────────
# 2.  Hash-based check (fast, tolerant to ordering)
# ───────────────────────────────────────────────────────────────────────────────
def model_checksum(model: nn.Module) -> Tuple[str, int]:
    """
    Compute an MD5 hash of all parameters concatenated.
    Returns (hex_digest, total_bytes).
    """
    md5 = hashlib.md5()
    total = 0
    with torch.no_grad():
        for p in model.parameters():
            t = p.detach().cpu().numpy().tobytes()
            md5.update(t)
            total += len(t)
    return md5.hexdigest(), total


def models_equal_hash(model_a: nn.Module, model_b: nn.Module) -> bool:
    """
    Compare models via MD5 checksum of raw bytes (bit-wise).
    Safer when state_dict key order might differ.
    """
    return model_checksum(model_a)[0] == model_checksum(model_b)[0]

