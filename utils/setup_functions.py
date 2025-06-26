import torch


def set_environment_flags():
    """
    Detects the current environment and sets flags for Google Drive, Linux, and Kaggle.
    Returns a dictionary with the environment flags.
    """
    import platform
    import os

    flags = {
        "gdrive": False,
        "linux": platform.system() == "Linux",
        "kaggle": "KAGGLE_URL_BASE" in os.environ,
        "ssl": True,  # Assuming this is always True
    }

    # Detect if running in Google Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        flags["gdrive"] = True
    except ImportError:
        flags["gdrive"] = False

    print("Environment settings:", flags)
    return flags


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
import os
import glob
import numpy as np

def get_tif_image_paths_from_folder(folder_path: str) -> list[str]:
    """
    Scans a directory for .tif files and returns a list of their full paths.

    Args:
        folder_path: The path to the directory to scan.

    Returns:
        A list of strings, where each string is the full path to a .tif file.
        Returns an empty list if the folder doesn't exist or no .tif files are found.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return []
    
    tif_files = glob.glob(os.path.join(folder_path, "*.tif"))
    if not tif_files:
        print(f"No .tif files found in '{folder_path}'")
    return tif_files

## Paths of ALL images into a numpy array without labels used for SSL
def from_tif_folder_to_np_paths_array(folder_path: str) -> np.ndarray:
    """
    Load all .tif images from a folder into a numpy array.
    """
    image_paths = glob.glob(os.path.join(folder_path, "*.tif"))
    image_paths_np = np.array(image_paths)
    print(f"Number of images in {folder_path}: {len(image_paths)}")
    return image_paths_np