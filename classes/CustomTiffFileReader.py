import tifffile
from monai.transforms import Compose, LambdaD, ScaleIntensityd, EnsureTyped
from monai.transforms.transform import MapTransform

class CustomTiffFileReader(MapTransform):
    """
    Loads an image from a TIFF file path using tifffile.imread().

    Args:
        data (dict): A dictionary with at least the key "image" containing the
            file path to a TIFF image. Example:
                {
                    "image": <str: path_to_tiff>,
                    "label": <int: label> #optional
                }
    Usage: CustomTiffFileReader(keys=["image"])
    Returns:
        dict: The input dictionary with the "image" key replaced by the loaded
            image as a numpy ndarray.

            - "image": np.ndarray, shape (C, H, W), dtype=uint8
                C = 3 (GBR) or 4 (GBGrR) channel order.

    Note:
        The images are loaded with channels in the order GBR or GBGrR.
    """

    def __call__(self, data: dict) -> dict:
        """
        Load a TIFF image from the file path in data["image"].

        Args:
            data (dict): Input dictionary with "image" key as file path.

        Returns:
            dict: Dictionary with "image" key as np.ndarray of shape (C, H, W),
                  dtype=uint8, channel order GBR or GBGrR.
        """
        d = dict(data) #shallow copy of the input dictionary to avoid mutating the original input
        path: str = d["image"] #get the path to the image
        img_array = tifffile.imread(path)  # shape: (C, H, W), dtype=uint8
        # print(f"img array type: {type(img_array)}") #np.ndarray
        d["image"] = img_array #replace the image path with the image array
        return d