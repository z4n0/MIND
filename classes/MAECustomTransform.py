from typing import Dict, List, Tuple, Union

from PIL.Image import Image
from torch import Tensor

from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
# from lightly.transforms.utils import IMAGENET_NORMALIZE


class MAECustomTransform:
    """ Custom transform for MAE pretraining.
    This transform applies a series of transformations to the input image,
    including random resized cropping, horizontal flipping, and normalization.
    args:
        input_size: The size of the input image after resizing.
            If an integer is provided, it will be used for both height and width.
            If a tuple is provided, it should be in the form (height, width).
        min_scale: The minimum scale for the random resized crop.
        normalization_stats_dict: A dictionary containing normalization statistics
            with keys "mean" and "std". If provided, normalization will be applied.
            eg {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    """
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.2,
        normalization_stats_dict: Dict[str, List[float]] = {},
    ):
        transforms = [
            T.RandomResizedCrop(
                input_size, scale=(min_scale, 1.0), interpolation=3
            ),  # 3 is bicubic
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
        if normalization_stats_dict:
            transforms.append(T.Normalize(mean=normalization_stats_dict["mean"], std=normalization_stats_dict["std"]))

        self.transform = T.Compose(transforms)

    def __call__(self, image: Union[Tensor, Image]) -> Tuple[Tensor]:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image. as a tuple containing a single Tensor.

        """
        transformed = self.transform(image)
        if not isinstance(transformed, Tensor):
            transformed = T.ToTensor()(transformed)
        return (transformed,)
