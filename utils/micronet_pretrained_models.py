from efficientnet_pytorch import EfficientNet  # pip install efficientnet_pytorch
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as tv_models
from pathlib import Path
import os
# Make sure to install the pretrainedmodels package:
# pip install pretrainedmodels
import pretrainedmodels

# Assume this helper is defined somewhere in your code:
def get_pretrained_microscopynet_url(model_name, pretrained_weights):
    """
    Returns the URL for downloading the pretrained MicroNet weights.
    This function should construct the URL based on the model name and weight type.
    """
    url_base = 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/'
    url_end = '_v1.0.pth.tar'
    return url_base + f'{model_name}_pretrained_{pretrained_weights}' + url_end

def hasattr_nonnone(obj, attr: str) -> bool:
    """
    Returns True if 'obj' has an attribute named 'attr' that is not None.
    """
    return hasattr(obj, attr) and (getattr(obj, attr) is not None)

def get_classifier_name(model: nn.Module) -> str:
    """
    Returns the name of the final classification layer for 'model'
    by checking known attributes in a priority order, while ensuring
    we skip any attribute that is None.
    
    This covers:
      - EfficientNet or similar that store final linear in '_fc'
      - pretrainedmodels ResNet that uses 'last_linear'
      - torchvision ResNet that uses 'fc'
      - DenseNet/VGG (and some DPN) that use 'classifier'
    """
    # 1) EfficientNet
    if hasattr_nonnone(model, '_fc'):
        return '_fc'
    # 2) pretrainedmodels ResNet, Inception, Xception, SENet, etc.
    elif hasattr_nonnone(model, 'last_linear'):
        return 'last_linear'
    # 3) torchvision ResNets (or if pretrainedmodels left fc non-None)
    elif hasattr_nonnone(model, 'fc'):
        return 'fc'
    # 4) DenseNet, VGG, some DPN, etc.
    elif hasattr_nonnone(model, 'classifier'):
        return 'classifier'
    else:
        raise ValueError("Could not find a recognized final classifier "
                         "attribute (_fc, last_linear, fc, classifier).")


def get_classifier_attr(model: nn.Module) -> str:
    """
    Automatically identifies the name of the attribute containing the final
    classification layer of the model.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        str: The attribute name of the final classifier layer.

    Raises:
        ValueError: If no known classifier attribute is found in the model.
    """
    candidates = ['fc', 'classifier', 'last_linear', '_fc']
    for name in candidates:
        if hasattr(model, name):
            return name
    raise ValueError(
        f"Could not find a known final classifier layer in {model.__class__.__name__}"
    )


def get_nasa_pretrained_model(
    model_name: str,
    num_classes: int,
    pretrained_weights: str = 'microscopynet',
    map_location: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    Instantiates the architecture `model_name`, downloads the checkpoint
    specified by `pretrained_weights` from get_pretrained_microscopynet_url,
    and replaces the final classifier layer to output `num_classes` classes.

    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet50').
        num_classes (int): Number of output classes for the classifier.
        pretrained_weights (str): Which pretrained weights to use.
        map_location (torch.device): Device to map the loaded weights.

    Returns:
        nn.Module: The model with loaded weights and updated classifier.

    Raises:
        ValueError: If the model name is not valid or the classifier layer
            type is not supported.
    """
    available_nets = [
        'densenet121', 'densenet161', 'densenet169', 'densenet201',
        'dpn107', 'dpn131', 'dpn68', 'dpn68b', 'dpn92', 'dpn98',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        'inceptionresnetv2', 'inceptionv4', 'mobilenet_v2',
        'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
        'resnext101_32x8d', 'resnext50_32x4d',
        'se_resnet101', 'se_resnet152', 'se_resnet50',
        'se_resnext101_32x4d', 'se_resnext50_32x4d',
        'senet154', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'xception'
    ]

    model_name = model_name.lower()
    assert model_name in available_nets, f"{model_name} is not a valid model to use nasa pretrained weights"
    if not hasattr(tv_models, model_name):
        raise ValueError(f"{model_name} is not a valid model in torchvision.models")

    # Get the constructor for the model defined by model_name
    constructor = getattr(tv_models, model_name)
    # 1) Instantiate model without default weights
    try:
        model = constructor(weights=None)         # torchvision ≥0.13
    except TypeError:
        model = constructor(pretrained=False)     # fallback for legacy versions

    # --- LOCAL FIRST -------------------------------------------------------
    # Construct expected local filename: <model>_pretrained_<pretrained_weights>_v1.0.pth.tar
    fname = f"{model_name}_pretrained_{pretrained_weights}_v1.0.pth.tar"
    local_dir = os.environ.get("PRETRAINED_WEIGHTS_DIR", "") #get the local directory
    local_path = Path(local_dir) / fname if local_dir else Path(fname)
    
    # Check if the local weights file exists
    if local_path.exists():
        print(f"[microscopy] Loading local weights: {local_path}")
        state_dict = torch.load(local_path, map_location=map_location)
        # unwrap if checkpoint saved as {'state_dict': {...}}
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    else:# --- DOWNLOAD IF LOCAL NOT FOUND ---------------------------------------
        print(f"[microscopy] Local file not found: {local_path} -> downloading")
        url = get_pretrained_microscopynet_url(model_name, pretrained_weights)
        state_dict = load_state_dict_from_url(url, progress=True, map_location=map_location)
    
    model.load_state_dict(state_dict, strict=False)  # strict=False to allow partial loading
    model.eval()

    # 3) Replace the final classifier layer
    attr = get_classifier_attr(model) # get the attribute name of the final classifier layer
    old = getattr(model, attr) # get the final classifier layer Object

    if isinstance(old, nn.Linear):
        new = nn.Linear(old.in_features, num_classes)
    elif isinstance(old, nn.Sequential): # if the final classifier layer is a Sequential object
        layers = list(old) # convert the Sequential object to a list
        last = layers[-1] # get the last element of the list
        if not isinstance(last, nn.Linear): # if the last element is not a Linear object
            raise ValueError(f"Cannot handle final layer of type {type(last)} in Sequential")
        layers[-1] = nn.Linear(last.in_features, num_classes) # replace the last element with a new Linear object
        new = nn.Sequential(*layers) # convert the list back to a Sequential object
    else:
        raise ValueError(f"Final classifier layer of type {type(old)} is not supported")

    setattr(model, attr, new) # set the new classifier layer to the model
    return model


def get_pretrained_model(
    model_name: str = 'resnet50',
    num_classes: int = 2,
    pretrained_weights: str = 'microscopynet'
) -> nn.Module:
    """
    Returns a model with architecture 'model_name', loaded (if possible) from
    MicroNet pretrained weights, and with the final classifier replaced by a
    new layer producing 'num_classes' outputs.

    :param model_name: Name of the model architecture
    :param num_classes: Number of classes for the classification head
    :param pretrained_weights: Identifier for the MicroNet pretrained weights
    :return: A PyTorch model
    """
    chosen_network_name = model_name.lower()
    print(f"Loading model: {chosen_network_name}")

    # Architectures that typically load from pretrainedmodels or fallback to PyTorch Hub
    available_nets = [
        'densenet121', 'densenet161', 'densenet169', 'densenet201',
        'dpn107', 'dpn131', 'dpn68', 'dpn68b', 'dpn92', 'dpn98',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        'inceptionresnetv2', 'inceptionv4', 'mobilenet_v2',
        'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
        'resnext101_32x8d', 'resnext50_32x4d',
        'se_resnet101', 'se_resnet152', 'se_resnet50',
        'se_resnext101_32x4d', 'se_resnext50_32x4d',
        'senet154', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'xception'
    ]

    # 1. Special-case for EfficientNet from 'efficientnet_pytorch'
    if 'efficientnet' in chosen_network_name:
        model = EfficientNet.from_name(chosen_network_name)

    # 2. If it's in the known list from pretrainedmodels, load from there
    elif chosen_network_name in available_nets:
        try:
            model = pretrainedmodels.__dict__[chosen_network_name](
                num_classes=1000,
                pretrained=None
            )
        except KeyError:
            raise ValueError(
                f"Architecture {chosen_network_name} not found in pretrainedmodels."
            )
    else:
        raise ValueError(f"Model {chosen_network_name} not found in pretrainedmodels.")
    # else:
    #     # 3. Fallback to PyTorch Hub (torchvision)
    #     print(
    #         f"Model {chosen_network_name} not found in pretrainedmodels. "
    #         "Loading from PyTorch Hub instead."
    #     )
    #     model = torch.hub.load(
    #         'pytorch/vision:v0.10.0', 
    #         model_name, 
    #         weights=None
    #     )

    print(f"Loaded model: {model_name} from {model.__class__.__name__}")
    #capitalized_model_name = model_name.capitalize()
    # Download and load MicroNet pretrained weights
    
    # 1) Istanzio ResNet50 senza pesi predefiniti
    model = resnet50(weights=None)  

    # 2) Scarico lo state_dict custom
    url = get_pretrained_microscopynet_url("resnet50", "microscopynet")
    state_dict = load_state_dict_from_url(
        url,
        progress=True,
        map_location=torch.device("cpu")  # o "cuda" se preferisci
    )

    # 3) Carico i pesi e metto in eval
    model.load_state_dict(state_dict)
    model.eval()

    print("MicroscopyNet pronto su ResNet50!")  
    # url = get_pretrained_microscopynet_url(chosen_network_name, pretrained_weights)
    # print(f"Loading weights from: {url}")
    # state_dict = model_zoo.load_url(url)
    # model.load_state_dict(state_dict, strict=False)
    # print("Model weights loaded successfully.")
    # print("Model structure before classifier replacement:\n", model)

    # Identify which attribute is the final classifier
    classifier_name = get_classifier_name(model)
    print(f"Using the '{classifier_name}' attribute for final classification.")

    # Replace the final classifier
    if classifier_name == '_fc':
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)

    elif classifier_name == 'last_linear':
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)

    elif classifier_name == 'fc':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif classifier_name == 'classifier':
        # e.g., DenseNet, VGG, DPN
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            # e.g., DPN might store a Conv2d layer
            raise ValueError(
                "Classifier is not a Linear layer. Please adapt for architectures "
                "that use a non-Linear final layer (like DPN)."
            )

    else:
        raise ValueError(
            "Unexpected classifier attribute name. "
            "Please extend 'get_classifier_name' to handle it."
        )

    # print("Model structure after classifier replacement:\n", model)
    return model

# --- Example calls to load the desired models ---

# # 1. SE-ResNet50 (available via torch.hub)
# model_se_resnet50 = get_pretrained_model(model_name='se_resnet50', num_classes=2, pretrained_weights='microscopynet')
# print("Loaded SE-ResNet50")

# # 2. InceptionV4 (requires pretrainedmodels; install via: pip install pretrainedmodels)
# model_inceptionv4 = get_pretrained_model(model_name='inceptionv4', num_classes=2, pretrained_weights='microscopynet')
# print("Loaded InceptionV4")

# # 3. Xception (requires pretrainedmodels; install via: pip install pretrainedmodels)
# model_xception = get_pretrained_model(model_name='xception', num_classes=2, pretrained_weights='microscopynet')
# print("Loaded Xception")

# eg resnet50
# resnet50 = get_pretrained_model(model_name='resnet50', num_classes=cfg.model["out_channels"], pretrained_weights='microscopynet')

# 2) resnet50
# from torchvision.models import resnet50, ResNet50_Weights
    
# # Initialize with no pretrained weights
# model = resnet50(weights=None)
# # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
# url = get_pretrained_microscopynet_url('resnet50', 'microscopynet')
# model.load_state_dict(model_zoo.load_url(url))
# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, cfg.model["out_channels"])  