from utils.micronet_pretrained_models import get_nasa_pretrained_model
from torch import nn
import torch
import torch.utils.model_zoo

def _check_model_name_match(model, cfg):
    """
    Verifies that the model class name corresponds to the expected model name
    defined in the configuration. Allows partial matches, but raises an error
    if there is a clear mismatch.

    Args:
        model (nn.Module): The PyTorch model instance.
        cfg (object): Configuration object with get_model_name() method.

    Raises:
        ValueError: If there is a mismatch between expected and actual model name.
    """
    model_name_actual = model.__class__.__name__.lower()
    model_name_expected = cfg.get_model_name().lower()

    # Allow a match if either is a substring of the other
    if model_name_actual in model_name_expected or model_name_expected in model_name_actual:
        return  # names are close enough

    raise ValueError(
        f"WARNING: Model name mismatch. Expected '{model_name_expected}', "
        f"got '{model_name_actual}' as inferred from the model class. "
        f"Please check your YAML configuration file."
    )

def setup_model(cfg):
    """
    Set up the model for training.

    Args:
        cfg.model (dict): Dictionary containing model parameters.
            - spatial_dims (int): Number of spatial dimensions (2D or 3D).
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels (classes).
        cfg.training (dict): Dictionary containing training parameters.
            - pretrained (bool): Whether to use a pretrained model.
            - distributed (bool, optional): Whether to use distributed data parallelism.

    Returns:
        model (torch.nn.Module): The initialized model.
        device (torch.device): The device on which the model is located (CPU or GPU).
    """
    model = None  # Initialize model to None

    # Check CUDA availability first
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    else:
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPU{'s' if n_gpus > 1 else ''}.")
        device = torch.device("cuda:0")

    # Create model instance
    #---------------
    # model = SimpleMONAICNN(
    #     cfg.model["in_channels"],
    #     cfg.model["out_channels"]
    #     )
    
    #----------------------
    from monai.networks.nets import DenseNet169, DenseNet121, DenseNet201
    # #from torchvision.models import DenseNet121_Weights
    # model = DenseNet169(
    #     spatial_dims=cfg.model["spatial_dims"],
    #     in_channels=cfg.model["in_channels"],
    #     out_channels=cfg.model["out_channels"],
    #     pretrained= cfg.training["transfer_learning"],
    #     dropout_prob=cfg.model["dropout_prob"],
    #     )
    # from torchvision.models import resnet50, ResNet50_Weights
    
    # Initialize with no pretrained weights
    # model = resnet50(weights=None)
    # # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    # url = get_pretrained_microscopynet_url('resnet50', 'microscopynet')
    # model.load_state_dict(model_zoo.load_url(url))
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, cfg.model["out_channels"])  
    model = get_nasa_pretrained_model(model_name='resnet18', num_classes=2, pretrained_weights='microscopynet')
    # safety check to see if yaml file corresponds to the loaded model
    _check_model_name_match(model, cfg)
    # SE resnet50---------------
    # from monai.networks.nets import SEResNet101
    # model = SEResNet101(
    #     spatial_dims=cfg.model["spatial_dims"],
    #     in_channels=cfg.model["in_channels"],
    #     num_classes=cfg.model["out_channels"],
    #     pretrained=cfg.training["pretrained"],
    # )
        
     # Initialize with no pretrained weights resnet50
    #model = get_pretrained_model(model_name='densenet121', num_classes=cfg.model["out_channels"], pretrained_weights='microscopynet')
    # Get the model
    # model = get_pretrained_model(
    #     model_name='efficientnet-b4', 
    #     num_classes=cfg.model["out_channels"],
    # )
    
    #print("Loaded SE-ResNet50")
    if model is not None:
        print(f"Model: {model.__class__.__name__} created")
    else:
        print("Model could not be created.")
    
    # Print model layers
    # print("Model layers:")
    # print("Model layers:")
    # for name, param in model.named_parameters():
    #     print(name)
    # Handle multi-GPU if available
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model) if cfg.training.get("distributed", False) else nn.DataParallel(model)

    # Move model to the appropriate device
    model = model.to(device)
    return model, device


# from utils.micronet_pretrained_models import get_pretrained_model
# import utils.transformations_functions as tf
# from configs.Config_loader import Config_loader
# from utils import setup_model

# yaml_path = "/home/zano/Documents/TESI/TESI/configs/densenet121.yaml"

# cfg = Config_loader(yaml_path) 
# cfg.set_freezed_layer_index(None)
# train_transforms, val_transforms, test_transforms = tf.get_transforms(cfg)