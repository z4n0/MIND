import torch
from torch import nn
from configs.ConfigLoader import ConfigLoader # Ensure you import the correct class or type
from utils.micronet_pretrained_models import get_nasa_pretrained_model
# from torchvision.models import resnet50, resnet101, resnet152, ResNet50_Weights
import torchvision.models as models
from monai.networks.nets.densenet import DenseNet121, DenseNet169, DenseNet201
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights, ResNet50_Weights
import torchvision.transforms as transforms
from utils.train_functions import LinearProbeHead
from monai.networks.nets import (
    ResNet, EfficientNetBN, ViT,
    DenseNet121, DenseNet169, DenseNet201
    )
from monai.networks.nets.resnet import get_inplanes
from monai.networks.nets.senet import SEResNet50, SEResNet101
import timm
from difflib import get_close_matches
# ------------------------------
# ModelManager: Handles device selection and multi-GPU support
# ------------------------------
class ModelManager:
    """
    Handles device selection, model instantiation, and multi-GPU support.

    Example usage:
        cfg = {
            "model": {
                "model_name": "Densenet121",
                "out_channels": 2,
                "spatial_dims": 2,
                "in_channels": 3
            },
            "training": {
                "distributed": False
            }
        }
        manager = ModelManager(cfg)
        model, device = manager.setup_model(num_classes=2, pretrained_weights='imagenet')
    """
    def __init__(self, cfg, library="torchvision"):
        self.cfg = cfg
        self.model = None
        self.device = None
        # self.input_channels = cfg.model["in_channels"]  # e.g. 1 for grayscale, 3 for RGB
        self.library = library.lower()
        self.model_name = cfg.model["model_name"]  # e.g. "Densenet169", "Resnet50", etc.
    
    def detect_device(self):
        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU.")
            self.device = torch.device("cpu")
        else:
            n_gpus = torch.cuda.device_count()
            # print(f"Found {n_gpus} GPU{'s' if n_gpus > 1 else ''}.")
            self.device = torch.device("cuda:0")
        return self.device
    
    def setup_model(self, num_classes=2, pretrained_weights='imagenet'):
        """
        Instantiate and configure a neural network model according to the configuration and library.

        Args:
            num_classes (int): Number of output classes for the model. Default is 2.
            pretrained_weights (str): Which pretrained weights to use NOTE this is usless if it's not "micronet" or "microscopynet"
                Options: 'imagenet', 'micronet', 'microscopynet', or None. Default is 'imagenet'.

        Returns:
            tuple: (model, device)
                model: The instantiated and configured model, moved to the correct device.
                device: The torch.device used for the model.

        Raises:
            ValueError: If configuration is missing, unsupported model library is specified,
                or unsupported input channels for torchvision are requested.

        Notes:
            - If pretrained_weights is 'micronet' or 'microscopynet', loads weights using get_pretrained_model.
            - Selects the model factory (MONAI or Torchvision) based on the library.
            - Handles multi-GPU support with DataParallel or DistributedDataParallel if multiple GPUs are available.
            - Moves the model to the detected device (CPU or CUDA).
        """
        if self.cfg is None:
            raise ValueError("Configuration must be provided before calling setup_model")
        
        # Set random seed for reproducible model initialization
        torch.manual_seed(self.cfg.data_splitting["random_seed"])
        
        self.detect_device()
        
        if pretrained_weights in ["micronet","microscopynet","imagenet-microscopynet"]:
            # Load pretrained weights from Micronet or Microscopynet
            print(f"Loading pretrained weights from {pretrained_weights}")
            self.model = get_nasa_pretrained_model(model_name=self.model_name, num_classes=num_classes, pretrained_weights=pretrained_weights)
            self.model = self.model.to(self.device) #move model to device
            return self.model, self.device
        
        # Select factory based on config: e.g., cfg.model["model_library"] can be 'torchvision' or 'monai'
        if self.library == "monai":
                print("Using MONAI for model instantiation.")
                factory = MonaiModelFactory(self.cfg)
        elif self.library == "torchvision":
                if self.cfg.get_model_input_channels() == 4:
                    raise ValueError(
                        "Torchvision factory: 4-channel input not supported out-of-the-box. "
                        "Use 'timm' or 'monai' instead, or patch the first conv."
                    )
                print("Using Torchvision for model instantiation.")
                factory = TorchvisionModelFactory(self.cfg)
        elif self.library == "timm":
                print("Using timm for model instantiation.")
                factory = TimmModelFactory(self.cfg)
        else:
                raise ValueError(
                    f"Model library '{self.library}' is not supported. "
                    "Supported: 'torchvision', 'monai', 'timm'."
                )

        self.model = factory.create_model(
                    model_name=self.model_name,
                    pretrained_weights=pretrained_weights,
                    num_classes=num_classes,
                )

        if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
        return self.model, self.device

## interface for model factories
# ------------------------------
# class BaseModelFactory:
#     def create_model(self, model_name, pretrained_weights, num_classes):
#         raise NotImplementedError("Subclasses must implement create_model()")
# ------------------------------
# ModelFactory: Encapsulates how to build each model
# ------------------------------
import torch.nn as nn
import torchvision.models as models
from utils.reproducibility_functions import set_global_seed

class TorchvisionModelFactory():
    def __init__(self, cfg):
        self.cfg = cfg
        self.pretrained = cfg.get_transfer_learning() # e.g. True or False
        # Dictionary mapping model names to their respective builder methods
        # Each builder method takes pretrained_weights and num_classes as arguments
        # and returns a configured model instance
        self.builders = {
            # DenseNet variants with different depths
            "Densenet121": self.build_densenet121,  # 121 layers
            "Densenet169": self.build_densenet169,  # 169 layers  
            "Densenet201": self.build_densenet201,  # 201 layers
            # ResNet variants with different depths
            "Resnet18": self.build_resnet18,    # 18 layers
            "Resnet50": self.build_resnet50,    # 50 layers
            "Resnet101": self.build_resnet101,  # 101 layers
            "Resnet152": self.build_resnet152,  # 152 layers
        }
        
    
    def get_available_models(self):
        return list(self.builders.keys())
        
    def create_model(self, model_name, pretrained_weights=None, num_classes=2):
        print(f" pretrained? {self.pretrained}")
        if self.pretrained:
            print(f" using weights: {pretrained_weights}")
        else:
            pretrained_weights = None
            print(f" using no weights")
        # Make case-insensitive lookup
        model_name_key = None
        for key in self.builders.keys():
            if key.lower() == model_name.lower():
                model_name_key = key
                break
        
        if model_name_key is None:
            raise ValueError(f"Model {model_name} not supported by Torchvision factory. Supported models: {list(self.builders.keys())}")
        
        builder = self.builders[model_name_key]
        if builder is None:
            raise ValueError(f"Model {model_name} not supported by Torchvision factory. Supported models: {list(self.builders.keys())}")
        return builder(pretrained_weights, num_classes)
    
    def build_densenet121(self, pretrained_weights, num_classes):
        """
        Build a torchvision DenseNet121 model with optional pretrained weights.

        Args:
            pretrained_weights: Not used, kept for interface compatibility.
            num_classes (int): Number of output classes.

        Returns:
            torch.nn.Module: Configured DenseNet121 model.
        """
        print("Building torchvision DenseNet121...")
        weights = models.DenseNet121_Weights.DEFAULT if self.pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = LinearProbeHead(in_features, num_classes)  # type: ignore
        return model

    def build_densenet169(self, pretrained_weights, num_classes):
        """
        Build a torchvision DenseNet169 model with optional pretrained weights.

        Args:
            pretrained_weights: Not used, kept for interface compatibility.
            num_classes (int): Number of output classes.

        Returns:
            torch.nn.Module: Configured DenseNet169 model.
        """
        print("Building torchvision DenseNet169...")
        weights = models.DenseNet169_Weights.DEFAULT if self.pretrained else None
        model = models.densenet169(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = LinearProbeHead(in_features, num_classes)  # type: ignore
        return model

    def build_densenet201(self, pretrained_weights, num_classes):
        """
        Build a torchvision DenseNet201 model with optional pretrained weights.

        Args:
            pretrained_weights: Not used, kept for interface compatibility.
            num_classes (int): Number of output classes.

        Returns:
            torch.nn.Module: Configured DenseNet201 model.
        """
        print("Building torchvision DenseNet201...")
        weights = models.DenseNet201_Weights.DEFAULT if self.pretrained else None
        model = models.densenet201(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = LinearProbeHead(in_features, num_classes)  # type: ignore
        return model

    def build_resnet18(self, pretrained_weights, num_classes):
        """
        Build a torchvision ResNet18 model with optional pretrained weights.

        Args:
            pretrained_weights: Not used, kept for interface compatibility.
            num_classes (int): Number of output classes.

        Returns:
            torch.nn.Module: Configured ResNet18 model.
        """
        print("Building torchvision ResNet18...")
        weights = models.ResNet18_Weights.DEFAULT if self.pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = LinearProbeHead(in_features, num_classes)  # type: ignore
        return model

    def build_resnet50(self, pretrained_weights, num_classes):
        """
        Build a torchvision ResNet50 model with optional pretrained weights.

        Args:
            pretrained_weights: Not used, kept for interface compatibility.
            num_classes (int): Number of output classes.

        Returns:
            torch.nn.Module: Configured ResNet50 model.
        """
        print("Building torchvision ResNet50...")
        weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = LinearProbeHead(in_features, num_classes)  # type: ignore
        return model

    def build_resnet101(self, pretrained_weights, num_classes):
        """
        Build a torchvision ResNet101 model with optional pretrained weights.

        Args:
            pretrained_weights: Not used, kept for interface compatibility.
            num_classes (int): Number of output classes.

        Returns:
            torch.nn.Module: Configured ResNet101 model.
        """
        print("Building torchvision ResNet101...")
        weights = models.ResNet101_Weights.DEFAULT if self.pretrained else None
        model = models.resnet101(weights=weights)
        in_features = model.fc.in_features
        model.fc = LinearProbeHead(in_features, num_classes)  # type: ignore
        return model

    def build_resnet152(self, pretrained_weights, num_classes):
        """
        Build a torchvision ResNet152 model with optional pretrained weights.

        Args:
            pretrained_weights: Not used, kept for interface compatibility.
            num_classes (int): Number of output classes.

        Returns:
            torch.nn.Module: Configured ResNet152 model.
        """
        print("Building torchvision ResNet152...")
        weights = models.ResNet152_Weights.DEFAULT if self.pretrained else None
        model = models.resnet152(weights=weights)
        in_features = model.fc.in_features
        model.fc = LinearProbeHead(in_features, num_classes)  # type: ignore
        return model

class MonaiModelFactory():
    """
    Factory for creating MONAI models.
    NB: densenets are can be created using imagenet weights, but resnets are not.
    to use pretrained resnets, you need to use the torchvision factory.
    """
    def __init__(self, cfg: ConfigLoader): # Added default None for cfg if not always used
        self.cfg = cfg
        self.cfg.get_model_input_channels()
        self.pretrained = cfg.get_transfer_learning() # e.g. True or False
        self.input_channels = cfg.get_model_input_channels() # e.g. 1 for grayscale, 3 for RGB
        
        # Added Resnet18 and Resnet34
        self.builders = {
            "Densenet169": self.build_densenet169,
            "Densenet121": self.build_densenet121,
            "Densenet201": self.build_densenet201,
            "Resnet18": self.build_resnet18,      # Added
            "Resnet34": self.build_resnet34,      # Added
            "Resnet50": self.build_resnet50,
            "Resnet101": self.build_resnet101,
            "Resnet152": self.build_resnet152,
            "ViT": self.build_vit,
            # SEResNet ----------------------------------------------------
            "seresnet50": self._build_seresnet50,
            "seresnet101": self._build_seresnet101,
            # EfficientNetBN --------------------------------------------
            "efficientnet_b0": self._build_efficientnet_b0,
            "efficientnet_b3": self._build_efficientnet_b3,
        }
    
    def _adapt_first_conv(self, model: nn.Module, in_channels: int) -> nn.Module:
        """Adapt the first convolution layer to handle different input channels."""
        if in_channels > 3:
            # Get the first conv layer
            first_conv = model.conv1
            if isinstance(first_conv, nn.Conv2d):
                # Ensure kernel_size, stride, and padding are proper 2D tuples
                kernel_size = (first_conv.kernel_size[0], first_conv.kernel_size[0]) if isinstance(first_conv.kernel_size, tuple) else (first_conv.kernel_size, first_conv.kernel_size)
                stride = (first_conv.stride[0], first_conv.stride[0]) if isinstance(first_conv.stride, tuple) else (first_conv.stride, first_conv.stride)
                padding = (first_conv.padding[0], first_conv.padding[0]) if isinstance(first_conv.padding, tuple) else (first_conv.padding, first_conv.padding)
                
                # Create new conv layer with correct number of input channels
                new_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=first_conv.out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=first_conv.bias is not None
                )
                # Initialize the new conv layer
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias, 0)
                # Replace the first conv layer
                model.conv1 = new_conv
        return model

    def create_model(self, model_name, pretrained_weights=None, num_classes=None): # Added defaults
        
        builder = self.builders.get(model_name)
        is_resnet = "resnet" in model_name.lower()
        if pretrained_weights is not None and is_resnet: 
            if pretrained_weights == "imagenet":
                print(f"using Pretrained weights: {pretrained_weights}")
            if is_resnet:
                raise ValueError(f"Resnet {model_name} does not support pretrained weights in the MONAI factory. Use torchvision instead.")
            
        if builder is None:
            raise ValueError(f"Model {model_name} not supported by MONAI factory. Supported models: {list(self.builders.keys())}")
        # Make sure num_classes is provided if needed by the builder
        if num_classes is None and "ViT" not in model_name : # ViT might get classes differently, adapt if needed
             raise ValueError(f"num_classes must be provided for model {model_name}")
        # Note: pretrained_weights argument is passed but not used yet for MONAI loading here.
        # Actual MONAI pretrained loading (like MedicalNet) has specific requirements
        # (e.g., spatial_dims=3, in_channels=1) and uses a boolean flag usually.
        # Loading custom weights from a path would require extra logic here.
        return builder(num_classes)

    def build_densenet169(self, num_classes):
        print(f"Building MONAI DenseNet169 with {self.input_channels}-channel input...")
        # MONAI DenseNet169 may not have the same pretrained options.
        model = DenseNet169(spatial_dims=2, in_channels = self.input_channels, out_channels=num_classes, pretrained=self.pretrained)
        # Replace the final classifier: in MONAI DenseNets, final block is in model.class_layers.
        # Check if class_layers and out exist before modification
        # if hasattr(model, 'class_layers') and hasattr(model.class_layers, 'out'):
        #      in_features = model.class_layers.out.in_features
        #      model.class_layers.out = nn.Linear(in_features, num_classes)
        #      print(f" -> Final layer replaced: Linear({in_features}, {num_classes})")
        # else:
        #      print(" -> Warning: Could not find model.class_layers.out to replace the final layer.")
        return model

    def build_densenet121(self, num_classes):
        print(f"Building MONAI DenseNet121 with {self.input_channels}-channel input...")
        model = DenseNet121(spatial_dims=2, in_channels=self.input_channels, out_channels=num_classes, pretrained=self.pretrained)
        # if hasattr(model, 'class_layers') and hasattr(model.class_layers, 'out'):
        #     in_features = model.class_layers.out.in_features
        #     model.class_layers.out = nn.Linear(in_features, num_classes)
        #     print(f" -> Final layer replaced: Linear({in_features}, {num_classes})")
        # else:
        #      print(" -> Warning: Could not find model.class_layers.out to replace the final layer.")
        return model

    def build_densenet201(self, num_classes):
        print(f"Building MONAI DenseNet201 with {self.input_channels}-channel input...")
        model = DenseNet201(spatial_dims=2, in_channels=self.input_channels, out_channels=num_classes, pretrained=self.pretrained)
        # if hasattr(model, 'class_layers') and hasattr(model.class_layers, 'out'):
        #     in_features = model.class_layers.out.in_features
        #     model.class_layers.out = nn.Linear(in_features, num_classes)
        #     print(f" -> Final layer replaced: Linear({in_features}, {num_classes})")
        # else:
        #      print(" -> Warning: Could not find model.class_layers.out to replace the final layer.")
        return model

    # --- NEW ResNet Builders ---
    def build_resnet18(self, num_classes):
        print(f"Building MONAI ResNet18 with {self.input_channels}-channel input...")
        model = ResNet(
            block="basic",              # Use BasicBlock for ResNet18/34
            layers=[2, 2, 2, 2],        # Standard configuration for ResNet18
            block_inplanes = get_inplanes(), # Standard channel dimensions [64, 128, 256, 512]
            spatial_dims=2,
            n_input_channels=self.input_channels,         # Your required input channels
            num_classes=num_classes,
            feed_forward=True           # Ensure final FC layer is present
        )
        # print(f" -> ResNet18 configured with {num_classes} output classes.")
        # Note: MONAI ResNet's final layer is 'fc'. It's already configured by num_classes.
        # No replacement needed like in DenseNet unless you have specific needs.
        return model

    def build_resnet34(self, num_classes):
        print(f"Building MONAI ResNet34 with {self.input_channels}-channel input...")
        model = ResNet(
            block="basic",              # Use BasicBlock for ResNet18/34
            layers=[3, 4, 6, 3],        # Standard configuration for ResNet34
            block_inplanes=get_inplanes(), # Standard channel dimensions [64, 128, 256, 512]
            spatial_dims=2,
            n_input_channels=self.input_channels,         # Your required input channels
            num_classes=num_classes,
            feed_forward=True           # Ensure final FC layer is present
        )
        # print(f" -> ResNet34 configured with {num_classes} output classes.")
        return model
    # --- End NEW ResNet Builders ---

    def build_resnet50(self, num_classes):
        print(f"Building MONAI ResNet50 with {self.input_channels}-channel input...")
        model = ResNet(
            block="bottleneck",         # Use Bottleneck blocks for ResNet50+
            layers=[3, 4, 6, 3],        # Standard configuration for ResNet50
            block_inplanes=get_inplanes(), # Standard channel dimensions
            spatial_dims=2,
            n_input_channels=self.input_channels,         # Your required input channels
            num_classes=num_classes,
            feed_forward=True
        )
        # print(f" -> ResNet50 configured with {num_classes} output classes.")
        return model

    def build_resnet101(self, num_classes):
        print(f"Building MONAI ResNet101 with {self.input_channels}-channel input...")
        model = ResNet(
            block="bottleneck",
            layers=[3, 4, 23, 3],       # Configuration for ResNet101
            block_inplanes=get_inplanes(), # Standard channel dimensions
            spatial_dims=2,
            n_input_channels=self.input_channels,         # Your required input channels
            num_classes=num_classes,
            feed_forward=True
        )
        print(f" -> ResNet101 configured with {num_classes} output classes.")
        return model

    def build_resnet152(self, num_classes):
        print(f"Building MONAI ResNet152 with {self.input_channels}-channel input...")
        model = ResNet(
            block="bottleneck",
            layers=[3, 8, 36, 3],       # Configuration for ResNet152
            block_inplanes=get_inplanes(), # Standard channel dimensions
            spatial_dims=2,
            n_input_channels=self.input_channels,         # Your required input channels
            num_classes=num_classes,
            feed_forward=True
        )
        # print(f" -> ResNet152 configured with {num_classes} output classes.")
        return model
    
    # SEResNet -----------------------------------------------------------

    def _build_seresnet50(self, num_classes: int) -> nn.Module:
        model = SEResNet50(
            spatial_dims=2,
            in_channels=self.input_channels,
            num_classes=num_classes,
            pretrained=self.pretrained,
        )
        return model

    def _build_seresnet101(self, num_classes: int) -> nn.Module:
        model = SEResNet101(
            spatial_dims=2,
            in_channels=self.input_channels,
            num_classes=num_classes,
            pretrained=self.pretrained,
        )
        return model

    # EfficientNetBN -----------------------------------------------------
    def _build_efficientnet_b0(self,num_classes: int) -> nn.Module:
        model = EfficientNetBN(
            spatial_dims=2,
            model_name="efficientnet-b0",
            in_channels=self.input_channels,
            num_classes=num_classes,
            pretrained=self.pretrained,
        )
        # EfficientNetBN needs manual first‑conv patch for >3 channels
        model = self._adapt_first_conv(model, self.input_channels)
        return model

    def _build_efficientnet_b3(self, num_classes: int) -> nn.Module:
        model = EfficientNetBN(
            spatial_dims=2,
            model_name="efficientnet-b3",
            in_channels=self.input_channels,
            num_classes=num_classes,
            pretrained=self.pretrained,
        )
        model = self._adapt_first_conv(model, self.input_channels)
        return model

    # Vision Transformer -------------------------------------------------
    def build_vit(self, num_classes):
        """Builds a MONAI Vision Transformer model using parameters from the config."""
        print(f"Building MONAI ViT for classification with {self.input_channels}-channel input...")
        
        model = ViT(
            in_channels=self.input_channels,
            img_size=tuple(self.cfg.model['img_size']),
            patch_size=tuple(self.cfg.model['patch_size']),
            hidden_size=self.cfg.model['hidden_size'],
            mlp_dim=self.cfg.model['mlp_dim'],
            num_layers=self.cfg.model['num_layers'],
            num_heads=self.cfg.model['num_heads'],
            classification=True,
            num_classes=num_classes,
            save_attn=True, # Crucial for attention maps
            spatial_dims=2
        )
        
        print(f" -> ViT configured with {num_classes} output classes.")
        return model

    


# TIMM Model Factory -----------------------------------------------------
# timm stands for torch image models
class TimmModelFactory:
    """
    Factory for creating timm classification backbones that natively support
    in_chans > 3 (e.g., 4-channel fluorescence inputs).
    to use this factory, you need to have timm installed. and 
    you have to change the cfg.model["model_library"] to "timm"

    Reads the following (optional) keys from cfg.model:
      - img_size: (H, W), default 224x224 (used by some models internally)
      - global_pool: 'avg' | 'max' | 'avgmax' | 'catavgmax' | 'token' (ViTs)
      - drop_rate: float, default 0.0
      - drop_path_rate: float, default 0.0
      - head_init_scale: float for some transformer heads (if supported)
      - timm_name: (optional) canonical timm name if you want to bypass synonyms
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.pretrained = bool(cfg.get_transfer_learning())
        self.input_channels = int(cfg.get_model_input_channels())
        self.num_classes = int(cfg.get_num_classes())
        self.model_cfg = getattr(cfg, "model", {})
        self.global_pool = self.model_cfg.get("global_pool", "avg")
        self.drop_rate = float(self.model_cfg.get("drop_rate", 0.0))
        self.drop_path_rate = float(self.model_cfg.get("drop_path_rate", 0.0))

        self._name_map = {
            "resnet18": "resnet18",
            "resnet34": "resnet34",
            "resnet50": "resnet50",
            "resnet101": "resnet101",
            "resnet152": "resnet152",
            "densenet121": "densenet121",
            "densenet169": "densenet169",
            "densenet201": "densenet201",
            "convnext_tiny": "convnext_tiny",
            "convnext_small": "convnext_small",
            "efficientnet_b0": "efficientnet_b0",
            "efficientnet_b3": "efficientnet_b3",
            "vit_base_patch16_224": "vit_base_patch16_224",
            "vit_small_patch16_224": "vit_small_patch16_224",      # ADD THIS
            "vit_base_patch16_384": "vit_base_patch16_384",        # ADD THIS
            "deit_small_patch16_224": "deit_small_patch16_224",
            "deit_base_patch16_224": "deit_base_patch16_224",      # ADD THIS
            "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224",
        }

    def _normalize_name(self, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("model_name must be a string.")
        key = name.replace("-", "_").lower()
        # Allow overriding with an explicit timm name in cfg.model
        override = self.model_cfg.get("timm_name")
        if isinstance(override, str) and override.strip():
            return override.strip()
        return self._name_map.get(key, key)

    def _resolve_pretrained_flag(self, pretrained_weights) -> bool:
        """
        We treat 'imagenet' (case-insensitive) as True; anything else falls
        back to cfg.get_transfer_learning().
        """
        if pretrained_weights is None:
            return self.pretrained
        if isinstance(pretrained_weights, str):
            return pretrained_weights.lower() == "imagenet"
        return bool(pretrained_weights)

    def _adapt_first_layer_for_4ch(self, model: nn.Module, pretrained: bool) -> nn.Module:
        """
        Adapt the first layer to handle 4-channel input when using 3-channel pretrained weights.
        
        Three strategies:
        1. 'repeat_avg': Average the 3-channel weights and repeat for the 4th channel (default)
        2. 'random_init': Keep first 3 channels pretrained, initialize 4th randomly
        3. 'zero_init': Keep first 3 channels pretrained, initialize 4th to zeros
        
        Args:
            model: The model with pretrained weights
            pretrained: Whether pretrained weights were loaded
            
        Returns:
            Modified model with 4-channel first layer
        """
        import torch
        
        strategy = self.model_cfg.get("channel_adaptation_strategy", "repeat_avg")
        
        # Find the first conv layer (different for different architectures)
        first_conv = None
        first_conv_name = None
        
        # Common first layer names in different architectures
        # ViT/DeiT use 'patch_embed.proj', CNNs use 'conv1' or similar
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Check if it's likely the first conv (has 3 input channels if pretrained)
                if module.in_channels == 3 and pretrained:
                    first_conv = module
                    first_conv_name = name
                    break
        
        if first_conv is None:
            print("⚠️  Could not find first conv layer with 3 input channels to adapt.")
            print("   The model may already support 4 channels or may fail during forward pass.")
            return model
            
        print(f"📌 Adapting first layer '{first_conv_name}' from 3 to 4 channels using '{strategy}' strategy")
        
        # Get original weights and parameters
        old_weight = first_conv.weight.data  # Shape: (out_channels, 3, kH, kW)
        out_channels = first_conv.out_channels
        # Ensure kernel_size, stride, padding are tuples for type safety
        kernel_size = first_conv.kernel_size if isinstance(first_conv.kernel_size, tuple) else (first_conv.kernel_size, first_conv.kernel_size)
        stride = first_conv.stride if isinstance(first_conv.stride, tuple) else (first_conv.stride, first_conv.stride)
        padding = first_conv.padding if isinstance(first_conv.padding, tuple) else (first_conv.padding, first_conv.padding)
        bias = first_conv.bias is not None
        
        # Create new conv layer with 4 input channels
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=out_channels,
            kernel_size=kernel_size,  # type: ignore
            stride=stride,  # type: ignore
            padding=padding,  # type: ignore
            bias=bias
        )
        
        # Initialize new weights based on strategy
        with torch.no_grad():
            if strategy == "repeat_avg":
                # Average the RGB weights for the 4th channel
                avg_weight = old_weight.mean(dim=1, keepdim=True)  # Average across input channels
                new_conv.weight[:, :3, :, :] = old_weight  # Copy RGB weights
                new_conv.weight[:, 3:4, :, :] = avg_weight  # Use average for 4th channel
                print(f"   ✓ Copied RGB pretrained weights + averaged for channel 4")
                
            elif strategy == "random_init":
                # Keep RGB pretrained, initialize 4th channel randomly
                new_conv.weight[:, :3, :, :] = old_weight  # Copy RGB weights
                nn.init.kaiming_normal_(new_conv.weight[:, 3:4, :, :], mode='fan_out', nonlinearity='relu')
                print(f"   ✓ Copied RGB pretrained weights + random init for channel 4")
                
            elif strategy == "zero_init":
                # Keep RGB pretrained, initialize 4th channel to zeros
                new_conv.weight[:, :3, :, :] = old_weight  # Copy RGB weights
                new_conv.weight[:, 3:4, :, :].zero_()
                print(f"   ✓ Copied RGB pretrained weights + zero init for channel 4")
                
            else:
                raise ValueError(f"Unknown channel adaptation strategy: {strategy}")
            
            # Copy bias if present
            if bias and first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data
        
        # Replace the first conv layer in the model
        # Navigate the module hierarchy to replace the layer
        if first_conv_name is None:
            raise ValueError("Could not determine first conv layer name")
        
        parts = first_conv_name.split('.')
        if len(parts) == 1:
            # Direct attribute (e.g., 'conv1')
            setattr(model, first_conv_name, new_conv)
        else:
            # Nested attribute (e.g., 'patch_embed.proj')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
        
        print(f"   ✓ First layer adapted successfully")
        return model

    def create_model(self, model_name: str, pretrained_weights=None,
                     num_classes: int = 2) -> nn.Module:
        """
        Build a timm model with `in_chans` taken from cfg, and the specified
        num_classes. Raises a helpful error when the name is not found.
        
        Supports 4-channel inputs with pretrained weights by:
        1. Loading model with 3 channels and pretrained weights
        2. Adapting the first conv layer to accept 4 channels
        3. Preserving pretrained weights for first 3 channels
        """
        timm_name = self._normalize_name(model_name)
        use_pretrained = self._resolve_pretrained_flag(pretrained_weights)

        # Handle 4-channel input with pretrained weights
        if self.input_channels == 4 and use_pretrained:
            print(f"🔧 Creating model with 3 channels first (for pretrained weights), then adapting to 4 channels")
            try:
                # First create model with 3 channels to load pretrained weights
                model = timm.create_model(
                    timm_name,
                    pretrained=use_pretrained,
                    in_chans=3,  # Load with 3 channels initially for pretrained weights
                    num_classes=num_classes,
                    global_pool=self.global_pool,
                    drop_rate=self.drop_rate,
                    drop_path_rate=self.drop_path_rate,
                )
                # Then adapt the first layer to 4 channels
                model = self._adapt_first_layer_for_4ch(model, pretrained=use_pretrained)
                
            except Exception as exc:
                all_models = timm.list_models(pretrained=use_pretrained)
                suggestions = get_close_matches(timm_name, all_models, n=5, cutoff=0.3)
                msg = (
                    f"[timm] Could not instantiate model '{timm_name}'. "
                    f"Close matches: {suggestions}. "
                    "Tip: set cfg.model['timm_name'] to the exact timm identifier."
                )
                raise ValueError(msg) from exc
        else:
            # Standard case: 3-channel or non-pretrained models
            try:
                model = timm.create_model(
                    timm_name,
                    pretrained=use_pretrained,
                    in_chans=self.input_channels,
                    num_classes=num_classes,
                    global_pool=self.global_pool,
                    drop_rate=self.drop_rate,
                    drop_path_rate=self.drop_path_rate,
                )
            except Exception as exc:
                all_models = timm.list_models(pretrained=use_pretrained)
                suggestions = get_close_matches(timm_name, all_models, n=5, cutoff=0.3)
                msg = (
                    f"[timm] Could not instantiate model '{timm_name}'. "
                    f"Close matches: {suggestions}. "
                    "Tip: set cfg.model['timm_name'] to the exact timm identifier."
                )
                raise ValueError(msg) from exc

        # Optional: linear-probe warmup support—freeze backbone if requested
        linear_probe = bool(self.model_cfg.get("linear_probe_warmup", False))
        if linear_probe:
            for name, param in model.named_parameters():
                # Heuristic: keep only final classifier trainable
                if "fc" in name or "classifier" in name or "head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        return model