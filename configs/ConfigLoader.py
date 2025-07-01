import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

def merge_dicts(base: dict, override: dict) -> dict:
    """Merge two dictionaries iteratively.
    
    For each key/value pair in `override`, if the value is a dict and the corresponding
    base value is also a dict, then they are merged. Otherwise, the override value
    replaces (or is added to) the base dictionary.
    """
    # Use a stack to simulate the recursion, starting with the top-level dictionaries.
    stack = [(base, override)]
    
    while stack:
        current_base, current_override = stack.pop()
        
        for key, value in current_override.items():
            # If the override value is a dict and there is an existing dict in base, merge them.
            if isinstance(value, dict) and key in current_base and isinstance(current_base[key], dict):
                stack.append((current_base[key], value))
            else:
                current_base[key] = value
                
    return base

@dataclass
class ConfigLoader:
    # Define fields with proper defaults
    dataset : Optional[List[str]] = None
    class_names: Optional[List[str]] = None
    data_splitting: Optional[Dict[str, Any]] = None
    data_augmentation: Optional[Dict[str, Any]] = None
    data_loading: Optional[Dict[str, Any]] = None
    model: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None
    optimizer: Optional[Dict[str, Any]] = None
    scheduler: Optional[Dict[str, Any]] = None
    num_epochs: Optional[int] = None
    freeze_layers: Optional[bool] = None
    pretrained_weights: Optional[str] = None
    
    def __init__(self, config_path=None):
        """
        Initialize configuration either from a path to YAML file or from direct parameters.
        
        Args:
            config_path: Path to YAML config file (if provided)
        """
        if config_path is not None:
            # Load from YAML file
            try:
                # parse the yaml into a dictionary
                with open(config_path) as f:
                    config_dict = yaml.safe_load(f)
                
                if "_base_" in config_dict: #ie if there's inheritance
                    base_path = Path(config_path).parent / config_dict["_base_"] # get the base config path
                    # Check if the base config file exists
                    if not base_path.exists():
                        raise FileNotFoundError(f"Base config file not found: {base_path}")
                    # Load the base config and merge it with the current config
                    base_config = ConfigLoader(base_path).as_dict()
                    # print(f"before merge - transfer_learning value: {base_config.get('training', {}).get('transfer_learning')}")
                    config_dict = merge_dicts(base_config, config_dict)
                    # print(f"After merge - transfer_learning value: {config_dict.get('training', {}).get('transfer_learning')}")
                    if "_base_" in config_dict:
                        del config_dict["_base_"]
                
                # Set attributes from config_dict
                for key, value in config_dict.items():
                    setattr(self, key, value)
                    
                print(f"Configuration loaded from {config_path}")
                print(f"Configuration: {config_dict}")
                
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {config_path}: {e}")
            except KeyError as e:
                raise ValueError(f"Missing required field in config: {e}")
            except FileNotFoundError:
                print(f"Error: {config_path} not found")
                raise
        
        # Run post-initialization
        self._post_init_logic()
    
    def _post_init_logic(self):
        """Common post-initialization logic"""
        if not all([self.data_splitting, self.data_augmentation, self.data_loading,
                   self.model, self.training, self.optimizer, self.scheduler]):
            raise ValueError("Missing required configuration sections")
        
        if self.model["in_channels"] == 4 and self.training["transfer_learning"] == True:
            print(f"found {self.model['in_channels']} channels in input data and {self.training['transfer_learning']} in transfer learning")
            raise ValueError("Transfer learning is not supported for 4-channel input")
            
        self.training["pretrained"] = self.training["transfer_learning"] or self.training["fine_tuning"]
        if self.training["transfer_learning"] and self.training["fine_tuning"]:
            raise ValueError("Choose either transfer learning OR fine tuning")
    
    # Dataset getters and setters
    def get_dataset(self) -> Optional[List[str]]:
        """Get the dataset configuration"""
        return self.dataset
    
    def set_dataset(self, dataset: List[str]) -> None:
        """Set the dataset configuration
        
        Args:
            dataset: List of dataset names or paths
        """
        self.dataset = dataset

    def get_class_names(self) -> Optional[List[str]]:
        """Restituisce sempre la lista di classi, qualunque sia
        la sezione in cui è stata definita nel YAML."""
        if self.class_names is not None:
            return self.class_names
        if isinstance(self.dataset, dict) and "class_names" in self.dataset:
            return self.dataset["class_names"]
        return None  

    def set_class_names(self, class_names: List[str]) -> None:
        """Set the class names for the dataset

        Args:
            class_names: List of class names
        """
        self.class_names = class_names

    def set_spatial_size(self, spatial_size: tuple) -> None:
        """Set the spatial size for data augmentation
        
        Args:
            spatial_size: Tuple of (height, width) for resizing images
        """
        if self.data_augmentation is None:
            self.data_augmentation = {}
        self.data_augmentation["resize_spatial_size"] = spatial_size
        
    def get_spatial_size(self) -> tuple:
        """Get the spatial size for data augmentation"""
        if self.data_augmentation is None:
            raise ValueError("Data augmentation configuration is not set")
        if "resize_spatial_size" not in self.data_augmentation:
            raise ValueError("resize_spatial_size is not set in data augmentation configuration")
        if self.data_augmentation is None:
            raise ValueError("Data augmentation configuration is not set")
        if "resize_spatial_size" not in self.data_augmentation:
            raise ValueError("resize_spatial_size is not set in data augmentation configuration")
        if self.data_augmentation is None:
            raise ValueError("Data augmentation configuration is not set")
        return tuple(self.data_augmentation["resize_spatial_size"])
    
    # Data Splitting getters and setters
    def get_data_splitting(self) -> Optional[Dict[str, Any]]:
        """Get the data splitting configuration"""
        return self.data_splitting
    
    def set_data_splitting(self, data_splitting: Dict[str, Any]) -> None:
        """Set the data splitting configuration
        
        Args:
            data_splitting: Data splitting configuration dictionary
        """
        self.data_splitting = data_splitting
    
    def get_train_ratio(self) -> float:
        """Get the training data ratio"""
        if self.data_splitting is None:
            raise ValueError("Data splitting configuration is not set")
        return self.data_splitting.get("train_ratio", 0.7)
    
    def set_train_ratio(self, ratio: float) -> None:
        """Set the training data ratio
        
        Args:
            ratio: Training data ratio (0.0-1.0)
        """
        if self.data_splitting is None:
            self.data_splitting = {}
        self.data_splitting["train_ratio"] = ratio
    
    def get_val_ratio(self) -> float:
        """Get the validation data ratio"""
        if self.data_splitting is None:
            raise ValueError("Data splitting configuration is not set")
        return self.data_splitting.get("val_ratio", 0.15)
    
    def set_val_ratio(self, ratio: float) -> None:
        """Set the validation data ratio
        
        Args:
            ratio: Validation data ratio (0.0-1.0)
        """
        if self.data_splitting is None:
            self.data_splitting = {}
        self.data_splitting["val_ratio"] = ratio
    
    def get_test_ratio(self) -> float:
        """Get the test data ratio"""
        if self.data_splitting is None:
            raise ValueError("Data splitting configuration is not set")
        return self.data_splitting.get("test_ratio", 0.15)
    
    def set_test_ratio(self, ratio: float) -> None:
        """Set the test data ratio
        
        Args:
            ratio: Test data ratio (0.0-1.0)
        """
        if self.data_splitting is None:
            self.data_splitting = {}
        self.data_splitting["test_ratio"] = ratio
    
    # Data Augmentation getters and setters
    def get_data_augmentation(self) -> Optional[Dict[str, Any]]:
        """Get the data augmentation configuration"""
        return self.data_augmentation
    
    def set_data_augmentation(self, data_augmentation: Dict[str, Any]) -> None:
        """Set the data augmentation configuration
        
        Args:
            data_augmentation: Data augmentation configuration dictionary
        """
        self.data_augmentation = data_augmentation
    
    def get_augmentation_enabled(self) -> bool:
        """Get whether data augmentation is enabled"""
        if self.data_augmentation is None:
            raise ValueError("Data augmentation configuration is not set")
        return self.data_augmentation.get("enabled", False)
    
    def get_patch_size(self) -> int:
        if self.model is None:
            raise ValueError("Model configuration is not set")
        if "patch_size" not in self.model or self.model["patch_size"] is None:
            raise ValueError("patch_size is not set in model configuration")
        return self.model["patch_size"]
    
    def get_num_folds(self) -> int:
        """Get the number of folds for cross-validation"""
        if self.data_splitting is None:
            raise ValueError("Data splitting configuration is not set")
        return self.data_splitting.get("num_folds", 6)
    
    def set_augmentation_enabled(self, enabled: bool) -> None:
        """Set whether data augmentation is enabled
        
        Args:
            enabled: Boolean flag to enable/disable augmentation
        """
        if self.data_augmentation is None:
            self.data_augmentation = {}
        self.data_augmentation["enabled"] = enabled
    
    
    # Data Loading getters and setters
    def get_data_loading(self) -> Optional[Dict[str, Any]]:
        """Get the data loading configuration"""
        return self.data_loading
    
    def set_data_loading(self, data_loading: Dict[str, Any]) -> None:
        """Set the data loading configuration
        
        Args:
            data_loading: Data loading configuration dictionary
        """
        self.data_loading = data_loading
    
    def get_batch_size(self) -> int:
        """Get the batch size for data loading"""
        if self.data_loading is None:
            raise ValueError("Data loading configuration is not set")
        return self.data_loading.get("batch_size", 32)
    
    def set_batch_size(self, batch_size: int) -> None:
        """Set the batch size for data loading
        
        Args:
            batch_size: Batch size for training and evaluation
        """
        if self.data_loading is None:
            self.data_loading = {}
        self.data_loading["batch_size"] = batch_size
    
    def get_num_workers(self) -> int:
        """Get the number of workers for data loading"""
        if self.data_loading is None:
            raise ValueError("Data loading configuration is not set")
        return self.data_loading.get("num_workers", 4)
    
    def set_num_workers(self, num_workers: int) -> None:
        """Set the number of workers for data loading
        
        Args:
            num_workers: Number of worker processes for data loading
        """
        if self.data_loading is None:
            self.data_loading = {}
        self.data_loading["num_workers"] = num_workers
    
    # Model getters and setters
    def get_model(self) -> Optional[Dict[str, Any]]:
        """Get the model configuration"""
        return self.model
    
    def set_model(self, model: Dict[str, Any]) -> None:
        """Set the model configuration
        
        Args:
            model: Model configuration dictionary
        """
        self.model = model
    
    def get_model_name(self) -> str:
        """Get the name of the model from configuration"""
        return self.model.get("model_name", "unknown") if self.model else "unknown"
    
    def set_model_name(self, model_name: str) -> None:
        """Set the name of the model
        
        Args:
            model_name: Name of the model architecture
        """
        if self.model is None:
            self.model = {}
        self.model["model_name"] = model_name
        
    def get_image_shape(self) -> tuple:
        """Get the image shape for the model"""
        if self.model is None:
            raise ValueError("Model configuration is not set")
        return tuple(self.data_augmentation["resize_spatial_size"])
    
    def get_model_input_channels(self) -> int:
        """Get the number of input channels for the model"""
        if self.model["in_channels"] is None:
            raise ValueError("Input channels are not set in model configuration")
        return self.model["in_channels"]
    
    def set_model_input_channels(self, in_channels: int) -> None:
        """Set the number of input channels for the model
        
        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        """
        if self.model is None:
            self.model = {}
        self.model["in_channels"] = in_channels
    
    def get_model_library(self) -> str:
        """Get the library used for the model (e.g., 'torchvision', 'monai')"""
        if self.model is None:
            raise ValueError("Model configuration is not set")
        if "library" not in self.model:
            raise ValueError("Library not defined in model configuration")
        return self.model["library"]

    def set_model_library(self, library: str) -> None:
        """Set the library used for the model
        
        Args:
            library: Library name (e.g., 'torchvision', 'monai')
        """
        if self.model is None:
            self.model = {}
        self.model["library"] = library
        
    def get_pretrained_weights(self) -> Optional[str]:
        """Get the path to pretrained weights if available"""
        if self.model is None:
            raise ValueError("Model configuration is not set")
        return self.model.get("pretrained_weights", None)
    
    def set_model_input_channels(self, in_channels: int) -> None:
        """Set the number of input channels for the model
        
        Args:
            in_channels: Number of input channels
        """
        if self.model is None:
            self.model = {}
        self.model["in_channels"] = in_channels
    
    def get_num_classes(self) -> int:
        """Get the number of output classes for the model"""
        if self.model is None:
            raise ValueError("Model configuration is not set")
        return self.model.get("num_classes", 1)
    
    def set_num_classes(self, num_classes: int) -> None:
        """Set the number of output classes for the model
        
        Args:
            num_classes: Number of output classes
        """
        if self.model is None:
            self.model = {}
        self.model["num_classes"] = num_classes
    
    # Training getters and setters
    def get_training(self) -> Optional[Dict[str, Any]]:
        """Get the training configuration"""
        return self.training
    
    def set_training(self, training: Dict[str, Any]) -> None:
        """Set the training configuration
        
        Args:
            training: Training configuration dictionary
        """
        self.training = training
    
    def get_transfer_learning(self) -> bool:
        """Get whether transfer learning is enabled"""
        if self.training is None:
            raise ValueError("Training configuration is not set")
        return self.training.get("transfer_learning", False)
    
    def set_transfer_learning(self, enabled: bool) -> None:
        """Set whether transfer learning is enabled
        
        Args:
            enabled: Boolean flag to enable/disable transfer learning
        """
        if self.training is None:
            self.training = {}
        self.training["transfer_learning"] = enabled
        # Update pretrained flag
        if self.training.get("fine_tuning", False) and enabled:
            raise ValueError("Choose either transfer learning OR fine tuning")
        self.training["pretrained"] = enabled or self.training.get("fine_tuning", False)
    
    def get_fine_tuning(self) -> bool:
        """Get whether fine tuning is enabled"""
        if self.training is None:
            raise ValueError("Training configuration is not set")
        return self.training.get("fine_tuning", False)
    
    def set_fine_tuning(self, enabled: bool) -> None:
        """Set whether fine tuning is enabled
        
        Args:
            enabled: Boolean flag to enable/disable fine tuning
        """
        if self.training is None:
            self.training = {}
        self.training["fine_tuning"] = enabled
        # Update pretrained flag
        if self.training.get("transfer_learning", False) and enabled:
            raise ValueError("Choose either transfer learning OR fine tuning")
        self.training["pretrained"] = enabled or self.training.get("transfer_learning", False)
    
    def get_pretrained(self) -> bool:
        """Get whether to use pretrained weights"""
        if self.training is None:
            raise ValueError("Training configuration is not set")
        return self.training.get("pretrained", False)
    
    def set_pretrained(self, pretrained: bool) -> None:
        """Set the pretrained flag
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        if self.training is None:
            self.training = {}
        self.training["pretrained"] = pretrained
    
    def get_freezed_layer_index(self) -> Optional[int]:
        """Get the index of the last layer to freeze if present else None"""
        if self.training is None:
            raise ValueError("Training configuration is not set")
        return self.training.get("freezed_layerIndex")
    
    def set_freezed_layer_index(self, index: Optional[int]) -> None:
        """Set the index of the last layer to freeze
        
        Args:
            index: Index of the last layer to freeze
        """
        if self.training is None:
            self.training = {}
        self.training["freezed_layerIndex"] = index
    
    # Optimizer getters and setters
    def get_optimizer(self) -> Optional[Dict[str, Any]]:
        """Get the optimizer configuration"""
        return self.optimizer
    
    def set_optimizer(self, optimizer: Dict[str, Any]) -> None:
        """Set the optimizer configuration
        
        Args:
            optimizer: Optimizer configuration dictionary
        """
        self.optimizer = optimizer
    
    def get_optimizer_type(self) -> str:
        """Get the optimizer type"""
        if self.optimizer is None:
            raise ValueError("Optimizer configuration is not set")
        return self.optimizer.get("type", "Adam")
    
    def set_optimizer_type(self, optimizer_type: str) -> None:
        """Set the optimizer type
        
        Args:
            optimizer_type: Type of optimizer (e.g., 'Adam', 'SGD')
        """
        if self.optimizer is None:
            self.optimizer = {}
        self.optimizer["type"] = optimizer_type
    
    def get_learning_rate(self) -> float:
        """Get the learning rate for the optimizer"""
        if self.optimizer is None:
            raise ValueError("Optimizer configuration is not set")
        return self.optimizer.get("learning_rate", 0.001)
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate for the optimizer
        
        Args:
            learning_rate: Learning rate value
        """
        if self.optimizer is None:
            self.optimizer = {}
        self.optimizer["learning_rate"] = learning_rate
    
    def get_weight_decay(self) -> float:
        """Get the weight decay for the optimizer"""
        if self.optimizer is None:
            raise ValueError("Optimizer configuration is not set")
        return self.optimizer.get("weight_decay", 0.0)
    
    def set_weight_decay(self, weight_decay: float) -> None:
        """Set the weight decay for the optimizer
        
        Args:
            weight_decay: Weight decay value
        """
        if self.optimizer is None:
            self.optimizer = {}
        self.optimizer["weight_decay"] = weight_decay
    
    # Scheduler getters and setters
    def get_scheduler(self) -> Optional[Dict[str, Any]]:
        """Get the learning rate scheduler configuration"""
        return self.scheduler
    
    def set_scheduler(self, scheduler: Dict[str, Any]) -> None:
        """Set the learning rate scheduler configuration
        
        Args:
            scheduler: Scheduler configuration dictionary
        """
        self.scheduler = scheduler
    
    def get_scheduler_type(self) -> str:
        """Get the scheduler type"""
        if self.scheduler is None:
            raise ValueError("Scheduler configuration is not set")
        return self.scheduler.get("type", "StepLR")
    
    def set_scheduler_type(self, scheduler_type: str) -> None:
        """Set the scheduler type
        
        Args:
            scheduler_type: Type of scheduler (e.g., 'StepLR', 'CosineAnnealing')
        """
        if self.scheduler is None:
            self.scheduler = {}
        self.scheduler["type"] = scheduler_type
    
    def get_scheduler_step_size(self) -> int:
        """Get the step size for StepLR scheduler"""
        if self.scheduler is None:
            raise ValueError("Scheduler configuration is not set")
        return self.scheduler.get("step_size", 10)
    
    def set_scheduler_step_size(self, step_size: int) -> None:
        """Set the step size for StepLR scheduler
        
        Args:
            step_size: Step size in epochs
        """
        if self.scheduler is None:
            self.scheduler = {}
        self.scheduler["step_size"] = step_size
    
    def get_dropout_prob(self) -> float:
        """Get the dropout probability for the model"""
        if (self.training) is None:
            raise ValueError("training configuration not set")
        return self.training.get("dropout_prob", 0.2)
    
    # Number of epochs getters and setters
    def get_num_epochs(self) -> Optional[int]:
        """Get the number of epochs for training"""
        return self.training["num_epochs"] if self.training else None
    
    def set_num_epochs(self, epochs: int) -> None:
        """Set the number of epochs for training
        
        Args:
            epochs: Number of training epochs
        """
        if self.training is None:
            self.training = {}
        self.training["num_epochs"] = epochs

    def as_dict(self):
        """Convert the Config instance to a dictionary"""
        return asdict(self)  # Using dataclasses.asdict()

# def load_config(config_path: str) -> Config_loader:
#     try:
#         with open(config_path) as f:
#             config_dict = yaml.safe_load(f)
        
#         if "_base_" in config_dict:
#             base_path = Path(config_path).parent / config_dict["_base_"]
#             if not base_path.exists():
#                 raise FileNotFoundError(f"Base config file not found: {base_path}")
#             base_config = load_config(base_path).as_dict()
#             config_dict = merge_dicts(base_config, config_dict)
#             del config_dict["_base_"]
        
#         return Config_loader(**config_dict)
#     except yaml.YAMLError as e:
#         raise ValueError(f"Invalid YAML in {config_path}: {e}")
#     except KeyError as e:
#         raise ValueError(f"Missing required field in config: {e}")