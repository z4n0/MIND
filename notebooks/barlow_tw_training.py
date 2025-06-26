from matplotlib.dates import SA
from torch.utils.data import DataLoader
from torch.optim import AdamW
# Import PyTorch schedulers
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torchvision
import torch
import torch.nn as nn
# Lightly imports (make sure these are installed and correct)
from lightly.data import LightlyDataset
from lightly.transforms.byol_transform import BYOLTransform, BYOLView1Transform, BYOLView2Transform
import pytorch_lightning as pl
# Assuming BarlowTwinsProjectionHead and BarlowTwinsLoss are defined correctly elsewhere
# from your_module import BarlowTwinsProjectionHead, BarlowTwinsLoss
from utils.setup_functions import get_tif_image_paths_from_folder
import numpy as np
LEARNING_RATE = 3e-4 # Initial learning rate (peak after warmup)
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100     # Or your desired number
WARMUP_EPOCHS = 10   # Number of epochs for linear warmup
BATCH_SIZE = 8      # Increase if GPU memory allows (e.g., 64, 128, 256)
NUM_WORKERS = 4
GRADIENT_CLIP_VAL = 1.0 # Max norm for gradient clipping
ssl_images_folder_path = "/home/zano/Documents/TESI/4c_MIP/CONTROL"
INPUT_DIR = ssl_images_folder_path
ssl_images_paths = get_tif_image_paths_from_folder(INPUT_DIR)
ssl_images_paths_np = np.array(ssl_images_paths)
BARLOW_PROJECT_DIM = 2048 # Or your desired projection dim
SAVE_DIR = "/home/zano/Documents/TESI/TESI/notebooks" # Directory to save the backbone weights


# --- Helper function for linear warmup lambda ---
def linear_warmup_decay(warmup_steps):
    """ Linear warmup for warmup_steps steps. """
    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0 # Constant multiplier after warmup
    return fn

########################################################
# 2) Barlow Twins LightningModule
########################################################
class BarlowTwins(pl.LightningModule):
    def __init__(self, learning_rate, warmup_epochs, max_epochs):
        """
        ResNet18 backbone -> 512-d features
        Projection head -> (512 -> proj_dim -> proj_dim)
        BarlowTwinsLoss for self-supervised training
        """
        super().__init__()
        # Save hyperparameters like learning_rate, warmup_epochs, max_epochs
        # These are needed for scheduler setup in configure_optimizers
        self.save_hyperparameters()

        # Create ResNet18, initialized with ImageNet weights
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        resnet_output_dim = 512

        # Barlow Twins head
        self.projection_head = BarlowTwinsProjectionHead(
            resnet_output_dim,
            BARLOW_PROJECT_DIM,
            BARLOW_PROJECT_DIM
        )
        self.criterion = BarlowTwinsLoss()

        # Store steps_per_epoch, calculated in setup
        self.steps_per_epoch = 0

    # --- REVISED setup method ---
    def setup(self, stage=None):
        """Calculate steps_per_epoch after dataloader is available."""
        if stage == 'fit' or stage is None:
            # Check if trainer and dataloaders are available
            if not self.trainer or not self.trainer.train_dataloader:
                print("Warning: Trainer or train_dataloader not available in setup.")
                return

            try:
                # Get the length of the dataloader
                self.steps_per_epoch = len(self.trainer.train_dataloader)
                print(f"Calculated steps per epoch in setup: {self.steps_per_epoch}")

            except TypeError:
                # Handle cases where dataloader has no __len__ (e.g., IterableDataset)
                # Estimate based on limit_train_batches if set
                if self.trainer.limit_train_batches:
                     if isinstance(self.trainer.limit_train_batches, int):
                         self.steps_per_epoch = self.trainer.limit_train_batches
                     elif isinstance(self.trainer.limit_train_batches, float):
                          # Estimate requires dataset size, difficult to get reliably here
                          self.steps_per_epoch = 500 # Fallback guess
                          print(f"Warning: Cannot determine length from dataloader and limit_train_batches is a float.")
                          print(f"Using fallback estimate for steps_per_epoch: {self.steps_per_epoch}")
                     else:
                          self.steps_per_epoch = 500 # Fallback guess
                          print(f"Warning: Unknown type for limit_train_batches.")
                          print(f"Using fallback estimate for steps_per_epoch: {self.steps_per_epoch}")
                     print(f"Using limit_train_batches to set steps_per_epoch: {self.steps_per_epoch}")
                else:
                     # If no length and no limit, estimation is hard. Use a default with warning.
                     self.steps_per_epoch = 500 # Fallback guess
                     print(f"Warning: Could not determine dataloader length (possibly IterableDataset without length).")
                     print(f"Using fallback estimate for steps_per_epoch: {self.steps_per_epoch}")
                     print(f"For accurate scheduler behaviour, ensure train dataloader has __len__ or set trainer's limit_train_batches.")

            if self.steps_per_epoch == 0:
                 raise ValueError("Could not determine steps_per_epoch for LR scheduler. Aborting.")

    def forward(self, x):
        """ Forward pass for one augmented view x. """
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_idx):
        """ Barlow Twins requires two augmented views (x0, x1). """
        # Adapt based on actual batch structure from LightlyDataset
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
             (x0, x1), _, _ = batch # common case: (views), labels, fnames
        elif isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[0], (list, tuple)):
             (x0, x1), _ = batch # case: (views), labels
        elif isinstance(batch, (list, tuple)) and len(batch) == 1 and isinstance(batch[0], (list, tuple)):
             (x0, x1) = batch[0] # case: (views)
        else:
            # Attempt to handle dictionary batches if applicable
            try:
                 views = batch[0] # Assuming views are the first element if it's not a standard lightly format
                 if isinstance(views, (list, tuple)) and len(views) == 2:
                      x0, x1 = views
                 else:
                      raise TypeError("Could not extract two views from batch[0]")
            except (IndexError, KeyError, TypeError) as e:
                 raise ValueError(f"Unexpected batch format in training_step. Type: {type(batch)}, Content sample: {str(batch)[:100]}, Error: {e}")


        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        # Log loss and learning rate
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        optimizers = self.optimizers()
        # If multiple optimizers, take the first one
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """ Use AdamW and a SequentialLR scheduler (Warmup + Cosine Decay). """
        optimizer = AdamW(self.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=WEIGHT_DECAY)

        # Ensure setup has run and calculated steps_per_epoch
        if self.steps_per_epoch == 0:
             # If setup didn't run or failed (e.g. Trainer(fast_dev_run=True))
             # Use a reasonable fallback, but warn the user.
             self.steps_per_epoch = 500
             print(f"Warning: steps_per_epoch was 0 in configure_optimizers. Using fallback: {self.steps_per_epoch}. "
                   "This might happen with fast_dev_run or if setup failed.")


        warmup_epochs = getattr(self, "warmup_epochs", getattr(self.hparams, "warmup_epochs", WARMUP_EPOCHS))
        max_epochs = getattr(self, "max_epochs", getattr(self.hparams, "max_epochs", MAX_EPOCHS))

        warmup_steps = self.steps_per_epoch * warmup_epochs
        # Adjust total_steps calculation based on max_epochs potentially changing
        if self.trainer and hasattr(self.trainer, "max_epochs") and self.trainer.max_epochs is not None:
            max_epochs = self.trainer.max_epochs
        total_steps = self.steps_per_epoch * max_epochs
        decay_steps = total_steps - warmup_steps

        # Ensure decay_steps is not negative if warmup_epochs >= max_epochs
        decay_steps = max(1, decay_steps)

        print(f"Optimizer: AdamW, Initial LR: {self.hparams.learning_rate}")
        print(f"Scheduler: SequentialLR(Warmup + CosineAnnealingLR)")
        print(f"  Total estimated steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Cosine decay steps: {decay_steps}")


        scheduler_warmup = LambdaLR(
            optimizer,
            lr_lambda=linear_warmup_decay(warmup_steps)
        )
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=1e-7 # Slightly higher minimum LR
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps]
        )

        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]

########################################################
# 3) Main Function to Run Training (Using PyTorch Schedulers)
########################################################
# (Make sure BarlowTwinsProjectionHead and BarlowTwinsLoss are correctly defined/imported)
# Example Stubs:
class BarlowTwinsProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
    def forward(self, x):
        return self.layers(x)

class BarlowTwinsLoss(nn.Module):
     def __init__(self, lambda_param=5e-3):
         super().__init__()
         self.lambda_param = lambda_param
     def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
         z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-6) # Add epsilon for stability
         z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-6) # Add epsilon for stability
         N = z_a.size(0)
         D = z_a.size(1)
         c = torch.mm(z_a_norm.T, z_b_norm) / N
         c_diff = (c - torch.eye(D, device=c.device)).pow(2)
         c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
         loss = c_diff.sum()
         return loss

# 1) Define Transforms
transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=256, gaussian_blur=0.1),
    view_2_transform=BYOLView2Transform(input_size=256, gaussian_blur=0.1),
)

# Use torchvision.transforms.Compose, not monai.transforms.Compose
from torchvision.transforms import Compose
transform = Compose([transform])

# Convert ndarray to list for LightlyDataset compatibility
images_paths_list = ssl_images_paths_np.tolist()
# 2) Create Dataset
dataset = LightlyDataset(
    input_dir=INPUT_DIR,
    transform=transform,
    )
    # Optionally, you can pass labels if needed for debugging or logging

print(f"Dataset size: {len(dataset)}")

# 3) Create DataLoader
# Consider using pin_memory=True if using GPU and not seeing slowdowns
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    # pin_memory=True # Optional
)

# 4) Create the Barlow Twins model instance
# ---------- train ----------
import os
import pytorch_lightning as pl
# 5) Setup Callbacks
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar

lr_monitor = LearningRateMonitor(logging_interval='step')
progress_bar = TQDMProgressBar(refresh_rate=20)

def main():
    model = BarlowTwins(
        learning_rate=LEARNING_RATE,
        warmup_epochs=WARMUP_EPOCHS,
        max_epochs=MAX_EPOCHS,
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[
            progress_bar,
            lr_monitor,
        ],
    )
    
    trainer.fit(model, train_dataloaders=dataloader)

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.backbone.state_dict(),
               os.path.join(SAVE_DIR, "barlow_backbone_ext.pth"))

if __name__ == "__main__":
    main()
    
    