# Vision Transformer (ViT) Transfer Learning Guide

## Understanding freezed_layerIndex

The `freezed_layerIndex` parameter determines which layers to freeze during training:
- **-1**: Don't freeze anything (full fine-tuning)
- **0 to N**: Freeze all layers from 0 up to and including index N

## ViT Architecture (vit_small_patch16_224)

```
Layers 0-3:     Patch Embedding (cls_token, pos_embed, patch_embed)
Layers 4-15:    Transformer Block 0
Layers 16-27:   Transformer Block 1
Layers 28-39:   Transformer Block 2
Layers 40-51:   Transformer Block 3
Layers 52-63:   Transformer Block 4
Layers 64-75:   Transformer Block 5
Layers 76-87:   Transformer Block 6
Layers 88-99:   Transformer Block 7
Layers 100-111: Transformer Block 8
Layers 112-123: Transformer Block 9
Layers 124-135: Transformer Block 10
Layers 136-147: Transformer Block 11
Layers 148-149: Final Norm
Layers 150-151: Classification Head
```

## Recommended Strategies

### 1. Linear Probe (Fastest, Least Adaptation)
```yaml
freezed_layerIndex: 149  # Freeze everything except the head
```
**Use when:**
- You have very little data (<100 samples)
- Your task is similar to ImageNet classification
- You want fastest training/inference
- Risk of overfitting is high

**Pros:** Very fast, prevents overfitting
**Cons:** Limited adaptation to your domain

---

### 2. Freeze Early Layers (Recommended for Small Datasets)
```yaml
freezed_layerIndex: 75  # Freeze patch embedding + first 6 transformer blocks
```
**Use when:**
- Small dataset (100-500 samples) ← **YOUR CASE**
- Task is moderately different from ImageNet
- You want balance between speed and adaptation

**Pros:** Good balance, reasonable training time
**Cons:** May not fully adapt low-level features

---

### 3. Freeze Patch Embedding Only
```yaml
freezed_layerIndex: 3  # Freeze only patch embedding
```
**Use when:**
- Medium dataset (500-2000 samples)
- Task requires different feature representations
- You have computational resources

**Pros:** Better adaptation while keeping patch extraction stable
**Cons:** Slower training, more parameters to tune

---

### 4. Full Fine-Tuning (Slowest, Most Adaptation)
```yaml
freezed_layerIndex: -1  # Don't freeze anything
```
**Use when:**
- Large dataset (>2000 samples)
- Task is very different from ImageNet
- You have significant computational resources
- Using strong regularization (dropout, weight decay)

**Pros:** Maximum adaptation to your task
**Cons:** Slow, risk of overfitting, requires careful tuning

---

## For 4-Channel Medical Images (Your Use Case)

**Recommended:** Start with **freezed_layerIndex: 75** (freeze first 6 blocks)

**Why?**
- You have ~116 images (small dataset)
- Medical images are different from ImageNet
- 4-channel input → first layer is already adapted
- Last 6 transformer blocks can learn medical-specific patterns

**Alternative strategies to try:**
1. `freezed_layerIndex: 111` (freeze first 9 blocks) - more frozen
2. `freezed_layerIndex: 39` (freeze first 3 blocks) - less frozen
3. `freezed_layerIndex: 149` (linear probe) - most frozen

---

## How to Find the Right Index

### Method 1: Use the Helper Script

Run this to find layer boundaries:
```bash
cd /home/zano/Documents/TESI/FOLDER_CINECA
source .venv/bin/activate
python utils/print_model_layers.py configs/pretrained_timm_4c/vit_small_patch16_224.yaml
```

### Method 2: Calculate Manually

For ViT-Small (12 blocks):
- **Freeze N blocks**: `freezed_layerIndex = 3 + (N * 12) + 11`
  - Freeze 3 blocks: `3 + (3 × 12) + 11 = 50`
  - Freeze 6 blocks: `3 + (6 × 12) + 11 = 86`
  - Freeze 9 blocks: `3 + (9 × 12) + 11 = 122`

For ViT-Base (12 blocks, similar structure):
- Same formula applies

---

## Practical Tips

1. **Start Conservative**: Begin with more frozen layers (e.g., 75), then gradually unfreeze if you have capacity

2. **Use Learning Rate Scaling**: If fine-tuning unfrozen layers, use:
   ```yaml
   backbone_lr_mult: 0.1  # Unfrozen layers learn 10x slower
   ```

3. **Monitor Overfitting**: If validation accuracy stops improving but training keeps going up:
   - Freeze more layers
   - Increase regularization
   - Use more data augmentation

4. **Experiment**: Run a few quick experiments with different freeze points:
   ```bash
   # In your YAML config, try:
   freezed_layerIndex: 149  # Linear probe
   freezed_layerIndex: 111  # Freeze 9 blocks
   freezed_layerIndex: 75   # Freeze 6 blocks  ← START HERE
   freezed_layerIndex: 39   # Freeze 3 blocks
   freezed_layerIndex: -1   # Full fine-tuning
   ```

---

## Scientific Reasoning

**From literature:**
- **Kornblith et al. (2019)**: Early layers learn universal features, later layers are task-specific
- **Dosovitskiy et al. (2020)**: ViT benefits from fine-tuning on target domain
- **He et al. (2021)**: For small datasets, freezing helps prevent overfitting

**For medical imaging:**
- Medical images have different statistics than ImageNet
- But low-level features (edges, textures) are still useful
- **Balance**: Freeze low-level, fine-tune high-level

---

## Example Configurations

### Example 1: Conservative (Recommended Start)
```yaml
# configs/pretrained_timm_4c/vit_small_patch16_224.yaml
training:
  transfer_learning: true
  freezed_layerIndex: 75  # Freeze first 6 blocks
  num_epochs: 200
  early_stopping_patience: 30

optimizer:
  learning_rate: 1e-4
  weight_decay: 1e-4
```

### Example 2: Aggressive Fine-Tuning
```yaml
# For if you have more data or need more adaptation
training:
  transfer_learning: true
  freezed_layerIndex: 3  # Only freeze patch embedding
  num_epochs: 300
  early_stopping_patience: 50

optimizer:
  learning_rate: 5e-5  # Lower LR for more fine-tuning
  weight_decay: 2e-4   # More regularization
```

### Example 3: Linear Probe Baseline
```yaml
# Quick baseline to see if pretrained features work
training:
  transfer_learning: true
  freezed_layerIndex: 149  # Only train head
  num_epochs: 50
  early_stopping_patience: 15

optimizer:
  learning_rate: 1e-3  # Higher LR for head-only
  weight_decay: 1e-5
```

