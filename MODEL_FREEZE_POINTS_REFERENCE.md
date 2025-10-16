# Model Layer Structure & Recommended Freeze Points

Quick reference for transfer learning with 4-channel medical images (~116 samples).

---

## Vision Transformers (ViT & DeiT)

All ViT and DeiT models have **152 total layers** organized into **12 transformer blocks**.

| Model | Total Layers | Blocks | Conservative Freeze | Moderate Freeze |
|-------|--------------|--------|---------------------|-----------------|
| `vit_small_patch16_224` | 152 | 12 | **111** (9 blocks) | **75** (6 blocks) |
| `vit_base_patch16_224` | 152 | 12 | **111** (9 blocks) | **75** (6 blocks) |
| `vit_base_patch16_384` | 152 | 12 | **111** (9 blocks) | **75** (6 blocks) |
| `deit_small_patch16_224` | 152 | 12 | **111** (9 blocks) | **75** (6 blocks) |
| `deit_base_patch16_224` | 152 | 12 | **111** (9 blocks) | **75** (6 blocks) |

### Layer Structure
```
Layers 0-3:     Patch embedding (cls_token, pos_embed, conv)
Layers 4-147:   12 Transformer blocks (12 layers each)
  • Block 0:  layers 4-15
  • Block 1:  layers 16-27
  • Block 2:  layers 28-39
  • Block 3:  layers 40-51
  • Block 4:  layers 52-63
  • Block 5:  layers 64-75   ← Freeze to here (moderate)
  • Block 6:  layers 76-87
  • Block 7:  layers 88-99
  • Block 8:  layers 100-111 ← Freeze to here (conservative)
  • Block 9:  layers 112-123
  • Block 10: layers 124-135
  • Block 11: layers 136-147
Layers 148-149: Final normalization
Layers 150-151: Classification head
```

### Recommendations
- **Start with `freezed_layerIndex: 75`** for small models (ViT-Small, DeiT-Small)
- **Start with `freezed_layerIndex: 111`** for large models (ViT-Base, DeiT-Base)
- Use **AdamW optimizer** with **lr: 3e-5 to 5e-5**

---

## ResNet18

**Total layers:** 62

| What to Freeze | freezed_layerIndex | Trainable Layers | Use Case |
|----------------|-------------------|------------------|----------|
| Through layer2 (conservative) | **43** | 19 layers | Small dataset, prevent overfitting |
| Through layer1 (moderate) | **28** | 34 layers | Balanced adaptation |

### Layer Structure
```
Layers 0-13:   conv1, bn1
Layers 14-29:  layer1 (8 conv blocks)
  └─ Ends at layer 28 ← Freeze to here (moderate)
Layers 30-44:  layer2 (8 conv blocks)
  └─ Ends at layer 43 ← Freeze to here (conservative)
Layers 45-59:  layer3 (8 conv blocks)
  └─ Ends at layer 58
Layers 60-61:  layer4 (8 conv blocks)
  └─ Ends at layer 59
Layers 60-61:  fc (classification head)
```

### Recommendations
- **Start with `freezed_layerIndex: 43`** (freeze through layer2)
- Use **Adam/AdamW** with **lr: 5e-5 to 1e-4**

---

## ResNet50

**Total layers:** 161

| What to Freeze | freezed_layerIndex | Trainable Layers | Use Case |
|----------------|-------------------|------------------|----------|
| Through layer2 (conservative) | **127** | 34 layers | Small dataset, large model |
| Through layer1 (moderate) | **70** | 91 layers | More aggressive fine-tuning |

### Layer Structure
```
Layers 0-31:    conv1, bn1
Layers 32-71:   layer1 (bottleneck blocks)
  └─ Ends at layer 70 ← Freeze to here (moderate)
Layers 72-128:  layer2 (bottleneck blocks)
  └─ Ends at layer 127 ← Freeze to here (conservative)
Layers 129-158: layer3 (bottleneck blocks)
  └─ Ends at layer 157
Layers 159-160: layer4 (bottleneck blocks)
  └─ Ends at layer 158
Layers 159-160: fc (classification head)
```

### Recommendations
- **Start with `freezed_layerIndex: 127`** (freeze through layer2)
- Use **Adam/AdamW** with **lr: 3e-5 to 5e-5** (lower for larger model)

---

## DenseNet121

**Total layers:** 364

| What to Freeze | freezed_layerIndex | Trainable Layers | Use Case |
|----------------|-------------------|------------------|----------|
| Through denseblock3 (conservative) | **263** | 101 layers | Small dataset, prevent overfitting |
| Through denseblock2 (moderate) | **116** | 248 layers | Balanced adaptation |

### Layer Structure
```
Layers 0-5:     Initial conv, bn, relu, pool
Layers 6-41:    denseblock1 (6 dense layers)
  └─ Ends at layer 41
Layers 42-116:  denseblock2 (12 dense layers)
  └─ Ends at layer 116 ← Freeze to here (moderate)
Layers 117-263: denseblock3 (24 dense layers)
  └─ Ends at layer 263 ← Freeze to here (conservative)
Layers 264-361: denseblock4 (16 dense layers)
  └─ Ends at layer 361
Layers 362-363: classifier (bn, fc)
```

### Recommendations
- **Start with `freezed_layerIndex: 263`** (freeze through denseblock3)
- Use **Adam/AdamW** with **lr: 5e-5 to 1e-4**

---

## DenseNet169

**Total layers:** 508

| What to Freeze | freezed_layerIndex | Trainable Layers | Use Case |
|----------------|-------------------|------------------|----------|
| Through denseblock3 (conservative) | **311** | 197 layers | Small dataset, large model |
| Through denseblock2 (moderate) | **116** | 392 layers | More aggressive fine-tuning |

### Layer Structure
```
Layers 0-5:     Initial conv, bn, relu, pool
Layers 6-41:    denseblock1 (6 dense layers)
  └─ Ends at layer 41
Layers 42-116:  denseblock2 (12 dense layers)
  └─ Ends at layer 116 ← Freeze to here (moderate)
Layers 117-311: denseblock3 (32 dense layers)
  └─ Ends at layer 311 ← Freeze to here (conservative)
Layers 312-505: denseblock4 (32 dense layers)
  └─ Ends at layer 505
Layers 506-507: classifier (bn, fc)
```

### Recommendations
- **Start with `freezed_layerIndex: 311`** (freeze through denseblock3)
- Use **Adam/AdamW** with **lr: 3e-5 to 5e-5** (lower for larger model)

---

## Summary Table

| Model | Total Layers | Conservative Freeze | Moderate Freeze | Recommended LR |
|-------|--------------|---------------------|-----------------|----------------|
| **ViT-Small** | 152 | 111 (75% frozen) | 75 (50% frozen) | 5e-5 |
| **ViT-Base** | 152 | 111 (75% frozen) | 75 (50% frozen) | 3e-5 |
| **DeiT-Small** | 152 | 111 (75% frozen) | 75 (50% frozen) | 5e-5 |
| **DeiT-Base** | 152 | 111 (75% frozen) | 75 (50% frozen) | 3e-5 |
| **ResNet18** | 62 | 43 (69% frozen) | 28 (45% frozen) | 5e-5 |
| **ResNet50** | 161 | 127 (79% frozen) | 70 (43% frozen) | 3e-5 |
| **DenseNet121** | 364 | 263 (72% frozen) | 116 (32% frozen) | 5e-5 |
| **DenseNet169** | 508 | 311 (61% frozen) | 116 (23% frozen) | 3e-5 |

---

## Usage in YAML Config

```yaml
training:
  transfer_learning: true
  freezed_layerIndex: 75  # ← Change this value based on table above

optimizer:
  learning_rate: 5e-5     # ← Adjust based on table above
  optimizer_name: "AdamW"
  weight_decay: 1e-4
```

---

## General Guidelines

### When to use Conservative (more freezing)
- ✅ Very small dataset (<100 samples)
- ✅ Large model (Base/169/50)
- ✅ Task very similar to ImageNet
- ✅ Risk of overfitting is high

### When to use Moderate (less freezing)
- ✅ Small-medium dataset (100-500 samples) ← **Your case**
- ✅ Smaller model (Small/121/18)
- ✅ Task moderately different from ImageNet
- ✅ Model shows signs of underfitting

### Optimizer Settings
- **AdamW** preferred for transformers (ViT, DeiT)
- **Adam** or **AdamW** for CNNs (ResNet, DenseNet)
- **Weight decay:** 1e-4 to 2e-4
- **Epochs:** 150-250 with early stopping

---

## Quick Start Commands

```bash
# ViT models
bash local_run_scripts_DS1/pretrained_4c/run_local_vit_small_patch16_224_4c.sh
bash local_run_scripts_DS1/pretrained_4c/run_local_deit_small_patch16_224_4c.sh

# CNN models  
bash local_run_scripts_DS1/pretrained_4c/run_local_resnet18_4c.sh
bash local_run_scripts_DS1/pretrained_4c/run_local_resnet50_4c.sh
bash local_run_scripts_DS1/pretrained_4c/run_local_densenet121_4c.sh
bash local_run_scripts_DS1/pretrained_4c/run_local_densenet169_4c.sh
```

---

**Note:** These are starting points. Monitor your learning curves and adjust based on:
- Training vs validation accuracy gap (overfitting → freeze more)
- Both accuracies low (underfitting → freeze less)
- Learning rate schedules in MLflow logs

