# Quick Reference: Transfer Learning with ViT

## TL;DR - What to Do

**For your 4-channel medical imaging task with ~116 images:**

1. **Start with `freezed_layerIndex: 75`** (ViT-Small) or **`111`** (ViT-Base)
2. Use **AdamW optimizer** with **learning rate 3e-5 to 5e-5**
3. Use **weight decay 1e-4 to 2e-4**
4. Train for **100-200 epochs** with **early stopping patience 25-30**

## Quick Copy-Paste Config

```yaml
# configs/pretrained_timm_4c/vit_small_patch16_224.yaml
training:
  num_epochs: 200
  transfer_learning: true
  freezed_layerIndex: 75  # ← Change this value to experiment
  early_stopping_patience: 30

optimizer:
  learning_rate: 5e-5
  optimizer_name: "AdamW"
  weight_decay: 1e-4
```

## Freeze Point Cheat Sheet

### ViT-Small & ViT-Base (12 blocks)
| freezed_layerIndex | What's Frozen | What's Trainable | When to Use |
|---|---|---|---|
| **-1** | Nothing | Everything (152 layers) | Large dataset (>1000), very different task |
| **39** | First 3 blocks (25%) | Last 9 blocks | Medium dataset (500-1000) |
| **75** | First 6 blocks (50%) | Last 6 blocks | **Small dataset (100-500) ← YOUR CASE** |
| **111** | First 9 blocks (75%) | Last 3 blocks | Very small dataset (<100), or large model |
| **149** | All except head | Only head (2 layers) | Tiny dataset (<50), linear probe baseline |

### DeiT (Same as ViT)
DeiT has the same architecture as ViT, so use the same freeze points.

## How to Experiment

### Step 1: Baseline (Linear Probe)
```yaml
freezed_layerIndex: 149  # Train only the head
num_epochs: 50
```
**Purpose:** Quick baseline to see if pretrained features work at all.

### Step 2: Conservative Fine-Tuning (Recommended)
```yaml
freezed_layerIndex: 75   # or 111 for larger models
num_epochs: 200
```
**Purpose:** Balance between adaptation and overfitting prevention.

### Step 3: If Underfitting (Optional)
```yaml
freezed_layerIndex: 39   # Unfreeze more layers
num_epochs: 300
learning_rate: 3e-5      # Lower LR when training more layers
```
**Purpose:** Allow more adaptation if Step 2 shows the model can handle it.

## Common Issues & Solutions

### Issue: Training accuracy high, validation accuracy low
**Problem:** Overfitting  
**Solutions:**
1. Increase `freezed_layerIndex` (freeze more layers)
2. Increase `weight_decay`
3. Increase data augmentation
4. Reduce `num_epochs` or rely more on early stopping

### Issue: Both training and validation accuracy are low
**Problem:** Underfitting  
**Solutions:**
1. Decrease `freezed_layerIndex` (freeze fewer layers)
2. Increase `num_epochs`
3. Increase `learning_rate` slightly
4. Ensure data preprocessing is correct

### Issue: Validation loss stops improving quickly
**Problem:** Learning rate too high or too low  
**Solutions:**
1. Try `learning_rate: 1e-5` (if too high) or `1e-4` (if too low)
2. Check if scheduler is working properly
3. Visualize learning curves

## Scientific Rationale

**Why freeze early layers?**
- Early ViT blocks learn general visual patterns (edges, textures, basic shapes)
- These patterns transfer well across domains (ImageNet → Medical)
- Later blocks learn task-specific features that need adaptation

**Why use lower learning rate for fine-tuning?**
- Pretrained weights are already good
- Large updates can destroy useful patterns
- Small, careful updates preserve knowledge while adapting

**Why AdamW for ViT?**
- ViT paper used AdamW optimizer
- Better weight decay handling than Adam
- Proven to work well with transformers

## Files Ready to Use

All ViT configs in `configs/pretrained_timm_4c/` are now configured with:
✅ Recommended freeze points (commented with alternatives)
✅ Appropriate learning rates
✅ Proper weight decay
✅ Reasonable epoch counts

Just run:
```bash
bash local_run_scripts_DS1/pretrained_4c/run_local_vit_small_patch16_224_4c.sh
```

## Next Steps

1. **Run baseline** with `freezed_layerIndex: 149` (linear probe)
2. **Run main experiment** with `freezed_layerIndex: 75`
3. **Compare results**
4. **Adjust if needed** based on learning curves

---

**Questions?**
- Check `ViT_TRANSFER_LEARNING_GUIDE.md` for detailed explanations
- Look at learning curves in MLflow
- Compare with CNN baselines (ResNet, DenseNet)

