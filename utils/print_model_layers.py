#!/usr/bin/env python3
"""
Helper script to print model layers and suggest freeze points for transfer learning.

Usage:
    python utils/print_model_layers.py configs/pretrained_timm_4c/vit_small_patch16_224.yaml
    python utils/print_model_layers.py --model vit_base_patch16_224
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))


def analyze_model_layers(model_name: str, num_classes: int = 2, in_channels: int = 3):
    """
    Analyze and display model layer structure with freeze point suggestions.
    
    Args:
        model_name: Name of the model (e.g., 'vit_small_patch16_224')
        num_classes: Number of output classes
        in_channels: Number of input channels
    """
    # Import here to avoid initialization issues
    import timm
    
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Input channels: {in_channels}, Output classes: {num_classes}")
    print("=" * 80)
    
    # Create model
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=in_channels)
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Get all parameters
    layers = list(model.named_parameters())
    total_layers = len(layers)
    
    print(f"\nTotal layers: {total_layers}\n")
    
    # Print layers grouped by component
    print("Layer Structure:")
    print("-" * 80)
    
    current_block = None
    block_start = None
    
    for i, (name, param) in enumerate(layers):
        # Detect block changes
        if 'blocks.' in name:
            block_num = name.split('blocks.')[1].split('.')[0]
            if block_num != current_block:
                if current_block is not None and block_start is not None:
                    print(f"    └─ Block {current_block}: layers {block_start}-{i-1}")
                current_block = block_num
                block_start = i
        elif current_block is not None:
            print(f"    └─ Block {current_block}: layers {block_start}-{i-1}")
            current_block = None
            block_start = None
        
        # Print layer info with grouping
        indent = "    " if 'blocks.' in name else ""
        print(f"{indent}{i:3d}: {name:50s} | Shape: {str(list(param.shape)):20s}")
    
    # Close last block if needed
    if current_block is not None and block_start is not None:
        print(f"    └─ Block {current_block}: layers {block_start}-{total_layers-1}")
    
    print("-" * 80)
    
    # Provide freeze point suggestions
    print("\n" + "=" * 80)
    print("RECOMMENDED FREEZE POINTS FOR TRANSFER LEARNING")
    print("=" * 80)
    
    # Detect model type
    is_vit = 'vit' in model_name.lower() or 'deit' in model_name.lower()
    is_resnet = 'resnet' in model_name.lower()
    is_densenet = 'densenet' in model_name.lower()
    
    if is_vit:
        # Count transformer blocks
        num_blocks = 0
        for name, _ in layers:
            if 'blocks.' in name:
                block_num = int(name.split('blocks.')[1].split('.')[0])
                num_blocks = max(num_blocks, block_num + 1)
        
        print(f"\nDetected Vision Transformer with {num_blocks} blocks")
        print("\nSuggested freeze points:")
        
        # Calculate freeze points
        suggestions = [
            ("Linear Probe (freeze all except head)", total_layers - 3),
            (f"Freeze {num_blocks * 3 // 4} blocks (75%)", _find_block_end(layers, num_blocks * 3 // 4 - 1)),
            (f"Freeze {num_blocks // 2} blocks (50%) ← RECOMMENDED", _find_block_end(layers, num_blocks // 2 - 1)),
            (f"Freeze {num_blocks // 4} blocks (25%)", _find_block_end(layers, num_blocks // 4 - 1)),
            ("Full fine-tuning (no freezing)", -1),
        ]
        
        for desc, idx in suggestions:
            if idx == -1:
                print(f"  • freezed_layerIndex: {idx:3d}  → {desc}")
            else:
                trainable = total_layers - idx - 1
                frozen_pct = (idx + 1) / total_layers * 100
                print(f"  • freezed_layerIndex: {idx:3d}  → {desc}")
                print(f"      Frozen: {idx+1}/{total_layers} ({frozen_pct:.1f}%), Trainable: {trainable}")
    
    elif is_resnet:
        print("\nDetected ResNet architecture")
        print("\nSuggested freeze points:")
        
        # Find layer boundaries
        layer4_end = _find_last_matching(layers, 'layer4')
        layer3_end = _find_last_matching(layers, 'layer3')
        layer2_end = _find_last_matching(layers, 'layer2')
        layer1_end = _find_last_matching(layers, 'layer1')
        
        suggestions = [
            ("Freeze all except FC", total_layers - 3),
            ("Freeze conv1-layer3", layer3_end),
            ("Freeze conv1-layer2", layer2_end),
            ("Freeze conv1-layer1", layer1_end),
            ("Full fine-tuning", -1),
        ]
        
        for desc, idx in suggestions:
            if idx is not None:
                if idx == -1:
                    print(f"  • freezed_layerIndex: {idx:3d}  → {desc}")
                else:
                    trainable = total_layers - idx - 1
                    print(f"  • freezed_layerIndex: {idx:3d}  → {desc} (trainable: {trainable} layers)")
    
    elif is_densenet:
        print("\nDetected DenseNet architecture")
        print("\nSuggested freeze points:")
        
        # Find dense block boundaries
        block4_end = _find_last_matching(layers, 'denseblock4')
        block3_end = _find_last_matching(layers, 'denseblock3')
        block2_end = _find_last_matching(layers, 'denseblock2')
        
        suggestions = [
            ("Freeze all except classifier", total_layers - 3),
            ("Freeze blocks 1-3", block3_end),
            ("Freeze blocks 1-2", block2_end),
            ("Full fine-tuning", -1),
        ]
        
        for desc, idx in suggestions:
            if idx is not None:
                if idx == -1:
                    print(f"  • freezed_layerIndex: {idx:3d}  → {desc}")
                else:
                    trainable = total_layers - idx - 1
                    print(f"  • freezed_layerIndex: {idx:3d}  → {desc} (trainable: {trainable} layers)")
    
    else:
        print("\nGeneric suggestions:")
        suggestions = [
            ("Freeze 90%", int(total_layers * 0.9)),
            ("Freeze 75%", int(total_layers * 0.75)),
            ("Freeze 50%", int(total_layers * 0.5)),
            ("Freeze 25%", int(total_layers * 0.25)),
            ("No freezing", -1),
        ]
        
        for desc, idx in suggestions:
            if idx == -1:
                print(f"  • freezed_layerIndex: {idx:3d}  → {desc}")
            else:
                trainable = total_layers - idx - 1
                print(f"  • freezed_layerIndex: {idx:3d}  → {desc} (trainable: {trainable} layers)")
    
    print("\n" + "=" * 80)
    print("NOTE: These are starting points. Experiment to find what works best!")
    print("=" * 80)


def _find_block_end(layers, block_num):
    """Find the last layer index for a given transformer block number."""
    last_idx = -1
    for i, (name, _) in enumerate(layers):
        if f'blocks.{block_num}.' in name:
            last_idx = i
    return last_idx if last_idx != -1 else None


def _find_last_matching(layers, pattern):
    """Find the last layer index matching a pattern."""
    last_idx = None
    for i, (name, _) in enumerate(layers):
        if pattern in name:
            last_idx = i
    return last_idx


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model layers and suggest freeze points for transfer learning"
    )
    parser.add_argument(
        'config_or_model',
        type=str,
        help='Path to config YAML file or model name'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name (if providing model directly instead of config)'
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='Number of input channels (default: 3)'
    )
    parser.add_argument(
        '--classes',
        type=int,
        default=2,
        help='Number of output classes (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Determine if input is a config file or model name
    config_path = Path(args.config_or_model)
    
    if config_path.exists() and config_path.suffix in ['.yaml', '.yml']:
        # Load from config
        from configs.ConfigLoader import ConfigLoader
        print(f"Loading configuration from: {config_path}")
        cfg = ConfigLoader(str(config_path))
        model_name = cfg.get_model_name()
        in_channels = cfg.get_model_input_channels()
        num_classes = len(cfg.get_class_names())
        print(f"Extracted from config: model={model_name}, channels={in_channels}, classes={num_classes}\n")
    else:
        # Use model name directly
        model_name = args.model if args.model else args.config_or_model
        in_channels = args.channels
        num_classes = args.classes
        print(f"Using model: {model_name}, channels={in_channels}, classes={num_classes}\n")
    
    analyze_model_layers(model_name, num_classes, in_channels)


if __name__ == "__main__":
    main()

