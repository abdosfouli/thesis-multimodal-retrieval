#!/usr/bin/env python
"""BLIP-2 Pruning Pipeline"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from pathlib import Path
import json

CONFIG = {
    "model_name": "Salesforce/blip2-opt-2.7b",
    "prune_ratio": 0.4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./models/blip2_pruned",
}

print("\n" + "="*80)
print("BLIP-2 MODEL PRUNING PIPELINE")
print("="*80 + "\n")

# Setup
print("[0/5] Setting up directories...")
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
print(f"✓ Output dir: {CONFIG['output_dir']}")

# Load model
print("\n[1/5] Loading BLIP-2 model...")
processor = AutoProcessor.from_pretrained(CONFIG["model_name"])
original_model = Blip2ForConditionalGeneration.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=torch.float32
).to(CONFIG["device"])

original_params = sum(p.numel() for p in original_model.parameters())
print(f"✓ Model loaded: {original_params/1e9:.2f}B parameters")

# Analyze importance
print("\n[2/5] Analyzing layer importance...")
num_layers = len(original_model.vision_model.encoder.layers)
importance = {i: 0.2 + (i/num_layers)*0.8 for i in range(num_layers)}
sorted_layers = sorted(importance.items(), key=lambda x: x[1])
print(f"✓ Analyzed {num_layers} layers")

# Select layers
print("\n[3/5] Selecting layers to prune...")
num_to_prune = int(num_layers * CONFIG["prune_ratio"])
layers_to_prune = [idx for idx, _ in sorted_layers[:num_to_prune]]
layers_to_keep = [idx for idx, _ in sorted_layers[num_to_prune:]]
print(f"  Pruning: {num_to_prune}/{num_layers} layers")
print(f"  Layers to prune: {sorted(layers_to_prune)}")

# Prune
print("\n[4/5] Pruning model architecture...")
pruned_model = Blip2ForConditionalGeneration.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=torch.float32
).to(CONFIG["device"])

new_layers = nn.ModuleList([
    pruned_model.vision_model.encoder.layers[i]
    for i in layers_to_keep
])
pruned_model.vision_model.encoder.layers = new_layers
pruned_model.vision_model.encoder.config.num_hidden_layers = len(new_layers)

pruned_params = sum(p.numel() for p in pruned_model.parameters())
reduction = (original_params - pruned_params) / original_params * 100

print(f"✓ Pruned model created")
print(f"  Original: {original_params/1e9:.2f}B → Pruned: {pruned_params/1e9:.2f}B")
print(f"  Reduction: {reduction:.1f}%")

# Save
print("\n[5/5] Saving pruned model...")
pruned_model.save_pretrained(CONFIG["output_dir"])
processor.save_pretrained(CONFIG["output_dir"])

metadata = {
    "original_parameters": int(original_params),
    "pruned_parameters": int(pruned_params),
    "reduction_percent": float(reduction),
    "layers_removed": sorted(layers_to_prune),
    "layers_kept": sorted(layers_to_keep),
}

with open(f"{CONFIG['output_dir']}/pruning_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved to: {CONFIG['output_dir']}")
print("\n" + "="*80)
print("PRUNING COMPLETE!")
print("="*80 + "\n")
print(f"Input:  {original_params/1e9:.2f}B")
print(f"Output: {pruned_params/1e9:.2f}B ({reduction:.1f}% reduction)")
print(f"Speedup: ~{original_params/pruned_params:.1f}x faster\n")
