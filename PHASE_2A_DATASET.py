#!/usr/bin/env python3
import json
import torch
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("🎨 PHASE 2A: DATASET PREPARATION (OFFLINE)")
print("="*80)

print("\n[1/3] Creating fashion dataset...")
colors = ["red", "blue", "black", "white", "green", "yellow", "pink", "purple"]
textures = ["silk", "cotton", "wool", "polyester", "denim", "velvet"]
styles = ["casual", "formal", "sporty", "bohemian", "vintage"]
sleeves = ["sleeveless", "short", "long", "3/4"]
materials = ["natural", "synthetic", "blend", "leather"]

Path("data/retrieval").mkdir(parents=True, exist_ok=True)
Path("indexes").mkdir(parents=True, exist_ok=True)

dataset = []
for i in range(1000):
    idx = i % 8
    desc = f"{colors[idx % len(colors)]} {textures[idx % len(textures)]} {styles[idx % len(styles)]} with {sleeves[idx % len(sleeves)]} sleeves, {materials[idx % len(materials)]}"
    dataset.append({
        "id": f"img_{i:06d}",
        "description": desc,
        "color": colors[idx % len(colors)],
        "texture": textures[idx % len(textures)],
    })

with open("data/retrieval/fashion_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"✅ Created {len(dataset)} samples")

print("\n[2/3] Generating synthetic embeddings...")
# Generate random embeddings (simulating embeddings from fine-tuned PUMA)
# In a real scenario, these would come from the model
np.random.seed(42)
embedding_dim = 768
embeddings = np.random.randn(len(dataset), embedding_dim).astype('float32')

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print(f"✅ Generated embeddings: {embeddings.shape}")

print("\n[3/3] Saving embeddings and metadata...")
np.save("indexes/fashion_embeddings.npy", embeddings)

metadata = {
    "num_samples": 1000,
    "embedding_dim": embedding_dim,
    "embedding_type": "simulated_from_puma",
    "description": "Fashion retrieval embeddings"
}

with open("indexes/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Save dataset mapping
dataset_map = {item["id"]: item for item in dataset}
with open("indexes/dataset_map.json", "w") as f:
    json.dump(dataset_map, f, indent=2)

print(f"✅ Saved metadata and mapping")

print("\n" + "="*80)
print("✅ PHASE 2A COMPLETE!")
print("="*80)
print("\n📁 Created:")
print("   • data/retrieval/fashion_dataset.jsonl (1000 samples)")
print("   • indexes/fashion_embeddings.npy (1000 x 768)")
print("   • indexes/metadata.json")
print("   • indexes/dataset_map.json")
print("\n🚀 Ready for Phase 2B (Contrastive Learning)")
print()
