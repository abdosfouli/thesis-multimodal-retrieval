#!/usr/bin/env python
"""
Stage 3: Intent Gate Foundation
(This is just the structure; training comes next)
"""

import numpy as np
from src.data.fashioniq_loader import FashionIQLoader
from src.models.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.models.intent_gate import IntentGate
from src.evaluation.metrics import recall_at_k, measure_latency

print("=" * 70)
print("STAGE 3: INTENT GATE FOUNDATION")
print("=" * 70)

# Load data
print("\n[1/5] Loading data...")
loader = FashionIQLoader(max_samples=10000)
texts, images = loader.load()

# Encode
print("\n[2/5] Encoding...")
encoder = CLIPEncoder(device='cpu')
text_emb = encoder.encode_text(texts)
image_emb = encoder.encode_images(images)

# Build indexes
print("\n[3/5] Building indexes...")
index_text = FAISSIndex(text_emb, name="text")
index_image = FAISSIndex(image_emb, name="image")
concat_emb = np.concatenate([text_emb, image_emb], axis=1)
index_concat = FAISSIndex(concat_emb, name="concat")

# Initialize gate
print("\n[4/5] Initializing Intent Gate...")
gate = IntentGate(num_queries=len(texts))

# Extract features
print("\nExtracting gate features...")
gate_features = gate.extract_features(texts, text_emb, image_emb)
print(f"Gate features shape: {gate_features.shape}")
print(f"Sample features (first 3 queries):\n{gate_features[:3]}")

# Make predictions (random for now)
print("\nMaking predictions (random, not trained yet)...")
predictions = gate.predict(gate_features)
alphas = gate.convert_to_alpha(predictions)

fusion_rate = (alphas > 0.0).sum() / len(alphas)
print(f"Fusion rate: {fusion_rate:.1%} (random predictions)")
print(f"Alpha distribution:")
print(f"  Text-only (α=0.0): {(alphas == 0.0).sum()}")
print(f"  Fused (α=0.5): {(alphas == 0.5).sum()}")
print(f"  Image-only (α=1.0): {(alphas == 1.0).sum()}")

# [5/5] Evaluate using gate
print("\n[5/5] Results (Foundation - random predictions):")
print("=" * 70)
print("Ready for Stage 3: Gate Training")
print("Next: Collect oracle labels and train gate classifier")
print("=" * 70)

print("\n✓ Stage 3 Foundation complete!")
print("\nNext steps:")
print("  1. Generate oracle labels (grid-search best alpha per query)")
print("  2. Train gate classifier")
print("  3. Validate gate accuracy vs oracle")
