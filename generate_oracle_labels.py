#!/usr/bin/env python
"""
Generate oracle labels: for each query, find best alpha (modality choice)
"""

import numpy as np
import time
import json
from src.data.fashioniq_loader import FashionIQLoader
from src.models.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.evaluation.metrics import recall_at_k

print("=" * 70)
print("GENERATING ORACLE LABELS FOR GATE TRAINING")
print("=" * 70)

# Load data
print("\n[1/4] Loading data...")
loader = FashionIQLoader(max_samples=5000)  # Use 5k for oracle generation
texts, images = loader.load()

# Encode
print("\n[2/4] Encoding...")
encoder = CLIPEncoder(device='cpu')
text_emb = encoder.encode_text(texts)
image_emb = encoder.encode_images(images)

# Build indexes
print("\n[3/4] Building indexes...")
index_text = FAISSIndex(text_emb, name="text")
index_image = FAISSIndex(image_emb, name="image")

# Oracle label generation: Grid search
print("\n[4/4] Grid-searching best alpha for each query...")
print("Testing alpha values: [0.0, 0.25, 0.5, 0.75, 1.0]")

oracle_labels = {}
alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
ground_truth = np.zeros(len(texts), dtype=int)

latency_budget = 15.0  # ms (P95 latency budget)

for query_id in range(len(texts)):
    if query_id % 500 == 0:
        print(f"  Processing query {query_id}/{len(texts)}...")
    
    best_alpha = None
    best_recall = -1.0
    best_latency = 0.0
    
    # Try each alpha
    for alpha in alpha_values:
        # Create weighted query
        query = alpha * text_emb[query_id] + (1 - alpha) * image_emb[query_id]
        query = query.reshape(1, -1)
        
        # Search
        start = time.time()
        _, indices = index_text.search(query, k=10)
        latency = (time.time() - start) * 1000
        
        # Evaluate
        recall = recall_at_k(indices, ground_truth[query_id:query_id+1], k=10)
        
        # Choose best alpha within latency budget
        if latency <= latency_budget and recall > best_recall:
            best_recall = recall
            best_alpha = alpha
            best_latency = latency
    
    # Fallback: if no alpha meets budget, use fastest (text-only)
    if best_alpha is None:
        best_alpha = 0.0
    
    oracle_labels[query_id] = {
        'alpha': float(best_alpha),
        'recall': float(best_recall),
        'latency': float(best_latency),
    }

# Save oracle labels
print("\n" + "=" * 70)
print(f"✓ Generated oracle labels for {len(oracle_labels)} queries")

# Statistics
alphas_list = [v['alpha'] for v in oracle_labels.values()]
print(f"\nOracle statistics:")
print(f"  Text-only (α=0.0):    {sum(1 for a in alphas_list if a == 0.0)} queries")
print(f"  Fused (α=0.25-0.75): {sum(1 for a in alphas_list if 0 < a < 1)} queries")
print(f"  Image-only (α=1.0):   {sum(1 for a in alphas_list if a == 1.0)} queries")

# Save
with open('data/processed/oracle_labels.json', 'w') as f:
    json.dump(oracle_labels, f, indent=2)

print(f"\n✓ Oracle labels saved to: data/processed/oracle_labels.json")
print("=" * 70)
print("\nNext: Train Intent Gate using these oracle labels")
