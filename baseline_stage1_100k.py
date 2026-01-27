#!/usr/bin/env python
"""
Stage 1: Scaled to 100k products
"""

import numpy as np
import time
from src.data.fashioniq_loader import FashionIQLoader
from src.models.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.evaluation.metrics import recall_at_k, measure_latency

print("=" * 70)
print("STAGE 1: BASELINE (100k products)")
print("=" * 70)

# Load 100k
print("\n[1/5] Loading 100k products...")
loader = FashionIQLoader(max_samples=100000)
texts, images = loader.load()

# Encode
print("\n[2/5] Encoding (this will take ~1-2 min)...")
encoder = CLIPEncoder(device='cpu')
text_emb = encoder.encode_text(texts)
image_emb = encoder.encode_images(images)
print(f"  Text: {text_emb.shape}, Image: {image_emb.shape}")

# Build indexes
print("\n[3/5] Building indexes...")
index_text = FAISSIndex(text_emb, name="text_100k")
index_image = FAISSIndex(image_emb, name="image_100k")
concat_emb = np.concatenate([text_emb, image_emb], axis=1)
index_concat = FAISSIndex(concat_emb, name="concat_100k")

# Evaluate (using 1000 queries for speed)
print("\n[4/5] Evaluating on 1000 queries...")
num_queries = 1000
query_text = text_emb[:num_queries]
query_image = image_emb[:num_queries]
query_concat = concat_emb[:num_queries]
ground_truth = np.zeros(num_queries, dtype=int)

results = {}

for name, index, query, key in [
    ("text_only", index_text, query_text, "text_only"),
    ("image_only", index_image, query_image, "image_only"),
    ("concat", index_concat, query_concat, "concat"),
]:
    print(f"\n  {name}:")
    _, indices = index.search(query, k=10)
    recall = recall_at_k(indices, ground_truth, k=10)
    p50, p95 = measure_latency(index, query, k=10, num_runs=10)
    results[key] = {'recall': recall, 'p50': p50, 'p95': p95}
    print(f"    Recall@10: {recall:.4f}, P50: {p50:.2f}ms, P95: {p95:.2f}ms")

# Summary
print("\n[5/5] RESULTS (100k products):")
print("=" * 70)
print(f"{'Method':<20} {'Recall@10':<15} {'P50 (ms)':<15} {'P95 (ms)':<15}")
print("-" * 70)
for method, metrics in results.items():
    print(f"{method:<20} {metrics['recall']:<15.4f} {metrics['p50']:<15.2f} {metrics['p95']:<15.2f}")
print("=" * 70)

print("\n✓ Stage 1 (100k) complete!")

# Save results
import json
with open('experiments/stage1_100k_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved to experiments/stage1_100k_results.json")
