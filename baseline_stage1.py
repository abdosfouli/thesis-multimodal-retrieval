#!/usr/bin/env python
"""
Stage 1: Baseline Reproduction
Test text-only, image-only, and concatenation baselines
"""

import numpy as np
import time
from src.data.fashioniq_loader import FashionIQLoader
from src.models.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.evaluation.metrics import recall_at_k, measure_latency

print("=" * 60)
print("STAGE 1: BASELINE REPRODUCTION")
print("=" * 60)

# Load data
print("\n[1/5] Loading data...")
loader = FashionIQLoader(max_samples=1000)
texts, images = loader.load()

# Encode
print("\n[2/5] Encoding texts and images...")
encoder = CLIPEncoder(device='cpu')
text_emb = encoder.encode_text(texts)
image_emb = encoder.encode_images(images)
print(f"  Text embeddings: {text_emb.shape}")
print(f"  Image embeddings: {image_emb.shape}")

# Build indexes
print("\n[3/5] Building indexes...")
index_text = FAISSIndex(text_emb, name="text_index")
index_image = FAISSIndex(image_emb, name="image_index")

# Concatenate for fusion baseline
concat_emb = np.concatenate([text_emb, image_emb], axis=1)
index_concat = FAISSIndex(concat_emb, name="concat_index")

# Evaluate
print("\n[4/5] Evaluating baselines...")
num_queries = 100
query_text = text_emb[:num_queries]
query_image = image_emb[:num_queries]
query_concat = concat_emb[:num_queries]

# Generate ground truth (first item in DB is relevant)
ground_truth = np.zeros(num_queries, dtype=int)

results = {}

# Text-only baseline
print("\n  Text-only baseline:")
_, indices_text = index_text.search(query_text, k=10)
recall_text = recall_at_k(indices_text, ground_truth, k=10)
p50_text, p95_text = measure_latency(index_text, query_text, k=10, num_runs=10)
results['text_only'] = {'recall': recall_text, 'p50': p50_text, 'p95': p95_text}
print(f"    Recall@10: {recall_text:.4f}")
print(f"    P50 latency: {p50_text:.2f}ms")
print(f"    P95 latency: {p95_text:.2f}ms")

# Image-only baseline
print("\n  Image-only baseline:")
_, indices_image = index_image.search(query_image, k=10)
recall_image = recall_at_k(indices_image, ground_truth, k=10)
p50_image, p95_image = measure_latency(index_image, query_image, k=10, num_runs=10)
results['image_only'] = {'recall': recall_image, 'p50': p50_image, 'p95': p95_image}
print(f"    Recall@10: {recall_image:.4f}")
print(f"    P50 latency: {p50_image:.2f}ms")
print(f"    P95 latency: {p95_image:.2f}ms")

# Concatenation baseline
print("\n  Concatenation baseline:")
_, indices_concat = index_concat.search(query_concat, k=10)
recall_concat = recall_at_k(indices_concat, ground_truth, k=10)
p50_concat, p95_concat = measure_latency(index_concat, query_concat, k=10, num_runs=10)
results['concat'] = {'recall': recall_concat, 'p50': p50_concat, 'p95': p95_concat}
print(f"    Recall@10: {recall_concat:.4f}")
print(f"    P50 latency: {p50_concat:.2f}ms")
print(f"    P95 latency: {p95_concat:.2f}ms")

# Print summary
print("\n[5/5] Results Summary:")
print("=" * 60)
print(f"{'Method':<20} {'Recall@10':<15} {'P50 (ms)':<15} {'P95 (ms)':<15}")
print("-" * 60)
for method, metrics in results.items():
    print(f"{method:<20} {metrics['recall']:<15.4f} {metrics['p50']:<15.2f} {metrics['p95']:<15.2f}")
print("=" * 60)

print("\n✓ Stage 1 baseline complete!")
print("\nNext steps:")
print("  1. Scale to 10k products")
print("  2. Build FAISS GPU indexes")
print("  3. Generate Pareto curves")
print("  4. Implement intent gate")
