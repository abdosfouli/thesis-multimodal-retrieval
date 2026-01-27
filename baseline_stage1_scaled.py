#!/usr/bin/env python
"""
Stage 1: Scaled Baseline (10k products)
"""

import numpy as np
import time
from src.data.fashioniq_loader import FashionIQLoader
from src.models.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.evaluation.metrics import recall_at_k, measure_latency

print("=" * 70)
print("STAGE 1: SCALED BASELINE (10k products)")
print("=" * 70)

# Load data
print("\n[1/5] Loading data...")
loader = FashionIQLoader(max_samples=10000)
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
concat_emb = np.concatenate([text_emb, image_emb], axis=1)
index_concat = FAISSIndex(concat_emb, name="concat_index")

# Evaluate
print("\n[4/5] Evaluating baselines...")
num_queries = 500
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
    p50, p95 = measure_latency(index, query, k=10, num_runs=20)
    results[key] = {'recall': recall, 'p50': p50, 'p95': p95}
    print(f"    Recall@10: {recall:.4f}")
    print(f"    P50: {p50:.2f}ms, P95: {p95:.2f}ms")

# Summary
print("\n[5/5] Results Summary (10k products):")
print("=" * 70)
print(f"{'Method':<20} {'Recall@10':<15} {'P50 (ms)':<15} {'P95 (ms)':<15}")
print("-" * 70)
for method, metrics in results.items():
    print(f"{method:<20} {metrics['recall']:<15.4f} {metrics['p50']:<15.2f} {metrics['p95']:<15.2f}")
print("=" * 70)

# Analysis
print("\n📊 KEY INSIGHTS:")
concat_recall_gain = (results['concat']['recall'] - results['text_only']['recall']) * 100
concat_latency_cost = results['concat']['p95'] - results['text_only']['p95']
print(f"  • Fusion adds {concat_recall_gain:.2f}% recall")
print(f"  • Fusion costs {concat_latency_cost:.2f}ms (P95 latency)")
print(f"  • Tradeoff: {concat_recall_gain / (concat_latency_cost + 1e-8):.2f} recall% per ms")

print("\n✓ Stage 1 complete! Ready for:")
print("  → Stage 2: Add reranker (ColBERT)")
print("  → Stage 3: Implement intent gate")
print("  → Stage 4: Ablations & error analysis")
