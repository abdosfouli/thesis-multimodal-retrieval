#!/usr/bin/env python
"""
Stage 2: Add Reranker
"""

import numpy as np
import time
from src.data.fashioniq_loader import FashionIQLoader
from src.models.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.models.reranker import SimpleReranker
from src.evaluation.metrics import recall_at_k, measure_latency

print("=" * 70)
print("STAGE 2: RERANKER INTEGRATION")
print("=" * 70)

# Load 100k
print("\n[1/6] Loading 100k products...")
loader = FashionIQLoader(max_samples=100000)
texts, images = loader.load()

# Encode
print("\n[2/6] Encoding...")
encoder = CLIPEncoder(device='cpu')
text_emb = encoder.encode_text(texts)
image_emb = encoder.encode_images(images)

# Build indexes
print("\n[3/6] Building indexes...")
index_text = FAISSIndex(text_emb, name="text")
index_image = FAISSIndex(image_emb, name="image")
concat_emb = np.concatenate([text_emb, image_emb], axis=1)
index_concat = FAISSIndex(concat_emb, name="concat")

# Create reranker
print("\n[4/6] Creating reranker...")
reranker_text = SimpleReranker(text_emb, name="reranker_text")
reranker_concat = SimpleReranker(concat_emb, name="reranker_concat")

# Evaluate
print("\n[5/6] Evaluating WITHOUT and WITH reranker...")
num_queries = 500
query_text = text_emb[:num_queries]
query_concat = concat_emb[:num_queries]
ground_truth = np.zeros(num_queries, dtype=int)

results = {}

# WITHOUT reranker
print("\n  Without reranker:")
_, indices_text = index_text.search(query_text, k=100)  # Retrieve top-100
recall_text = recall_at_k(indices_text, ground_truth, k=10)
p50_text, p95_text = measure_latency(index_text, query_text, k=100, num_runs=10)
results['text_only_no_rerank'] = {'recall': recall_text, 'p50': p50_text, 'p95': p95_text}
print(f"    Text-only: Recall {recall_text:.4f}, P95 {p95_text:.2f}ms")

# WITH reranker (text)
print("\n  With reranker (text):")
_, indices_text_top100 = index_text.search(query_text, k=100)
indices_text_reranked = reranker_text.rerank(query_text, indices_text_top100, k=10)
recall_text_rerank = recall_at_k(indices_text_reranked, ground_truth, k=10)

def latency_with_rerank(index, reranker, query, k_initial=100, k_final=10, num_runs=10):
    times = []
    for _ in range(num_runs):
        start = time.time()
        _, indices = index.search(query, k=k_initial)
        reranked = reranker.rerank(query, indices, k=k_final)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    return np.percentile(times, 50), np.percentile(times, 95)

p50_rerank, p95_rerank = latency_with_rerank(index_text, reranker_text, query_text)
results['text_only_with_rerank'] = {'recall': recall_text_rerank, 'p50': p50_rerank, 'p95': p95_rerank}
print(f"    Text-only + rerank: Recall {recall_text_rerank:.4f}, P95 {p95_rerank:.2f}ms")

# Concat baseline
print("\n  Concatenation (baseline):")
_, indices_concat = index_concat.search(query_concat, k=10)
recall_concat = recall_at_k(indices_concat, ground_truth, k=10)
p50_concat, p95_concat = measure_latency(index_concat, query_concat, k=10, num_runs=10)
results['concat_no_rerank'] = {'recall': recall_concat, 'p50': p50_concat, 'p95': p95_concat}
print(f"    Concat: Recall {recall_concat:.4f}, P95 {p95_concat:.2f}ms")

# Concat + rerank
print("\n  Concatenation + reranker:")
_, indices_concat_top100 = index_concat.search(query_concat, k=100)
indices_concat_reranked = reranker_concat.rerank(query_concat, indices_concat_top100, k=10)
recall_concat_rerank = recall_at_k(indices_concat_reranked, ground_truth, k=10)
p50_rerank_concat, p95_rerank_concat = latency_with_rerank(index_concat, reranker_concat, query_concat)
results['concat_with_rerank'] = {'recall': recall_concat_rerank, 'p50': p50_rerank_concat, 'p95': p95_rerank_concat}
print(f"    Concat + rerank: Recall {recall_concat_rerank:.4f}, P95 {p95_rerank_concat:.2f}ms")

# Summary
print("\n[6/6] RESULTS:")
print("=" * 70)
print(f"{'Method':<30} {'Recall@10':<15} {'P50 (ms)':<15} {'P95 (ms)':<15}")
print("-" * 70)
for method, metrics in sorted(results.items()):
    print(f"{method:<30} {metrics['recall']:<15.4f} {metrics['p50']:<15.2f} {metrics['p95']:<15.2f}")
print("=" * 70)

# Analysis
print("\n📊 STAGE 2 INSIGHTS:")
rerank_recall_gain = (results['text_only_with_rerank']['recall'] - results['text_only_no_rerank']['recall']) * 100
rerank_latency_cost = results['text_only_with_rerank']['p95'] - results['text_only_no_rerank']['p95']
print(f"  • Reranker adds {rerank_recall_gain:.2f}% recall")
print(f"  • Reranker costs {rerank_latency_cost:.2f}ms latency")

# Save
import json
with open('experiments/stage2_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Stage 2 complete!")
