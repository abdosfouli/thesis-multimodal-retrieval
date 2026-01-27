#!/usr/bin/env python
"""
Stage 4: Ablations - isolate where gains come from
"""

import numpy as np
import json
import pickle
from src.data.fashioniq_loader import FashionIQLoader
from src.models.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.evaluation.metrics import recall_at_k

print("=" * 70)
print("STAGE 4: ABLATION STUDIES")
print("=" * 70)

# Load data
print("\n[1/4] Loading data...")
loader = FashionIQLoader(max_samples=2000)
texts, images = loader.load()

encoder = CLIPEncoder(device='cpu')
text_emb = encoder.encode_text(texts)
image_emb = encoder.encode_images(images)

# Build indexes
print("\n[2/4] Building indexes...")
concat_emb = np.concatenate([text_emb, image_emb], axis=1)
index_text = FAISSIndex(text_emb, name="text")
index_image = FAISSIndex(image_emb, name="image")
index_concat = FAISSIndex(concat_emb, name="concat")

# Load trained gate
print("\n[3/4] Loading trained gate...")
with open('models/intent_gate_trained.pkl', 'rb') as f:
    clf, scaler = pickle.load(f)

# Ablations: 4 configurations
print("\n[4/4] Running ablations (200 queries)...")

results = {}
ground_truth = np.zeros(200, dtype=int)

# 1. Text-only baseline
print("\n  Text-only baseline:")
recalls_text = []
for i in range(200):
    query = text_emb[i].reshape(1, -1)
    _, indices = index_text.search(query, k=10)
    recall = recall_at_k(indices, ground_truth[i:i+1], k=10)
    recalls_text.append(recall)
results['text_only'] = {'recall': np.mean(recalls_text), 'std': np.std(recalls_text)}
print(f"    Recall@10: {np.mean(recalls_text):.4f} ± {np.std(recalls_text):.4f}")

# 2. Image-only baseline
print("\n  Image-only baseline:")
recalls_image = []
for i in range(200):
    query = image_emb[i].reshape(1, -1)
    _, indices = index_image.search(query, k=10)
    recall = recall_at_k(indices, ground_truth[i:i+1], k=10)
    recalls_image.append(recall)
results['image_only'] = {'recall': np.mean(recalls_image), 'std': np.std(recalls_image)}
print(f"    Recall@10: {np.mean(recalls_image):.4f} ± {np.std(recalls_image):.4f}")

# 3. Always-fuse baseline (α=0.5)
print("\n  Always-fuse baseline (α=0.5):")
recalls_fuse = []
for i in range(200):
    query = concat_emb[i].reshape(1, -1)
    _, indices = index_concat.search(query, k=10)
    recall = recall_at_k(indices, ground_truth[i:i+1], k=10)
    recalls_fuse.append(recall)
results['always_fuse'] = {'recall': np.mean(recalls_fuse), 'std': np.std(recalls_fuse)}
print(f"    Recall@10: {np.mean(recalls_fuse):.4f} ± {np.std(recalls_fuse):.4f}")

# 4. Gate-based routing
print("\n  Gate-based routing:")
from src.models.intent_gate import IntentGate
gate = IntentGate()
gate_features = gate.extract_features(texts[:200], text_emb[:200], image_emb[:200])
gate_features_scaled = scaler.transform(gate_features)
predictions = clf.predict(gate_features_scaled)
alpha_map = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
predicted_alphas = np.array([alpha_map[p] for p in predictions])

recalls_gate = []
for i in range(200):
    alpha = predicted_alphas[i]
    query_text_part = alpha * text_emb[i]
    query_image_part = (1 - alpha) * image_emb[i]
    query = np.concatenate([query_text_part, query_image_part]).reshape(1, -1)
    _, indices = index_concat.search(query, k=10)
    recall = recall_at_k(indices, ground_truth[i:i+1], k=10)
    recalls_gate.append(recall)
results['gate_routing'] = {'recall': np.mean(recalls_gate), 'std': np.std(recalls_gate)}
print(f"    Recall@10: {np.mean(recalls_gate):.4f} ± {np.std(recalls_gate):.4f}")

# Summary
print("\n" + "=" * 70)
print("ABLATION RESULTS")
print("=" * 70)
print(f"{'Method':<20} {'Recall@10':<20} {'Std Dev':<15}")
print("-" * 70)
for method, metrics in sorted(results.items()):
    print(f"{method:<20} {metrics['recall']:<20.4f} {metrics['std']:<15.4f}")
print("=" * 70)

# Insights
print("\n📊 KEY FINDINGS:")
text_recall = results['text_only']['recall']
gate_recall = results['gate_routing']['recall']
always_fuse_recall = results['always_fuse']['recall']

gate_vs_text = (gate_recall - text_recall) * 100
gate_vs_fuse = (gate_recall - always_fuse_recall) * 100

print(f"  • Gate vs text-only:    {gate_vs_text:+.2f}% recall")
print(f"  • Gate vs always-fuse:  {gate_vs_fuse:+.2f}% recall")
print(f"  • Best baseline:        {max(results.items(), key=lambda x: x[1]['recall'])[0]}")

# Save results
with open('experiments/stage4_ablations.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Stage 4 ablations complete!")
print("✓ Results saved to: experiments/stage4_ablations.json")
