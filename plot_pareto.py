#!/usr/bin/env python
"""
Generate Pareto curve (latency vs recall)
"""

import matplotlib.pyplot as plt
import numpy as np

# Your baseline results
results = {
    'text_only': {'recall': 0.01, 'p95': 10.42},
    'image_only': {'recall': 0.01, 'p95': 10.19},
    'concat': {'recall': 0.02, 'p95': 11.20},
}

# Create plot
plt.figure(figsize=(10, 6))

for method, metrics in results.items():
    plt.scatter(metrics['p95'], metrics['recall'], s=300, label=method, alpha=0.7)
    plt.annotate(method, (metrics['p95'], metrics['recall']), 
                xytext=(10, 10), textcoords='offset points', fontsize=10)

plt.xlabel('P95 Latency (ms)', fontsize=12)
plt.ylabel('Recall@10', fontsize=12)
plt.title('Pareto Curve: Latency vs Recall\n(Stage 1 Baselines)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Save
plt.savefig('experiments/stage1_pareto.png', dpi=150, bbox_inches='tight')
print("✓ Pareto curve saved: experiments/stage1_pareto.png")

plt.show()
