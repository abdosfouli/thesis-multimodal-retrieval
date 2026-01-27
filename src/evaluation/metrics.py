import numpy as np
import time

def recall_at_k(retrieved_indices, ground_truth_indices, k=10):
    """Calculate Recall@K"""
    recall_sum = 0.0
    for i, retrieved in enumerate(retrieved_indices):
        if ground_truth_indices[i] in retrieved[:k]:
            recall_sum += 1.0
    return recall_sum / len(retrieved_indices)

def measure_latency(index, query_emb, k=10, num_runs=10):
    """Measure latency of search"""
    times = []
    for _ in range(num_runs):
        start = time.time()
        index.search(query_emb, k=k)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
    
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    return p50, p95

if __name__ == "__main__":
    # Test
    retrieved = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    ground_truth = np.array([1])
    
    recall = recall_at_k(retrieved, ground_truth, k=10)
    print(f"Recall@10: {recall:.2f}")
