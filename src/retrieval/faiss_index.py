import numpy as np

class FAISSIndex:
    def __init__(self, embeddings, name="index"):
        """Simple nearest neighbor search (no FAISS needed for testing)"""
        self.embeddings = embeddings
        self.name = name
        print(f"Index '{name}' created with {len(embeddings)} items")
    
    def search(self, query_emb, k=10):
        """Search for top-k nearest neighbors using cosine similarity"""
        # Normalize
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
        db_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarities = query_norm @ db_norm.T  # (num_queries, num_db)
        
        # Top-k
        indices = np.argsort(-similarities, axis=1)[:, :k]
        distances = -np.sort(-similarities, axis=1)[:, :k]
        
        return distances, indices

if __name__ == "__main__":
    # Test
    embeddings = np.random.randn(1000, 512).astype('float32')
    index = FAISSIndex(embeddings, name="test_index")
    
    query = np.random.randn(10, 512).astype('float32')
    distances, indices = index.search(query, k=10)
    
    print(f"Query shape: {query.shape}")
    print(f"Top-10 indices shape: {indices.shape}")
    print(f"Top-10 distances shape: {distances.shape}")
