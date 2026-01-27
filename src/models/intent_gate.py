import numpy as np

class IntentGate:
    """
    Simple intent gate that decides modality routing
    learns: text-only (0), image-only (1), or fused (2)
    """
    
    def __init__(self, num_queries=1000):
        print("Intent Gate initialized")
        print("(Will be trained in Stage 3)")
    
    def extract_features(self, texts, text_embeddings, image_embeddings):
        """
        Extract 7 cheap features for gating
        """
        features = []
        
        for i, text in enumerate(texts):
            # Feature 1: Caption length
            caption_len = len(text.split()) / 20.0  # normalize
            
            # Feature 2: Has color word
            colors = {'red', 'blue', 'black', 'white', 'green', 'yellow', 'pink'}
            has_color = 1.0 if any(c in text.lower() for c in colors) else 0.0
            
            # Feature 3: Has attribute word
            attrs = {'small', 'large', 'long', 'short', 'tight'}
            has_attr = 1.0 if any(a in text.lower() for a in attrs) else 0.0
            
            # Feature 4: Text ambiguity
            ambiguity = 1.0 / (1.0 + len(text.split()))
            
            # Feature 5: Text-image similarity gap
            text_norm = text_embeddings[i] / (np.linalg.norm(text_embeddings[i]) + 1e-8)
            img_norm = image_embeddings[i] / (np.linalg.norm(image_embeddings[i]) + 1e-8)
            sim = np.dot(text_norm, img_norm)
            sim_norm = (sim + 1.0) / 2.0  # normalize to [0, 1]
            
            # Feature 6: Text specificity
            unique_ratio = len(set(text.split())) / (len(text.split()) + 1e-8)
            
            # Feature 7: Dummy feature
            dummy = np.random.rand()
            
            features.append([caption_len, has_color, has_attr, ambiguity, sim_norm, unique_ratio, dummy])
        
        return np.array(features, dtype='float32')
    
    def predict(self, features):
        """
        Predict modality choice: 0=text, 1=image, 2=fused
        (For now, just return random; will be trained in Stage 3)
        """
        num_queries = len(features)
        predictions = np.random.choice([0, 1, 2], size=num_queries)
        return predictions
    
    def convert_to_alpha(self, predictions):
        """Convert predictions to alpha (fusion weight)"""
        alpha_map = {0: 0.0, 1: 1.0, 2: 0.5}
        return np.array([alpha_map[p] for p in predictions])

if __name__ == "__main__":
    gate = IntentGate(num_queries=100)
    
    texts = ["a red shirt", "blue pants", "black jacket"]
    text_emb = np.random.randn(3, 512)
    image_emb = np.random.randn(3, 512)
    
    features = gate.extract_features(texts, text_emb, image_emb)
    print(f"Features shape: {features.shape}")
    
    predictions = gate.predict(features)
    alphas = gate.convert_to_alpha(predictions)
    print(f"Predictions: {predictions}")
    print(f"Alpha values: {alphas}")
