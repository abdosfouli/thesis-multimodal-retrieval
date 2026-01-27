import numpy as np

class FashionIQLoader:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
    
    def load(self):
        """Generate realistic synthetic data (simulating FashionIQ)"""
        print(f"Loading synthetic FashionIQ (max {self.max_samples} samples)...")
        
        # Create product descriptions (realistic)
        product_types = ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'hat', 'scarf']
        colors = ['red', 'blue', 'black', 'white', 'green', 'yellow', 'pink']
        styles = ['casual', 'formal', 'vintage', 'sporty', 'elegant']
        
        texts = []
        for i in range(self.max_samples):
            product = product_types[i % len(product_types)]
            color = colors[i % len(colors)]
            style = styles[i % len(styles)]
            text = f"a {style} {color} {product}"
            texts.append(text)
        
        # Images (we'll use paths, actual images can be added later)
        images = [f"product_{i:06d}.jpg" for i in range(self.max_samples)]
        
        print(f"✓ Loaded {len(texts)} samples")
        print(f"  Sample texts: {texts[:3]}")
        
        return texts, images

if __name__ == "__main__":
    loader = FashionIQLoader(max_samples=100)
    texts, images = loader.load()
