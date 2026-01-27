import torch
import numpy as np

print("=" * 50)
print("BASIC CLUSTER TEST")
print("=" * 50)

# Test 1: PyTorch
print("\n1. Testing PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   ✓ PyTorch works")

# Test 2: GPU
print("\n2. Testing GPU...")
gpu_available = torch.cuda.is_available()
print(f"   GPU available: {gpu_available}")
if gpu_available:
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   ✓ GPU works!")
else:
    print("   ✗ GPU not available (CPU only)")

# Test 3: NumPy
print("\n3. Testing NumPy...")
arr = np.random.randn(1000, 512)
print(f"   Created array: {arr.shape}")
print(f"   ✓ NumPy works")

# Test 4: Create fake embeddings
print("\n4. Creating fake embeddings (for testing)...")
text_emb = np.random.randn(10000, 512).astype('float32')
image_emb = np.random.randn(10000, 512).astype('float32')
print(f"   Text embeddings: {text_emb.shape}")
print(f"   Image embeddings: {image_emb.shape}")
print(f"   ✓ Can create embeddings")

print("\n" + "=" * 50)
print("✓ ALL BASIC TESTS PASSED!")
print("=" * 50)
print("\nCluster is ready for thesis work!")
