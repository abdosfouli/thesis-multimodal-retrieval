import numpy as np

class CLIPEncoderReal:
    def __init__(self, device='cpu', use_mock=True):
        """
        Real CLIP encoder (will use actual models when available)
        """
        self.device = device
        self.use_mock = use_mock
        
        if use_mock:
            print(f"Using MOCK CLIP embeddings (device: {device})")
            print("(To use real CLIP: download models and set use_mock=False)")
        else:
            print(f"Loading real CLIP (device: {device})...")
            try:
                from transformers import CLIPModel, CLIPProcessor
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-packet32")
                print("✓ Real CLIP loaded!")
            except Exception as e:
                print(f"✗ Failed to load real CLIP: {e}")
                print("Falling back to mock embeddings")
                self.use_mock = True
    
    def encode_text(self, texts):
        if self.use_mock:
            return np.random.randn(len(texts), 512).astype('float32')
        else:
            # Real CLIP encoding
            import torch
            with torch.no_grad():
                inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()
    
    def encode_images(self, image_paths):
        if self.use_mock:
            return np.random.randn(len(image_paths), 512).astype('float32')
        else:
            # Real CLIP encoding
            import torch
            from PIL import Image
            with torch.no_grad():
                images = [Image.open(path) for path in image_paths]
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()

if __name__ == "__main__":
    encoder = CLIPEncoderReal(use_mock=True)
    text_emb = encoder.encode_text(["red shirt", "blue pants"])
    image_emb = encoder.encode_images(["img1.jpg", "img2.jpg"])
    print(f"Text: {text_emb.shape}, Image: {image_emb.shape}")
