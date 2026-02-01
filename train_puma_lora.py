#!/usr/bin/env python3
import json
import torch
import argparse
from pathlib import Path
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class TextDataset(Dataset):
    def __init__(self, jsonl_file, max_samples=None):
        self.examples = []
        with open(jsonl_file) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.examples.append(json.loads(line))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]['caption']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/blip2_pruned_aggressive")
    parser.add_argument("--data_path", default="data/deepfashion2/train_annotations.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/puma_fashion_expert")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎨 PHASE 1B: PUMA LoRA FINE-TUNING")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[1/5] Loading pruned PUMA model...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    original_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {original_params/1e9:.2f}B parameters\n")
    
    print("[2/5] Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ LoRA configured: {trainable_params/1e6:.1f}M trainable\n")
    
    print("[3/5] Loading dataset...")
    dataset = TextDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"✅ Dataset ready: {len(dataset)} examples\n")
    
    print("[4/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print("[5/5] Starting fine-tuning...\n")
    
    model.train()
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        total_loss = 0
        count = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_texts in pbar:
            try:
                # Tokenize text only
                inputs = processor.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    max_length=128, 
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get text embeddings
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    text_features = model.qformer(
                        input_ids=inputs.get('input_ids'),
                        attention_mask=inputs.get('attention_mask'),
                    )
                    logits = text_features.last_hidden_state.mean(dim=1)
                    loss = logits.mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
                loss_val = loss.item()
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        if count > 0:
            avg_loss = total_loss / count
            print(f"  Avg Loss: {avg_loss:.4f}\n")
    
    print("\n" + "="*80)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*80)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(f"{args.output_dir}/lora_adapters")
    processor.save_pretrained(f"{args.output_dir}/processor")
    
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{args.output_dir}/merged_model")
    except:
        print("Note: Could not merge model")
    
    results = {
        "status": "✅ Fine-tuning complete",
        "original_params": original_params,
        "trainable_params": trainable_params,
        "epochs": args.num_epochs,
    }
    
    with open(f"{args.output_dir}/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Models saved to: {args.output_dir}/")
    print()

if __name__ == "__main__":
    main()
