#!/bin/bash

echo "=========================================="
echo "DOWNLOADING BLIP-2 6.7B MODEL"
echo "=========================================="
echo ""
echo "⏳ Downloading Salesforce/blip2-opt-6.7b (~13GB)..."
echo ""

cd ~/thesis-multimodal-retrieval

huggingface-cli download Salesforce/blip2-opt-6.7b --local-dir models/blip2_6.7b

echo ""
echo "✅ Download complete!"
echo ""
echo "Running REAL pruning..."
