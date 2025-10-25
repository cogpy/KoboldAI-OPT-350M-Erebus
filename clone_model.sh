#!/bin/bash

# Script to clone the KoboldAI/OPT-350M-Erebus model from HuggingFace
# This script provides multiple methods to download the model

set -e

MODEL_NAME="KoboldAI/OPT-350M-Erebus"
OUTPUT_DIR="./model"

echo "=== KoboldAI OPT-350M-Erebus Model Download Script ==="
echo ""

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs is not installed."
    echo "Please install git-lfs from https://git-lfs.com"
    exit 1
fi

# Initialize git-lfs
echo "Initializing git-lfs..."
git lfs install

echo ""
echo "Select download method:"
echo "1) Clone with full model files (requires git-lfs)"
echo "2) Clone without large files (pointers only)"
echo "3) Download using Hugging Face CLI (hf download)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Cloning model with full files..."
        echo "This will download the complete model including all large files."
        git clone https://huggingface.co/$MODEL_NAME $OUTPUT_DIR
        echo "Model cloned successfully to $OUTPUT_DIR"
        ;;
    2)
        echo ""
        echo "Cloning model without large files (pointers only)..."
        GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/$MODEL_NAME $OUTPUT_DIR
        echo "Model repository cloned (without large files) to $OUTPUT_DIR"
        echo "To download large files later, run: cd $OUTPUT_DIR && git lfs pull"
        ;;
    3)
        echo ""
        # Check if huggingface-cli is installed
        if ! command -v huggingface-cli &> /dev/null && ! command -v hf &> /dev/null; then
            echo "Error: Hugging Face CLI is not installed."
            echo "Please install it with: pip install -U \"huggingface_hub[cli]\""
            exit 1
        fi
        
        echo "Downloading model using Hugging Face CLI..."
        if command -v hf &> /dev/null; then
            hf download $MODEL_NAME --local-dir $OUTPUT_DIR
        else
            huggingface-cli download $MODEL_NAME --local-dir $OUTPUT_DIR
        fi
        echo "Model downloaded successfully to $OUTPUT_DIR"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=== Download Complete ==="
echo "Model location: $OUTPUT_DIR"
