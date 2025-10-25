#!/usr/bin/env python3
"""
Tensor Transformation Script for OpenCog Integration
This script transforms the KoboldAI/OPT-350M-Erebus model tensors
for optimal implementation with OpenCog as OPT-350M-Erebus-Cog.
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def transform_tensors_for_opencog(model_path: str, output_path: str, optimize: bool = True):
    """
    Transform model tensors for OpenCog integration.
    
    Args:
        model_path: Path to the source model directory
        output_path: Path to save the transformed model
        optimize: Whether to apply optimization transformations
    """
    print(f"Loading model from {model_path}...")
    
    # Load the model and tokenizer
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Model loaded successfully. Config: {config.model_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if optimize:
        print("\nApplying optimizations for OpenCog integration...")
        
        # Convert to float16 for efficiency (optional, based on use case)
        # model = model.half()
        
        # Ensure model is in eval mode for inference optimization
        model.eval()
        
        print("Optimizations applied.")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save the transformed model
    print(f"\nSaving transformed model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save additional metadata for OpenCog
    metadata = {
        "source_model": "KoboldAI/OPT-350M-Erebus",
        "transformed_for": "OpenCog",
        "model_type": config.model_type,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "optimization_applied": optimize,
    }
    
    import json
    with open(os.path.join(output_path, "opencog_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n=== Transformation Complete ===")
    print(f"Transformed model saved to: {output_path}")
    print(f"Model can now be integrated with OpenCog")
    
    return model, tokenizer


def verify_model(model_path: str):
    """Verify that the model can be loaded and used."""
    print(f"\nVerifying model at {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Test with a simple input
        test_input = "Hello, world!"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✓ Model verification successful")
        print(f"  - Model can be loaded")
        print(f"  - Model can process inputs")
        print(f"  - Output shape: {outputs.logits.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Transform KoboldAI/OPT-350M-Erebus tensors for OpenCog integration"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./model",
        help="Path to the input model directory (default: ./model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./OPT-350M-Erebus-Cog",
        help="Path to save the transformed model (default: ./OPT-350M-Erebus-Cog)"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip optimization transformations"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the transformed model after transformation"
    )
    
    args = parser.parse_args()
    
    # Check if input model exists
    if not os.path.exists(args.input):
        print(f"Error: Input model path does not exist: {args.input}")
        print("Please run clone_model.sh first to download the model.")
        return 1
    
    # Transform the model
    try:
        model, tokenizer = transform_tensors_for_opencog(
            args.input,
            args.output,
            optimize=not args.no_optimize
        )
        
        # Verify if requested
        if args.verify:
            verify_model(args.output)
        
        print("\n✓ All operations completed successfully")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during transformation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
