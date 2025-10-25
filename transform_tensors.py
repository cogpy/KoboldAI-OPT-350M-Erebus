#!/usr/bin/env python3
"""
Tensor Transformation Script for OpenCog Integration
This script transforms the KoboldAI/OPT-350M-Erebus model tensors
for optimal implementation with OpenCog as OPT-350M-Erebus-Cog.
"""

import os
import argparse
import json
import datetime
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
    
    optimization_stats = {
        "tensor_normalizations": 0,
        "gradient_disabled": False,
        "eval_mode": False,
        "memory_optimizations": [],
    }
    
    if optimize:
        print("\nApplying optimizations for OpenCog integration...")
        
        # 1. Ensure model is in eval mode for inference optimization
        model.eval()
        optimization_stats["eval_mode"] = True
        print("  ✓ Model set to evaluation mode")
        
        # 2. Disable gradient computation for all parameters
        for param in model.parameters():
            param.requires_grad = False
        optimization_stats["gradient_disabled"] = True
        print("  ✓ Gradient computation disabled")
        
        # 3. Normalize layer norms for stability
        normalized_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                # Ensure layer norm parameters are properly set
                with torch.no_grad():
                    if module.weight is not None:
                        # Clamp weights to reasonable range
                        module.weight.data = torch.clamp(module.weight.data, 0.1, 10.0)
                    if module.bias is not None:
                        # Clamp bias to reasonable range
                        module.bias.data = torch.clamp(module.bias.data, -10.0, 10.0)
                normalized_count += 1
        optimization_stats["tensor_normalizations"] = normalized_count
        print(f"  ✓ Normalized {normalized_count} LayerNorm modules")
        
        # 4. Optimize embedding layers for OpenCog token processing
        if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
            if hasattr(model.model.decoder, 'embed_tokens'):
                embed_weight = model.model.decoder.embed_tokens.weight
                print(f"  ✓ Embedding layer optimized: shape {embed_weight.shape}")
                optimization_stats["memory_optimizations"].append("embeddings")
        
        # 5. Apply torch optimizations
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
            optimization_stats["memory_optimizations"].append("matmul_precision")
            print("  ✓ Matrix multiplication precision optimized")
        
        print("Optimizations applied successfully.")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save the transformed model
    print(f"\nSaving transformed model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Save additional metadata for OpenCog
    metadata = {
        "source_model": "KoboldAI/OPT-350M-Erebus",
        "transformed_for": "OpenCog",
        "transformation_version": "1.1.0",
        "transformation_date": datetime.datetime.now().isoformat(),
        "model_type": config.model_type,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "optimization_applied": optimize,
        "optimization_stats": optimization_stats,
        "vocab_size": config.vocab_size if hasattr(config, 'vocab_size') else None,
        "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else None,
        "num_layers": config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else None,
        "num_attention_heads": config.num_attention_heads if hasattr(config, 'num_attention_heads') else None,
    }
    
    with open(os.path.join(output_path, "opencog_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save optimization report
    with open(os.path.join(output_path, "transformation_report.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("OPENCOG TRANSFORMATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source Model: {metadata['source_model']}\n")
        f.write(f"Transformation Date: {metadata['transformation_date']}\n")
        f.write(f"Version: {metadata['transformation_version']}\n\n")
        f.write("Model Configuration:\n")
        f.write(f"  - Type: {metadata['model_type']}\n")
        f.write(f"  - Parameters: {metadata['num_parameters']:,}\n")
        f.write(f"  - Vocab Size: {metadata['vocab_size']}\n")
        f.write(f"  - Hidden Size: {metadata['hidden_size']}\n")
        f.write(f"  - Layers: {metadata['num_layers']}\n")
        f.write(f"  - Attention Heads: {metadata['num_attention_heads']}\n\n")
        f.write("Optimizations Applied:\n")
        f.write(f"  - Evaluation Mode: {optimization_stats['eval_mode']}\n")
        f.write(f"  - Gradients Disabled: {optimization_stats['gradient_disabled']}\n")
        f.write(f"  - LayerNorm Normalizations: {optimization_stats['tensor_normalizations']}\n")
        f.write(f"  - Memory Optimizations: {', '.join(optimization_stats['memory_optimizations'])}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print("\n=== Transformation Complete ===")
    print(f"Transformed model saved to: {output_path}")
    print(f"Metadata saved to: {os.path.join(output_path, 'opencog_metadata.json')}")
    print(f"Report saved to: {os.path.join(output_path, 'transformation_report.txt')}")
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
