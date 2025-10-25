#!/usr/bin/env python3
"""
Example script demonstrating how to use the transformed OPT-350M-Erebus-Cog model.
This script shows basic text generation using the transformed model with performance metrics.
"""

import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_text(model_path: str, prompt: str, max_length: int = 100, temperature: float = 0.8):
    """
    Generate text using the transformed model.
    
    Args:
        model_path: Path to the transformed model directory
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        str: The generated text including the original prompt
    """
    print(f"Loading model from {model_path}...")
    load_start = time.time()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    # Get model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded. Generating text for prompt: '{prompt}'\n")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_length = inputs.input_ids.shape[1]
    
    # Generate
    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - gen_start
    
    # Calculate performance metrics
    output_length = outputs.shape[1]
    tokens_generated = output_length - input_length
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
    
    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    
    # Print performance metrics
    print("\nPERFORMANCE METRICS:")
    print("=" * 80)
    print(f"Generation Time: {gen_time:.3f}s")
    print(f"Tokens Generated: {tokens_generated}")
    print(f"Tokens per Second: {tokens_per_sec:.2f}")
    print(f"Input Length: {input_length} tokens")
    print(f"Output Length: {output_length} tokens")
    print("=" * 80)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using the transformed OPT-350M-Erebus-Cog model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./OPT-350M-Erebus-Cog",
        help="Path to the transformed model directory (default: ./OPT-350M-Erebus-Cog)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time in a land far away",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum length of generated text (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    try:
        generate_text(
            args.model,
            args.prompt,
            args.max_length,
            args.temperature
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
