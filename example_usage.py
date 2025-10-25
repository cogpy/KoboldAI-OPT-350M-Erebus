#!/usr/bin/env python3
"""
Example script demonstrating how to use the transformed OPT-350M-Erebus-Cog model.
This script shows basic text generation using the transformed model.
"""

import argparse
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
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded. Generating text for prompt: '{prompt}'\n")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate
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
    
    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(generated_text)
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
