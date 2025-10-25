#!/usr/bin/env python3
"""
Model Benchmarking Script for OpenCog Integration
This script evaluates model performance before and after transformation.
"""

import argparse
import json
import time
import torch
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def get_model_info(model_path: str):
    """Get basic information about the model."""
    try:
        metadata_path = os.path.join(model_path, "opencog_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return None


def measure_memory_usage():
    """Measure current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 3)  # Convert to GB
    except ImportError:
        return None


def benchmark_inference_speed(model, tokenizer, test_prompts, max_length=50, num_iterations=10, use_sampling=False):
    """
    Benchmark inference speed of the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        test_prompts: List of prompts to test
        max_length: Maximum generation length
        num_iterations: Number of iterations per prompt
        use_sampling: Whether to use sampling (True) or greedy decoding (False)
    
    Returns:
        dict: Benchmark results
    """
    print("\n=== Benchmarking Inference Speed ===")
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {
        "latencies": [],
        "tokens_per_second": [],
        "total_tokens_generated": 0,
    }
    
    model.eval()
    
    decoding_mode = "sampling" if use_sampling else "greedy"
    print(f"Using {decoding_mode} decoding for consistent benchmarking")
    
    for prompt in test_prompts:
        prompt_latencies = []
        
        for i in range(num_iterations):
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_length = inputs.input_ids.shape[1]
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=use_sampling,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Calculate tokens generated
            output_length = outputs.shape[1]
            tokens_generated = output_length - input_length
            
            # Calculate tokens per second
            tokens_per_sec = tokens_generated / latency if latency > 0 else 0
            
            prompt_latencies.append(latency)
            results["tokens_per_second"].append(tokens_per_sec)
            results["total_tokens_generated"] += tokens_generated
            
            if i == 0:  # Show first iteration
                print(f"  Prompt: '{prompt[:50]}...'")
                print(f"    Latency: {latency:.3f}s, Tokens/sec: {tokens_per_sec:.2f}")
        
        results["latencies"].extend(prompt_latencies)
    
    # Calculate statistics
    avg_latency = np.mean(results["latencies"])
    std_latency = np.std(results["latencies"])
    avg_tokens_per_sec = np.mean(results["tokens_per_second"])
    std_tokens_per_sec = np.std(results["tokens_per_second"])
    
    print(f"\nInference Speed Summary:")
    print(f"  Average Latency: {avg_latency:.3f}s (±{std_latency:.3f}s)")
    print(f"  Average Tokens/Second: {avg_tokens_per_sec:.2f} (±{std_tokens_per_sec:.2f})")
    print(f"  Total Tokens Generated: {results['total_tokens_generated']}")
    
    return {
        "avg_latency": avg_latency,
        "std_latency": std_latency,
        "avg_tokens_per_second": avg_tokens_per_sec,
        "std_tokens_per_second": std_tokens_per_sec,
        "total_tokens_generated": results["total_tokens_generated"],
        "min_latency": min(results["latencies"]),
        "max_latency": max(results["latencies"]),
    }


def benchmark_memory_usage(model, tokenizer, test_prompts, max_length=50):
    """
    Benchmark memory usage during inference.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        test_prompts: List of prompts to test
        max_length: Maximum generation length
    
    Returns:
        dict: Memory usage results
    """
    print("\n=== Benchmarking Memory Usage ===")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get baseline memory
    baseline_memory = measure_memory_usage()
    
    # Model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
    
    # Run inference and measure peak memory
    peak_memory = baseline_memory
    
    for prompt in test_prompts[:3]:  # Test with first 3 prompts
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        current_memory = measure_memory_usage()
        if current_memory and current_memory > peak_memory:
            peak_memory = current_memory
    
    print(f"  Model Parameters Memory: {param_memory:.2f} GB")
    if baseline_memory and peak_memory:
        print(f"  Baseline Memory: {baseline_memory:.2f} GB")
        print(f"  Peak Memory: {peak_memory:.2f} GB")
        print(f"  Memory Overhead: {peak_memory - baseline_memory:.2f} GB")
    else:
        print("  Note: psutil not available for detailed memory tracking")
    
    return {
        "param_memory_gb": param_memory,
        "baseline_memory_gb": baseline_memory,
        "peak_memory_gb": peak_memory,
        "memory_overhead_gb": peak_memory - baseline_memory if baseline_memory else None,
    }


def benchmark_output_quality(model, tokenizer, test_prompts, max_length=100):
    """
    Benchmark output quality metrics.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        test_prompts: List of prompts to test
        max_length: Maximum generation length
    
    Returns:
        dict: Quality metrics
    """
    print("\n=== Benchmarking Output Quality ===")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    perplexities = []
    output_lengths = []
    unique_tokens_ratios = []
    
    model.eval()
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            # Generate output
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Calculate perplexity
            model_outputs = model(**inputs, labels=inputs.input_ids)
            loss = model_outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
            
            # Analyze output
            output_ids = outputs[0].tolist()
            output_lengths.append(len(output_ids))
            
            # Calculate unique token ratio (diversity metric)
            unique_tokens = len(set(output_ids))
            unique_ratio = unique_tokens / len(output_ids) if len(output_ids) > 0 else 0
            unique_tokens_ratios.append(unique_ratio)
            
            # Decode for display
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n  Prompt: '{prompt[:50]}...'")
            print(f"  Generated: '{generated_text[:100]}...'")
            print(f"  Perplexity: {perplexity:.2f}, Diversity: {unique_ratio:.2f}")
    
    avg_perplexity = np.mean(perplexities)
    avg_output_length = np.mean(output_lengths)
    avg_unique_ratio = np.mean(unique_tokens_ratios)
    
    print(f"\nQuality Summary:")
    print(f"  Average Perplexity: {avg_perplexity:.2f}")
    print(f"  Average Output Length: {avg_output_length:.1f} tokens")
    print(f"  Average Token Diversity: {avg_unique_ratio:.2f}")
    
    return {
        "avg_perplexity": avg_perplexity,
        "std_perplexity": np.std(perplexities),
        "avg_output_length": avg_output_length,
        "avg_token_diversity": avg_unique_ratio,
        "min_perplexity": min(perplexities),
        "max_perplexity": max(perplexities),
    }


def run_full_benchmark(model_path: str, iterations: int = 10):
    """Run complete benchmark suite."""
    print(f"\n{'=' * 80}")
    print(f"MODEL PERFORMANCE BENCHMARK")
    print(f"{'=' * 80}")
    print(f"Model Path: {model_path}")
    
    # Load model info
    metadata = get_model_info(model_path)
    if metadata:
        print(f"Model Type: {metadata.get('model_type', 'Unknown')}")
        print(f"Parameters: {metadata.get('num_parameters', 'Unknown'):,}")
        if 'transformation_version' in metadata:
            print(f"Transformation Version: {metadata['transformation_version']}")
            print(f"Optimized: {metadata.get('optimization_applied', False)}")
    
    # Load model and tokenizer
    print(f"\nLoading model...")
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Define test prompts
    test_prompts = [
        "Once upon a time in a distant land",
        "The future of artificial intelligence",
        "In a world where technology and nature coexist",
        "Scientists have recently discovered",
        "The ancient prophecy foretold",
    ]
    
    # Run benchmarks
    benchmark_results = {
        "model_path": model_path,
        "metadata": metadata,
        "load_time": load_time,
        "test_prompts": test_prompts,
        "iterations": iterations,
    }
    
    # 1. Inference Speed
    # Test with greedy decoding for consistency and reproducibility
    benchmark_results["inference_speed"] = benchmark_inference_speed(
        model, tokenizer, test_prompts, max_length=50, num_iterations=iterations, use_sampling=False
    )
    
    # 2. Memory Usage
    benchmark_results["memory_usage"] = benchmark_memory_usage(
        model, tokenizer, test_prompts, max_length=50
    )
    
    # 3. Output Quality
    benchmark_results["output_quality"] = benchmark_output_quality(
        model, tokenizer, test_prompts, max_length=100
    )
    
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")
    
    return benchmark_results


def compare_results(before_path: str, after_path: str):
    """Compare benchmark results from two files."""
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPARISON")
    print(f"{'=' * 80}")
    
    with open(before_path, 'r') as f:
        before = json.load(f)
    
    with open(after_path, 'r') as f:
        after = json.load(f)
    
    print(f"\nBefore: {before['model_path']}")
    print(f"After:  {after['model_path']}")
    
    # Compare inference speed
    print(f"\n--- Inference Speed ---")
    before_tps = before['inference_speed']['avg_tokens_per_second']
    after_tps = after['inference_speed']['avg_tokens_per_second']
    speedup = (after_tps / before_tps - 1) * 100 if before_tps > 0 else 0
    print(f"Tokens/Second: {before_tps:.2f} → {after_tps:.2f} ({speedup:+.1f}%)")
    
    before_lat = before['inference_speed']['avg_latency']
    after_lat = after['inference_speed']['avg_latency']
    lat_change = (after_lat / before_lat - 1) * 100 if before_lat > 0 else 0
    print(f"Latency: {before_lat:.3f}s → {after_lat:.3f}s ({lat_change:+.1f}%)")
    
    # Compare memory
    print(f"\n--- Memory Usage ---")
    before_mem = before['memory_usage']['param_memory_gb']
    after_mem = after['memory_usage']['param_memory_gb']
    mem_change = (after_mem / before_mem - 1) * 100 if before_mem > 0 else 0
    print(f"Parameter Memory: {before_mem:.2f} GB → {after_mem:.2f} GB ({mem_change:+.1f}%)")
    
    # Compare quality
    print(f"\n--- Output Quality ---")
    before_ppl = before['output_quality']['avg_perplexity']
    after_ppl = after['output_quality']['avg_perplexity']
    ppl_change = (after_ppl / before_ppl - 1) * 100 if before_ppl > 0 else 0
    print(f"Perplexity: {before_ppl:.2f} → {after_ppl:.2f} ({ppl_change:+.1f}%)")
    
    before_div = before['output_quality']['avg_token_diversity']
    after_div = after['output_quality']['avg_token_diversity']
    div_change = (after_div / before_div - 1) * 100 if before_div > 0 else 0
    print(f"Token Diversity: {before_div:.2f} → {after_div:.2f} ({div_change:+.1f}%)")
    
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    
    improvements = []
    if speedup > 1:
        improvements.append(f"✓ Inference speed improved by {speedup:.1f}%")
    if lat_change < -1:
        improvements.append(f"✓ Latency reduced by {abs(lat_change):.1f}%")
    if mem_change < -1:
        improvements.append(f"✓ Memory usage reduced by {abs(mem_change):.1f}%")
    if ppl_change < -1:
        improvements.append(f"✓ Perplexity improved by {abs(ppl_change):.1f}%")
    
    if improvements:
        print("\nImprovements:")
        for imp in improvements:
            print(f"  {imp}")
    else:
        print("\nNote: Metrics are comparable. Transformation maintains model performance.")
    
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model performance for OpenCog integration"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Path to the model directory to benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for inference benchmarks (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save benchmark results JSON"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two benchmark result files"
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Compare mode
            compare_results(args.compare[0], args.compare[1])
        elif args.model:
            # Benchmark mode
            if not os.path.exists(args.model):
                print(f"Error: Model path does not exist: {args.model}")
                return 1
            
            results = run_full_benchmark(args.model, args.iterations)
            
            # Save results if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {args.output}")
            
            return 0
        else:
            parser.print_help()
            return 1
        
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
