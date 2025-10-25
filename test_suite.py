#!/usr/bin/env python3
"""
Test script to validate code structure and functionality.
This validates syntax, imports, and basic functionality without requiring model downloads.
"""

import os
import sys
import json
from pathlib import Path


def test_imports():
    """Test that all scripts can be imported."""
    print("=" * 80)
    print("TESTING IMPORTS")
    print("=" * 80)
    
    try:
        print("\nTesting transform_tensors.py imports...")
        import transform_tensors
        print("✓ transform_tensors.py imports successfully")
        
        print("\nTesting example_usage.py imports...")
        import example_usage
        print("✓ example_usage.py imports successfully")
        
        print("\nTesting benchmark_model.py imports...")
        import benchmark_model
        print("✓ benchmark_model.py imports successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_documentation():
    """Test that documentation files exist and are valid."""
    print("\n" + "=" * 80)
    print("TESTING DOCUMENTATION")
    print("=" * 80)
    
    required_docs = [
        "README.md",
        "USAGE.md",
        "QUICKSTART.md",
        "TESTING.md",
    ]
    
    all_exist = True
    for doc in required_docs:
        if os.path.exists(doc):
            print(f"✓ {doc} exists")
            # Check if file has content
            with open(doc, 'r') as f:
                content = f.read()
                if len(content) > 100:
                    print(f"  - Size: {len(content)} bytes")
                else:
                    print(f"  ✗ Warning: {doc} seems too small")
                    all_exist = False
        else:
            print(f"✗ {doc} missing")
            all_exist = False
    
    return all_exist


def test_script_structure():
    """Test the structure and functions of each script."""
    print("\n" + "=" * 80)
    print("TESTING SCRIPT STRUCTURE")
    print("=" * 80)
    
    try:
        # Test transform_tensors.py
        print("\nChecking transform_tensors.py...")
        import transform_tensors
        
        required_functions = ['transform_tensors_for_opencog', 'verify_model', 'main']
        for func_name in required_functions:
            if hasattr(transform_tensors, func_name):
                print(f"  ✓ Function '{func_name}' exists")
            else:
                print(f"  ✗ Function '{func_name}' missing")
                return False
        
        # Test example_usage.py
        print("\nChecking example_usage.py...")
        import example_usage
        
        if hasattr(example_usage, 'generate_text'):
            print(f"  ✓ Function 'generate_text' exists")
        else:
            print(f"  ✗ Function 'generate_text' missing")
            return False
        
        # Test benchmark_model.py
        print("\nChecking benchmark_model.py...")
        import benchmark_model
        
        required_benchmark_functions = [
            'benchmark_inference_speed',
            'benchmark_memory_usage',
            'benchmark_output_quality',
            'run_full_benchmark',
            'compare_results'
        ]
        for func_name in required_benchmark_functions:
            if hasattr(benchmark_model, func_name):
                print(f"  ✓ Function '{func_name}' exists")
            else:
                print(f"  ✗ Function '{func_name}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_help_output():
    """Test that scripts display help correctly."""
    print("\n" + "=" * 80)
    print("TESTING HELP OUTPUT")
    print("=" * 80)
    
    import subprocess
    
    scripts = [
        ("transform_tensors.py", ["--help"]),
        ("example_usage.py", ["--help"]),
        ("benchmark_model.py", ["--help"]),
    ]
    
    all_passed = True
    for script, args in scripts:
        print(f"\nTesting {script} help...")
        try:
            result = subprocess.run(
                [sys.executable, script] + args,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and "usage:" in result.stdout:
                print(f"  ✓ Help output working")
            else:
                print(f"  ✗ Help output failed")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            all_passed = False
    
    return all_passed


def test_code_quality():
    """Test code quality and standards."""
    print("\n" + "=" * 80)
    print("TESTING CODE QUALITY")
    print("=" * 80)
    
    import subprocess
    
    python_files = [
        "transform_tensors.py",
        "example_usage.py",
        "benchmark_model.py",
    ]
    
    all_passed = True
    for filename in python_files:
        print(f"\nChecking {filename}...")
        
        # Test syntax
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", filename],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ Syntax valid")
        else:
            print(f"  ✗ Syntax error: {result.stderr}")
            all_passed = False
        
        # Check for key quality indicators
        with open(filename, 'r') as f:
            content = f.read()
            
            # Check for docstrings
            if '"""' in content:
                print(f"  ✓ Has docstrings")
            else:
                print(f"  ✗ Missing docstrings")
            
            # Check for argparse usage
            if 'argparse' in content:
                print(f"  ✓ Uses argparse for CLI")
            
            # Check for proper error handling
            if 'try:' in content and 'except' in content:
                print(f"  ✓ Has error handling")
    
    return all_passed


def test_feature_completeness():
    """Test that key features are implemented."""
    print("\n" + "=" * 80)
    print("TESTING FEATURE COMPLETENESS")
    print("=" * 80)
    
    features_found = 0
    total_features = 0
    
    # Check transform_tensors.py features
    print("\nChecking transform_tensors.py features...")
    with open("transform_tensors.py", 'r') as f:
        content = f.read()
        
        features = [
            ("LayerNorm optimization", "LayerNorm"),
            ("Gradient disabling", "requires_grad"),
            ("Safe serialization", "safe_serialization"),
            ("Metadata generation", "opencog_metadata.json"),
            ("Transformation report", "transformation_report.txt"),
            ("Optimization stats", "optimization_stats"),
        ]
        
        for feature_name, keyword in features:
            total_features += 1
            if keyword in content:
                print(f"  ✓ {feature_name}")
                features_found += 1
            else:
                print(f"  ✗ {feature_name} missing")
    
    # Check example_usage.py features
    print("\nChecking example_usage.py features...")
    with open("example_usage.py", 'r') as f:
        content = f.read()
        
        features = [
            ("Performance timing", "time.time()"),
            ("Tokens per second metric", "tokens_per_sec"),
            ("Parameter count display", "param_count"),
        ]
        
        for feature_name, keyword in features:
            total_features += 1
            if keyword in content:
                print(f"  ✓ {feature_name}")
                features_found += 1
            else:
                print(f"  ✗ {feature_name} missing")
    
    # Check benchmark_model.py features
    print("\nChecking benchmark_model.py features...")
    with open("benchmark_model.py", 'r') as f:
        content = f.read()
        
        features = [
            ("Inference speed benchmark", "benchmark_inference_speed"),
            ("Memory usage benchmark", "benchmark_memory_usage"),
            ("Output quality benchmark", "benchmark_output_quality"),
            ("Comparison functionality", "compare_results"),
            ("Perplexity calculation", "perplexity"),
        ]
        
        for feature_name, keyword in features:
            total_features += 1
            if keyword in content:
                print(f"  ✓ {feature_name}")
                features_found += 1
            else:
                print(f"  ✗ {feature_name} missing")
    
    print(f"\nFeatures implemented: {features_found}/{total_features}")
    
    return features_found >= total_features * 0.9  # 90% threshold


def main():
    print("\nKoboldAI OPT-350M-Erebus-Cog Test Suite")
    print("This validates the implementation without requiring model downloads\n")
    
    # Change to repository directory
    repo_dir = Path(__file__).parent
    os.chdir(repo_dir)
    
    results = []
    
    # Run all tests
    results.append(("Import Tests", test_imports()))
    results.append(("Documentation Tests", test_documentation()))
    results.append(("Script Structure Tests", test_script_structure()))
    results.append(("Help Output Tests", test_help_output()))
    results.append(("Code Quality Tests", test_code_quality()))
    results.append(("Feature Completeness Tests", test_feature_completeness()))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    print("=" * 80)
    if passed == total:
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
