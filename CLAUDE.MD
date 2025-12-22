# CLAUDE.MD

## Project Overview

KoboldAI-OPT-350M-Erebus-Cog is a toolkit for cloning and transforming the KoboldAI/OPT-350M-Erebus language model for optimal integration with OpenCog. The OPT-350M-Erebus is a 350M parameter language model fine-tuned by KoboldAI.

## Tech Stack

- **Python**: 3.8+
- **PyTorch**: Deep learning framework for tensor operations
- **Hugging Face Transformers**: Model loading, tokenization, and inference
- **Hugging Face Hub CLI**: Model downloading from Hugging Face

## Project Structure

```
.
├── clone_model.sh          # Interactive script to download model from Hugging Face
├── transform_tensors.py    # Transform model tensors for OpenCog integration
├── example_usage.py        # Example text generation with performance metrics
├── benchmark_model.py      # Performance benchmarking (speed, memory, quality)
├── test_suite.py           # Validation tests (imports, structure, quality)
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
├── USAGE.md                # Comprehensive step-by-step usage guide
├── QUICKSTART.md           # Quick reference guide
├── TESTING.md              # Testing instructions
├── model/                  # Downloaded model directory (gitignored)
└── OPT-350M-Erebus-Cog/    # Transformed model output (gitignored)
```

## Common Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Clone the Model
```bash
./clone_model.sh
# Select option 1 for full clone, option 2 for lightweight clone
```

### Transform Model for OpenCog
```bash
python3 transform_tensors.py --input ./model --output ./OPT-350M-Erebus-Cog --verify
```

Options:
- `--input PATH`: Source model directory (default: `./model`)
- `--output PATH`: Output directory (default: `./OPT-350M-Erebus-Cog`)
- `--no-optimize`: Skip optimization transformations
- `--verify`: Verify model after transformation

### Run Example Text Generation
```bash
python3 example_usage.py --prompt "Once upon a time" --max-length 100 --temperature 0.8
```

### Benchmark Model Performance
```bash
# Benchmark a model
python3 benchmark_model.py --model ./OPT-350M-Erebus-Cog --iterations 10 --output results.json

# Compare before/after results
python3 benchmark_model.py --compare results_before.json results_after.json
```

### Run Tests
```bash
python3 test_suite.py
```

This runs: import tests, documentation checks, script structure validation, help output tests, code quality checks, and feature completeness verification.

## Key Optimizations Applied During Transformation

1. **Evaluation Mode**: Model set to `model.eval()` for inference
2. **Gradient Disabling**: All parameters set to `requires_grad=False`
3. **LayerNorm Normalization**: Weights/biases clamped to stable ranges
4. **Embedding Optimization**: Embedding layers prepared for token processing
5. **Safe Serialization**: Uses safetensors format for storage

## Generated Artifacts

After transformation, the output directory contains:
- `opencog_metadata.json`: OpenCog integration metadata (version, params, etc.)
- `transformation_report.txt`: Detailed report of applied optimizations
- Standard Hugging Face model files (config.json, model weights, tokenizer)

## Code Conventions

- Scripts use `argparse` for CLI interfaces
- All scripts have `--help` support
- Error handling with try/except blocks and traceback printing
- Docstrings on all public functions
- Scripts are executable (`#!/usr/bin/env python3`)

## Testing Notes

- Tests can run without downloading the actual model
- For CI, use a tiny model like `sshleifer/tiny-gpt2` instead
- The test suite validates code structure without requiring 2GB+ model weights

## System Requirements

- **RAM**: Minimum 8GB (16GB recommended)
- **Disk**: Minimum 10GB free space
- **GPU**: Optional, CUDA-compatible for faster inference
