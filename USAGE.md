# Comprehensive Usage Guide

This guide provides detailed step-by-step instructions for using the KoboldAI-OPT-350M-Erebus-Cog model with OpenCog integration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Step-by-Step Usage](#step-by-step-usage)
4. [Advanced Configuration](#advanced-configuration)
5. [Performance Evaluation](#performance-evaluation)
6. [Troubleshooting](#troubleshooting)
7. [OpenCog Integration](#opencog-integration)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: Minimum 10GB free space
- **GPU** (Optional): CUDA-compatible GPU for faster inference

### Required Software

1. **Git with LFS Support**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install git git-lfs
   
   # macOS
   brew install git git-lfs
   
   # Initialize Git LFS
   git lfs install
   ```

2. **Python and pip**
   ```bash
   # Check Python version (should be 3.8+)
   python3 --version
   
   # Check pip version
   pip --version
   ```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/cogpy/KoboldAI-OPT-350M-Erebus.git
cd KoboldAI-OPT-350M-Erebus
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch deep learning framework
- `huggingface_hub[cli]` - Hugging Face Hub CLI tools

### Step 3: Verify Installation

```bash
python3 -c "import torch; import transformers; print('Installation successful!')"
```

## Step-by-Step Usage

### Step 1: Download the Base Model

You have three options for downloading the model. Choose the one that best fits your needs:

#### Option A: Full Clone (Recommended for Most Users)

This downloads the complete model including all weights:

```bash
./clone_model.sh
# Select option 1 when prompted
```

**Pros:**
- Everything in one step
- Works offline after download
- Standard Git workflow

**Cons:**
- Requires ~2.5GB disk space
- Slower initial download

#### Option B: Clone Without Large Files

This clones the repository structure without downloading model weights initially:

```bash
./clone_model.sh
# Select option 2 when prompted
```

To download the weights later:
```bash
cd model
git lfs pull
cd ..
```

**Pros:**
- Fast initial clone
- Useful for CI/CD pipelines
- Can selectively download files

**Cons:**
- Requires additional step to get weights
- Need to remember to pull LFS files

#### Option C: Hugging Face CLI

Uses the Hugging Face CLI tool:

```bash
# First install the CLI if not already installed
pip install -U "huggingface_hub[cli]"

# Then download the model
./clone_model.sh
# Select option 3 when prompted
```

**Pros:**
- Integrates with Hugging Face ecosystem
- Can resume interrupted downloads
- Progress tracking

**Cons:**
- Requires additional dependency
- Different file structure

### Step 2: Transform Tensors for OpenCog

Transform the downloaded model for optimal OpenCog integration:

```bash
python3 transform_tensors.py --input ./model --output ./OPT-350M-Erebus-Cog --verify
```

#### Command Options

- `--input PATH` - Path to the downloaded model directory (default: `./model`)
- `--output PATH` - Path to save the transformed model (default: `./OPT-350M-Erebus-Cog`)
- `--no-optimize` - Skip optimization transformations
- `--verify` - Verify the model after transformation (recommended)

#### What Happens During Transformation?

1. **Model Loading**: The original model is loaded into memory
2. **Tensor Optimization**: Tensors are optimized for OpenCog integration
3. **Model Conversion**: The model is set to evaluation mode for inference
4. **Metadata Creation**: OpenCog-specific metadata is generated
5. **Saving**: The transformed model is saved to the output directory

#### Expected Output

```
Loading model from ./model...
Model loaded successfully. Config: opt
Model parameters: 331,000,000

Applying optimizations for OpenCog integration...
Optimizations applied.

Saving transformed model to ./OPT-350M-Erebus-Cog...

=== Transformation Complete ===
Transformed model saved to: ./OPT-350M-Erebus-Cog
Model can now be integrated with OpenCog

Verifying model at ./OPT-350M-Erebus-Cog...
✓ Model verification successful
  - Model can be loaded
  - Model can process inputs
  - Output shape: torch.Size([1, 3, 50272])

✓ All operations completed successfully
```

### Step 3: Use the Transformed Model

#### Quick Test with Example Script

```bash
python3 example_usage.py --prompt "In a world where AI and humans coexist,"
```

Options:
- `--model PATH` - Path to the transformed model (default: `./OPT-350M-Erebus-Cog`)
- `--prompt TEXT` - Text prompt for generation
- `--max-length NUM` - Maximum length of generated text (default: 100)
- `--temperature FLOAT` - Sampling temperature (default: 0.8)

#### Using in Your Own Python Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the transformed model
model_path = "./OPT-350M-Erebus-Cog"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare input
prompt = "Once upon a time in a distant galaxy"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.8,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Advanced Configuration

### Optimization Options

#### Memory Optimization

For systems with limited RAM, you can use half-precision (float16):

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
```

#### GPU Acceleration

If you have a CUDA-compatible GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
```

### Generation Parameters

Fine-tune text generation with these parameters:

```python
outputs = model.generate(
    **inputs,
    max_length=150,           # Maximum length of output
    min_length=50,            # Minimum length of output
    temperature=0.9,          # Randomness (0.1-2.0)
    top_p=0.95,              # Nucleus sampling threshold
    top_k=50,                # Top-k sampling
    repetition_penalty=1.2,   # Penalize repetitions
    do_sample=True,          # Enable sampling
    num_return_sequences=3,   # Generate multiple outputs
)
```

### Batch Processing

Process multiple prompts efficiently:

```python
prompts = [
    "The future of AI is",
    "In a world where",
    "Scientists have discovered"
]

# Tokenize all prompts
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# Generate for all prompts at once
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)

# Decode all outputs
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\nPrompt {i+1}: {prompts[i]}")
    print(f"Generated: {text}")
```

## Performance Evaluation

### Benchmark the Model

Run the benchmark script to evaluate model performance:

```bash
python3 benchmark_model.py --model ./OPT-350M-Erebus-Cog --iterations 10
```

This measures:
- **Inference Speed**: Tokens per second
- **Latency**: Time per generation
- **Memory Usage**: RAM consumption
- **Output Quality**: Perplexity and coherence metrics

### Compare Before and After Transformation

```bash
# Benchmark original model
python3 benchmark_model.py --model ./model --output results_before.json

# Benchmark transformed model
python3 benchmark_model.py --model ./OPT-350M-Erebus-Cog --output results_after.json

# Compare results
python3 benchmark_model.py --compare results_before.json results_after.json
```

## Troubleshooting

### Common Issues

#### Issue 1: Git LFS Files Not Downloaded

**Symptom**: Model files are small (a few KB instead of GB)

**Solution**:
```bash
cd model
git lfs pull
cd ..
```

#### Issue 2: Out of Memory During Loading

**Symptom**: `RuntimeError: CUDA out of memory` or `MemoryError`

**Solutions**:
1. Use CPU instead of GPU:
   ```python
   model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
   ```

2. Use float16 precision:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       torch_dtype=torch.float16
   )
   ```

3. Close other applications to free up RAM

#### Issue 3: Slow Generation

**Symptom**: Text generation takes too long

**Solutions**:
1. Use GPU if available
2. Reduce `max_length` parameter
3. Disable sampling: `do_sample=False`
4. Use greedy decoding instead of sampling

#### Issue 4: Poor Quality Output

**Symptom**: Generated text is nonsensical or repetitive

**Solutions**:
1. Adjust temperature (try 0.7-1.0)
2. Increase `top_p` (try 0.95)
3. Add `repetition_penalty=1.2`
4. Try different prompts
5. Ensure model was transformed correctly

#### Issue 5: Import Errors

**Symptom**: `ModuleNotFoundError` or `ImportError`

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

## OpenCog Integration

### Metadata for OpenCog

The transformation process creates `opencog_metadata.json` with information for OpenCog integration:

```json
{
  "source_model": "KoboldAI/OPT-350M-Erebus",
  "transformed_for": "OpenCog",
  "model_type": "opt",
  "num_parameters": 331000000,
  "optimization_applied": true
}
```

### Using with OpenCog Framework

To integrate with OpenCog:

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and metadata
model_path = "./OPT-350M-Erebus-Cog"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

with open(f"{model_path}/opencog_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Model optimized for: {metadata['transformed_for']}")
print(f"Parameters: {metadata['num_parameters']:,}")

# Use model with OpenCog
# ... your OpenCog integration code here ...
```

### Advanced OpenCog Features

The transformed model includes optimizations for:
- **Inference Mode**: Model is pre-set to evaluation mode
- **Efficient Storage**: Optimized tensor formats
- **Metadata Support**: OpenCog-specific configuration

## Best Practices

1. **Always verify transformations**: Use the `--verify` flag
2. **Benchmark your setup**: Run benchmarks to understand performance
3. **Save disk space**: Delete the original model after successful transformation
4. **Use version control**: Track your configuration and prompts
5. **Monitor resources**: Watch RAM and GPU usage during inference
6. **Test prompts**: Experiment with different prompts for best results
7. **Keep dependencies updated**: Regularly update transformers and torch

## Next Steps

- Read the [README.md](README.md) for project overview
- See [TESTING.md](TESTING.md) for testing instructions
- Check [QUICKSTART.md](QUICKSTART.md) for quick reference
- Explore the example scripts in the repository
- Join the community for support and discussions

## Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs)
- [OpenCog Documentation](https://wiki.opencog.org)
- [KoboldAI/OPT-350M-Erebus Model Card](https://huggingface.co/KoboldAI/OPT-350M-Erebus)

## Support

If you encounter issues not covered in this guide:
1. Check existing GitHub issues
2. Review the troubleshooting section
3. Open a new issue with detailed information
4. Include error messages and system specifications
