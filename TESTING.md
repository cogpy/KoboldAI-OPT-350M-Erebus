# Testing Guide

This document provides instructions for testing the model cloning and transformation scripts.

## Testing the Clone Script

### Basic Syntax Check
The clone script syntax has been verified using bash's built-in syntax checker:
```bash
bash -n clone_model.sh
```

### Interactive Test
To test the clone script interactively:
```bash
./clone_model.sh
```

Then select one of the three options:
1. Clone with full model files
2. Clone without large files (pointers only)
3. Download using Hugging Face CLI

**Note:** Option 3 requires the Hugging Face CLI to be installed:
```bash
pip install -U "huggingface_hub[cli]"
```

### Automated Test (Non-Interactive)
For automated testing, you can use option 2 (clone without large files) by providing input:
```bash
echo "2" | ./clone_model.sh
```

This will clone the repository structure without downloading the large model files.

## Testing the Transform Script

### Syntax Check
The Python script syntax has been verified:
```bash
python3 -m py_compile transform_tensors.py
```

### Help Output Test
To verify the script accepts command-line arguments correctly:
```bash
python3 transform_tensors.py --help
```

**Note:** This requires the Python dependencies to be installed:
```bash
pip install -r requirements.txt
```

### Full Transformation Test
To test the full transformation pipeline:

1. First, clone the model (using the lightweight option):
   ```bash
   echo "2" | ./clone_model.sh
   ```

2. Then pull only the necessary files to test:
   ```bash
   cd model
   git lfs pull --include="config.json,tokenizer.json,vocab.json"
   cd ..
   ```

3. Run the transformation:
   ```bash
   python3 transform_tensors.py --input ./model --output ./OPT-350M-Erebus-Cog
   ```

**Note:** For a full test with model weights, you'll need significant disk space (>2GB) and RAM (>4GB).

## Continuous Integration Testing

For CI/CD pipelines, consider testing with a smaller model or mock data:

```bash
# Example: Test with a tiny model for CI
pip install transformers torch
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Download a tiny model for testing
model = AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2')
tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')

# Save to test directory
os.makedirs('./test_model', exist_ok=True)
model.save_pretrained('./test_model')
tokenizer.save_pretrained('./test_model')
print('Test model created')
"

# Test transformation with the tiny model
python3 transform_tensors.py --input ./test_model --output ./test_output --verify

# Cleanup
rm -rf ./test_model ./test_output
```

## Expected Outputs

### Successful Clone
```
=== KoboldAI OPT-350M-Erebus Model Download Script ===

Initializing git-lfs...
...
Model cloned successfully to ./model
```

### Successful Transformation
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
```

## Troubleshooting

### Git LFS Not Found
```bash
# Install git-lfs first
sudo apt-get install git-lfs  # Ubuntu/Debian
brew install git-lfs          # macOS

# Initialize
git lfs install
```

### Python Dependencies Missing
```bash
pip install -r requirements.txt
```

### Out of Memory
If you encounter memory issues:
- Use a machine with at least 8GB RAM
- Use the `--no-optimize` flag
- Test with a smaller model first

## Validation Checklist

- [ ] clone_model.sh is executable
- [ ] transform_tensors.py is executable
- [ ] Both scripts pass syntax checks
- [ ] Dependencies are listed in requirements.txt
- [ ] README.md provides clear instructions
- [ ] .gitignore excludes model directories
- [ ] Scripts handle errors gracefully
- [ ] Help output is informative
