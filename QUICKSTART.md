# Quick Start Guide

This is a quick reference for getting started with OPT-350M-Erebus-Cog.

## 1. Setup (One-time)

```bash
# Install dependencies
pip install -r requirements.txt

# Verify git-lfs is installed
git lfs install
```

## 2. Download the Model

Choose one method:

### Option A: Interactive Script (Recommended)
```bash
./clone_model.sh
# Follow the prompts
```

### Option B: Direct Clone
```bash
git clone https://huggingface.co/KoboldAI/OPT-350M-Erebus ./model
```

### Option C: Hugging Face CLI
```bash
hf download KoboldAI/OPT-350M-Erebus --local-dir ./model
```

## 3. Transform for OpenCog

```bash
python3 transform_tensors.py --input ./model --output ./OPT-350M-Erebus-Cog --verify
```

## 4. Use the Model

```bash
python3 example_usage.py --prompt "Your text here"
```

Or in your own Python code:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./OPT-350M-Erebus-Cog")
tokenizer = AutoTokenizer.from_pretrained("./OPT-350M-Erebus-Cog")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## File Overview

| File | Purpose |
|------|---------|
| `clone_model.sh` | Downloads the model from Hugging Face |
| `transform_tensors.py` | Transforms tensors for OpenCog integration |
| `example_usage.py` | Example of using the transformed model |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `TESTING.md` | Testing instructions |

## Common Commands

```bash
# Show help for clone script
./clone_model.sh

# Show help for transform script
python3 transform_tensors.py --help

# Show help for example script
python3 example_usage.py --help

# Check model metadata
cat ./OPT-350M-Erebus-Cog/opencog_metadata.json
```

## Disk Space Requirements

- Model download: ~2.5 GB
- Transformed model: ~2.5 GB
- Total: ~5 GB (both models)

To save space, you can delete the original model after transformation:
```bash
rm -rf ./model
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- See [TESTING.md](TESTING.md) for testing instructions
- Customize `transform_tensors.py` for your specific OpenCog integration needs
