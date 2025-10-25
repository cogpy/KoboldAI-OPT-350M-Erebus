# KoboldAI-OPT-350M-Erebus-Cog

This repository provides tools to clone and transform the KoboldAI/OPT-350M-Erebus model for optimal implementation with OpenCog.

## Overview

The OPT-350M-Erebus is a 350M parameter language model fine-tuned by KoboldAI. This repository includes scripts to:
- Clone the model from Hugging Face
- Transform model tensors for OpenCog integration with advanced optimizations
- Optimize the model for inference
- Benchmark model performance before and after transformation

## Documentation

- **[USAGE.md](USAGE.md)** - Comprehensive step-by-step usage guide
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide
- **[TESTING.md](TESTING.md)** - Testing and validation instructions

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Git with Git LFS support
- pip (Python package manager)

### Installing Git LFS

Git LFS (Large File Storage) is required to download large model files.

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install git-lfs
```

**macOS:**
```bash
brew install git-lfs
```

**Windows:**
Download and install from [https://git-lfs.com](https://git-lfs.com)

After installation, initialize Git LFS:
```bash
git lfs install
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/cogpy/KoboldAI-OPT-350M-Erebus.git
cd KoboldAI-OPT-350M-Erebus
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Clone the Model

Use the provided script to download the KoboldAI/OPT-350M-Erebus model from Hugging Face:

```bash
./clone_model.sh
```

The script offers three download methods:

1. **Full clone with large files** - Downloads everything including model weights (recommended for most users)
2. **Clone without large files** - Downloads only metadata and pointers (useful for version control)
3. **Hugging Face CLI download** - Uses the `hf` command-line tool

#### Manual Clone Methods

If you prefer to clone manually:

**Method 1: Standard clone (with large files)**
```bash
git lfs install
git clone https://huggingface.co/KoboldAI/OPT-350M-Erebus ./model
```

**Method 2: Clone without large files (pointers only)**
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/KoboldAI/OPT-350M-Erebus ./model
# Later, download large files with:
cd ./model && git lfs pull
```

**Method 3: Using Hugging Face CLI**
```bash
# Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
hf download KoboldAI/OPT-350M-Erebus --local-dir ./model
```

### Step 2: Transform Tensors for OpenCog

After downloading the model, transform the tensors for OpenCog integration:

```bash
python3 transform_tensors.py --input ./model --output ./OPT-350M-Erebus-Cog --verify
```

The transformation applies advanced optimizations including:
- Tensor normalization for stability
- Gradient computation optimization
- LayerNorm parameter clamping
- Embedding layer optimization
- Safe serialization for efficient storage

#### Transform Options

- `--input PATH` - Path to the downloaded model directory (default: `./model`)
- `--output PATH` - Path to save the transformed model (default: `./OPT-350M-Erebus-Cog`)
- `--no-optimize` - Skip optimization transformations
- `--verify` - Verify the model after transformation

**Example with verification:**
```bash
python3 transform_tensors.py --input ./model --output ./OPT-350M-Erebus-Cog --verify
```

### Step 3: Using the Transformed Model

The transformed model can be loaded using the Hugging Face Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./OPT-350M-Erebus-Cog"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Step 4: Benchmark Performance

Evaluate model performance before and after transformation:

```bash
# Benchmark the original model
python3 benchmark_model.py --model ./model --output results_before.json --iterations 10

# Benchmark the transformed model
python3 benchmark_model.py --model ./OPT-350M-Erebus-Cog --output results_after.json --iterations 10

# Compare results
python3 benchmark_model.py --compare results_before.json results_after.json
```

The benchmark evaluates:
- **Inference Speed**: Tokens per second and latency
- **Memory Usage**: Parameter memory and runtime overhead
- **Output Quality**: Perplexity and token diversity

For detailed usage instructions, see [USAGE.md](USAGE.md).

## Project Structure

```
.
├── clone_model.sh          # Script to download the model from Hugging Face
├── transform_tensors.py    # Script to transform tensors for OpenCog
├── example_usage.py        # Example usage with performance metrics
├── benchmark_model.py      # Performance benchmarking script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── USAGE.md               # Comprehensive usage guide
├── QUICKSTART.md          # Quick start guide
├── TESTING.md             # Testing instructions
├── model/                 # Downloaded model (created after running clone_model.sh)
└── OPT-350M-Erebus-Cog/  # Transformed model (created after running transform_tensors.py)
    ├── opencog_metadata.json        # OpenCog integration metadata
    └── transformation_report.txt    # Detailed transformation report
```

## Troubleshooting

### Git LFS Issues

If you encounter errors related to Git LFS:
```bash
git lfs install
git lfs pull
```

### Hugging Face CLI Not Found

Install the Hugging Face CLI:
```bash
pip install -U "huggingface_hub[cli]"
```

### Out of Memory

If you run out of memory during transformation, try:
- Using a machine with more RAM
- Using the `--no-optimize` flag
- Processing on a GPU if available

## License

This project follows the same license as the original KoboldAI/OPT-350M-Erebus model.

## Credits

- Original model: [KoboldAI/OPT-350M-Erebus](https://huggingface.co/KoboldAI/OPT-350M-Erebus)
- OpenCog integration: cogpy

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
