# Tensor-to-File Mapping

## Overview

This document provides a complete mapping of OPT-350M-Erebus transformer tensors to GitHub repository files. Each tensor is represented as a file with specific attributes that preserve its mathematical properties.

---

## Tensor File Schema

### File Naming Convention

```
<module_path>.<parameter_type>.safetensor
```

Examples:
- `model.decoder.layers.0.self_attn.q_proj.weight.safetensor`
- `model.decoder.layers.0.self_attn.q_proj.bias.safetensor`
- `model.decoder.embed_tokens.weight.safetensor`

### File Metadata (Git Attributes)

```gitattributes
# Tensor files use Git LFS
*.safetensor filter=lfs diff=lfs merge=lfs -text

# Tensor metadata in extended attributes
#   tensor.shape: [dim1, dim2, ...]
#   tensor.dtype: float32 | float16 | bfloat16
#   tensor.requires_grad: true | false
#   tensor.device: cpu | cuda
```

---

## Complete Tensor Inventory

### 1. Embedding Tensors

| Tensor Name | Shape | Size (MB) | File Path |
|------------|-------|-----------|-----------|
| `embed_tokens.weight` | [50272, 512] | 98.0 | `model/embed_tokens/weight.safetensor` |
| `embed_positions.weight` | [2050, 512] | 4.0 | `model/embed_positions/weight.safetensor` |

**Embedding Directory Structure:**
```
model/
├── embed_tokens/
│   ├── weight.safetensor          # Token embedding matrix
│   └── metadata.json              # Vocabulary mappings
│       {
│         "shape": [50272, 512],
│         "dtype": "float32",
│         "vocab_size": 50272,
│         "embed_dim": 512,
│         "padding_idx": 1,
│         "description": "Maps token IDs to dense vectors"
│       }
└── embed_positions/
    ├── weight.safetensor          # Positional embedding matrix
    └── metadata.json
        {
          "shape": [2050, 512],
          "dtype": "float32",
          "max_position": 2048,
          "offset": 2,
          "description": "Learned positional encodings"
        }
```

---

### 2. Decoder Layer Tensors (Per Layer)

Each of the 24 decoder layers contains the following tensors:

#### Self-Attention Module

| Tensor | Shape | Size (KB) | Description |
|--------|-------|-----------|-------------|
| `self_attn.q_proj.weight` | [512, 512] | 1024 | Query projection matrix |
| `self_attn.q_proj.bias` | [512] | 2 | Query projection bias |
| `self_attn.k_proj.weight` | [512, 512] | 1024 | Key projection matrix |
| `self_attn.k_proj.bias` | [512] | 2 | Key projection bias |
| `self_attn.v_proj.weight` | [512, 512] | 1024 | Value projection matrix |
| `self_attn.v_proj.bias` | [512] | 2 | Value projection bias |
| `self_attn.out_proj.weight` | [512, 512] | 1024 | Output projection matrix |
| `self_attn.out_proj.bias` | [512] | 2 | Output projection bias |

**Attention Directory Structure:**
```
model/layers/0/self_attn/
├── q_proj/
│   ├── weight.safetensor
│   ├── bias.safetensor
│   └── metadata.json
│       {
│         "in_features": 512,
│         "out_features": 512,
│         "num_heads": 16,
│         "head_dim": 32,
│         "operation": "query_projection"
│       }
├── k_proj/
│   ├── weight.safetensor
│   ├── bias.safetensor
│   └── metadata.json
├── v_proj/
│   ├── weight.safetensor
│   ├── bias.safetensor
│   └── metadata.json
├── out_proj/
│   ├── weight.safetensor
│   ├── bias.safetensor
│   └── metadata.json
└── config.json
    {
      "num_heads": 16,
      "head_dim": 32,
      "dropout": 0.0,
      "is_causal": true,
      "scale": 0.1767766952966369  # 1/sqrt(32)
    }
```

#### Layer Normalization Modules

| Tensor | Shape | Size (KB) | Description |
|--------|-------|-----------|-------------|
| `self_attn_layer_norm.weight` | [512] | 2 | Pre-attention norm scale |
| `self_attn_layer_norm.bias` | [512] | 2 | Pre-attention norm shift |
| `final_layer_norm.weight` | [512] | 2 | Post-FFN norm scale |
| `final_layer_norm.bias` | [512] | 2 | Post-FFN norm shift |

**LayerNorm Directory Structure:**
```
model/layers/0/
├── self_attn_layer_norm/
│   ├── weight.safetensor         # γ (gamma) scale parameter
│   ├── bias.safetensor           # β (beta) shift parameter
│   └── metadata.json
│       {
│         "normalized_shape": [512],
│         "eps": 1e-5,
│         "elementwise_affine": true,
│         "operation": "pre_attention_norm"
│       }
└── final_layer_norm/
    ├── weight.safetensor
    ├── bias.safetensor
    └── metadata.json
        {
          "normalized_shape": [512],
          "eps": 1e-5,
          "operation": "post_ffn_norm"
        }
```

#### Feed-Forward Network

| Tensor | Shape | Size (KB) | Description |
|--------|-------|-----------|-------------|
| `fc1.weight` | [2048, 512] | 4096 | First linear expansion |
| `fc1.bias` | [2048] | 8 | First linear bias |
| `fc2.weight` | [512, 2048] | 4096 | Second linear compression |
| `fc2.bias` | [512] | 2 | Second linear bias |

**FFN Directory Structure:**
```
model/layers/0/ffn/
├── fc1/
│   ├── weight.safetensor         # [2048, 512] up-projection
│   ├── bias.safetensor
│   └── metadata.json
│       {
│         "in_features": 512,
│         "out_features": 2048,
│         "activation": "relu",
│         "expansion_ratio": 4,
│         "operation": "ffn_up_proj"
│       }
└── fc2/
    ├── weight.safetensor         # [512, 2048] down-projection
    ├── bias.safetensor
    └── metadata.json
        {
          "in_features": 2048,
          "out_features": 512,
          "operation": "ffn_down_proj"
        }
```

---

### 3. Output Layer Tensors

| Tensor | Shape | Size (MB) | Description |
|--------|-------|-----------|-------------|
| `final_layer_norm.weight` | [512] | 0.002 | Final normalization scale |
| `final_layer_norm.bias` | [512] | 0.002 | Final normalization shift |
| `lm_head.weight` | [50272, 512] | 98.0 | Output vocabulary projection |

**Output Directory Structure:**
```
model/
├── final_layer_norm/
│   ├── weight.safetensor
│   ├── bias.safetensor
│   └── metadata.json
│       {
│         "normalized_shape": [512],
│         "eps": 1e-5,
│         "operation": "final_norm"
│       }
└── lm_head/
    ├── weight.safetensor          # Often tied to embed_tokens
    └── metadata.json
        {
          "in_features": 512,
          "out_features": 50272,
          "bias": false,
          "tied_to": "model.embed_tokens.weight",
          "operation": "vocabulary_projection"
        }
```

---

## Parameter Count Summary

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Token Embeddings | 25,739,264 | 7.77% |
| Position Embeddings | 1,049,600 | 0.32% |
| Self-Attention (24 layers) | 50,331,648 | 15.20% |
| FFN (24 layers) | 201,326,592 | 60.82% |
| Layer Norms (24 layers) | 98,304 | 0.03% |
| Final Layer Norm | 1,024 | 0.00% |
| LM Head | 25,739,264 | 7.77% |
| **Total** | **331,000,000** | **100%** |

---

## Tensor Flow Visualization

```
Input Token IDs: [batch, seq_len]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  embed_tokens/weight.safetensor                           │
│  [50272, 512] → Lookup → [batch, seq_len, 512]            │
└───────────────────────────────────────────────────────────┘
        │
        ▼  (+)
┌───────────────────────────────────────────────────────────┐
│  embed_positions/weight.safetensor                        │
│  [2050, 512] → Lookup → [batch, seq_len, 512]             │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  DECODER LAYER 0-23 (repeat 24 times)                     │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  self_attn_layer_norm/{weight,bias}.safetensor      │  │
│  │  LayerNorm → normalized hidden state                │  │
│  └─────────────────────────────────────────────────────┘  │
│              │                                            │
│              ▼                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  self_attn/                                         │  │
│  │  ├── q_proj/{weight,bias} → Q: [batch, seq, 512]    │  │
│  │  ├── k_proj/{weight,bias} → K: [batch, seq, 512]    │  │
│  │  ├── v_proj/{weight,bias} → V: [batch, seq, 512]    │  │
│  │  │                                                  │  │
│  │  │  Split into 16 heads: [batch, 16, seq, 32]       │  │
│  │  │  Attention: softmax(QK^T/√32) × V                │  │
│  │  │  Concat heads: [batch, seq, 512]                 │  │
│  │  │                                                  │  │
│  │  └── out_proj/{weight,bias} → [batch, seq, 512]     │  │
│  └─────────────────────────────────────────────────────┘  │
│              │  (+) Residual                              │
│              ▼                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  final_layer_norm/{weight,bias}.safetensor          │  │
│  │  LayerNorm → normalized hidden state                │  │
│  └─────────────────────────────────────────────────────┘  │
│              │                                            │
│              ▼                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  FFN (Feed-Forward Network)                         │  │
│  │  fc1/{weight,bias} → [batch, seq, 2048]             │  │
│  │       │                                             │  │
│  │       ▼  ReLU activation                            │  │
│  │  fc2/{weight,bias} → [batch, seq, 512]              │  │
│  └─────────────────────────────────────────────────────┘  │
│              │  (+) Residual                              │
│              ▼                                            │
└───────────────────────────────────────────────────────────┘
        │
        ▼  (after 24 layers)
┌───────────────────────────────────────────────────────────┐
│  final_layer_norm/{weight,bias}.safetensor                │
│  LayerNorm → [batch, seq_len, 512]                        │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  lm_head/weight.safetensor                                │
│  [50272, 512] → Linear → [batch, seq_len, 50272]          │
│  Output: logits over vocabulary                           │
└───────────────────────────────────────────────────────────┘
        │
        ▼
   Softmax → Token Probabilities
        │
        ▼
   Sample/Argmax → Next Token ID
```

---

## File System Implementation

### Directory Layout

```
model-as-repo/
├── .git/
│   ├── lfs/                       # LFS object storage
│   │   └── objects/               # Actual tensor binary data
│   └── config                     # LFS configuration
├── model/
│   ├── embed_tokens/
│   │   ├── weight.safetensor      # 98 MB
│   │   └── metadata.json
│   ├── embed_positions/
│   │   ├── weight.safetensor      # 4 MB
│   │   └── metadata.json
│   ├── layers/
│   │   ├── 0/
│   │   │   ├── self_attn/
│   │   │   │   ├── q_proj/
│   │   │   │   │   ├── weight.safetensor
│   │   │   │   │   ├── bias.safetensor
│   │   │   │   │   └── metadata.json
│   │   │   │   ├── k_proj/...
│   │   │   │   ├── v_proj/...
│   │   │   │   └── out_proj/...
│   │   │   ├── self_attn_layer_norm/
│   │   │   │   ├── weight.safetensor
│   │   │   │   ├── bias.safetensor
│   │   │   │   └── metadata.json
│   │   │   ├── ffn/
│   │   │   │   ├── fc1/...
│   │   │   │   └── fc2/...
│   │   │   └── final_layer_norm/...
│   │   ├── 1/...
│   │   ├── ...
│   │   └── 23/...
│   ├── final_layer_norm/
│   │   ├── weight.safetensor
│   │   ├── bias.safetensor
│   │   └── metadata.json
│   └── lm_head/
│       ├── weight.safetensor      # 98 MB (or symlink to embed_tokens)
│       └── metadata.json
├── config/
│   └── model_config.json
├── tokenizer/
│   ├── tokenizer.json
│   ├── vocab.json
│   └── merges.txt
└── manifest.json                  # Complete tensor inventory
```

### Manifest File

```json
{
  "model_name": "OPT-350M-Erebus-Cog",
  "total_parameters": 331000000,
  "total_size_bytes": 1324000000,
  "architecture": "OPTForCausalLM",
  "tensors": [
    {
      "name": "model.decoder.embed_tokens.weight",
      "path": "model/embed_tokens/weight.safetensor",
      "shape": [50272, 512],
      "dtype": "float32",
      "size_bytes": 102955008
    },
    {
      "name": "model.decoder.embed_positions.weight",
      "path": "model/embed_positions/weight.safetensor",
      "shape": [2050, 512],
      "dtype": "float32",
      "size_bytes": 4198400
    }
    // ... all other tensors
  ],
  "layer_count": 24,
  "tied_weights": [
    {
      "source": "model.decoder.embed_tokens.weight",
      "target": "lm_head.weight",
      "type": "symlink"
    }
  ]
}
```

---

## Git Operations on Tensors

### Loading a Tensor (git checkout)
```bash
# Checkout specific tensor file
git checkout main -- model/layers/0/self_attn/q_proj/weight.safetensor

# Load with Git LFS
git lfs pull --include="model/layers/0/**"
```

### Updating a Tensor (Training Step)
```bash
# After optimizer step updates weights
git add model/layers/0/self_attn/q_proj/weight.safetensor
git commit -m "step(1000): update layer 0 attention weights, loss=2.34"
```

### Comparing Tensors (git diff)
```bash
# Custom diff driver for safetensor files
git diff HEAD~1 -- model/layers/0/self_attn/q_proj/weight.safetensor

# Output:
# Tensor diff for model/layers/0/self_attn/q_proj/weight.safetensor
# Shape: [512, 512] (unchanged)
# Mean: 0.00023 → 0.00019 (Δ: -0.00004)
# Std:  0.0412 → 0.0408 (Δ: -0.0004)
# Max:  0.182 → 0.179 (Δ: -0.003)
# L2 norm: 14.82 → 14.67 (Δ: -0.15)
```

### Branching for Experiments
```bash
# Create branch for LoRA experiment
git checkout -b experiment/lora-rank-16

# Add LoRA adapter tensors
mkdir -p model/adapters/lora
# Add lora_A, lora_B matrices for each attention layer
```

---

## Tensor Integrity Checks

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

for tensor in $(git diff --cached --name-only | grep '.safetensor$'); do
    # Check for NaN values
    python3 -c "
import safetensors
data = safetensors.safe_open('$tensor', framework='pt')
for key in data.keys():
    t = data.get_tensor(key)
    if t.isnan().any():
        print(f'ERROR: NaN detected in {key}')
        exit(1)
    if t.isinf().any():
        print(f'ERROR: Inf detected in {key}')
        exit(1)
"
done
```

### GitHub Action for Validation
```yaml
name: Validate Tensors
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Check tensor integrity
        run: |
          python3 scripts/validate_tensors.py model/
```
