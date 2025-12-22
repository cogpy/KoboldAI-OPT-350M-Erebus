# Transformer Architecture as GitHub Repository

## Conceptual Mapping Overview

This blueprint maps the OPT-350M transformer architecture to GitHub repository structures, creating an isomorphic representation where GitHub features serve as functional analogs to neural network components.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRANSFORMER ↔ GITHUB MAPPING                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Transformer Component          │  GitHub Analog                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Model                          │  Repository                              │
│  Layers                         │  Directories (folders)                   │
│  Tensors/Weights                │  Files (.safetensors, .json)             │
│  Forward Pass                   │  GitHub Actions Workflow                 │
│  Backward Pass (Training)       │  Pull Request + Review Process           │
│  Attention Heads                │  GitHub Agents / Parallel Jobs           │
│  Hidden States                  │  Repository Secrets / Variables          │
│  Checkpoints                    │  Releases / Tags                         │
│  Gradient Updates               │  Commits                                 │
│  Hyperparameters                │  Repository Settings / Config Files      │
│  Tokenizer                      │  .github/CODEOWNERS + Templates          │
│  Inference Pipeline             │  GitHub Actions + Environments           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Mapping

### 1. Repository = Complete Model

The GitHub repository represents the entire transformer model. Just as a repository contains all code, history, and metadata, a transformer model encapsulates all weights, architecture, and configuration.

```
Repository: OPT-350M-Erebus
├── Architecture: OPT (Open Pre-trained Transformer)
├── Parameters: 331,000,000
├── Vocabulary: 50,272 tokens
├── Hidden Size: 512
├── Layers: 24
└── Attention Heads: 16
```

**GitHub Mapping:**
- Repository name = Model identifier
- Repository description = Model card
- Repository visibility = Model access (public/private)
- Repository size = Parameter count
- Fork count = Model derivatives/fine-tunes

---

### 2. Directory Structure = Layer Hierarchy

```
model/                                    # Root model namespace
├── model.embed_tokens/                   # Embedding Layer
│   └── weight.safetensor                 # [50272, 512] embedding matrix
├── model.embed_positions/                # Positional Encoding
│   └── weight.safetensor                 # [2050, 512] position embeddings
├── model.layers/                         # Decoder Stack
│   ├── 0/                                # Layer 0
│   │   ├── self_attn/                    # Self-Attention Module
│   │   │   ├── q_proj.weight             # Query projection
│   │   │   ├── k_proj.weight             # Key projection
│   │   │   ├── v_proj.weight             # Value projection
│   │   │   └── out_proj.weight           # Output projection
│   │   ├── self_attn_layer_norm/         # Pre-attention LayerNorm
│   │   │   ├── weight.safetensor
│   │   │   └── bias.safetensor
│   │   ├── fc1/                          # FFN first layer
│   │   │   ├── weight.safetensor         # [2048, 512]
│   │   │   └── bias.safetensor
│   │   ├── fc2/                          # FFN second layer
│   │   │   ├── weight.safetensor         # [512, 2048]
│   │   │   └── bias.safetensor
│   │   └── final_layer_norm/             # Post-FFN LayerNorm
│   │       ├── weight.safetensor
│   │       └── bias.safetensor
│   ├── 1/                                # Layer 1
│   │   └── ...
│   └── 23/                               # Layer 23 (final)
│       └── ...
├── model.final_layer_norm/               # Final LayerNorm
│   ├── weight.safetensor
│   └── bias.safetensor
└── lm_head/                              # Language Model Head
    └── weight.safetensor                 # [50272, 512] output projection
```

---

### 3. Files = Tensors/Weights

Each tensor (weight matrix or bias vector) maps to a file. The file format preserves tensor metadata.

| Tensor Type | File Extension | GitHub Analog |
|-------------|---------------|---------------|
| Weight matrices | `.safetensor` | Binary files (LFS) |
| Biases | `.safetensor` | Binary files (LFS) |
| Configuration | `.json` | Text files |
| Tokenizer vocab | `.json` | Text files |
| Merges | `.txt` | Text files |

**Tensor File Metadata (as Git attributes):**
```gitattributes
*.safetensor filter=lfs diff=lfs merge=lfs -text
*.safetensor linguist-generated=true
*.safetensor lockable=true
```

---

### 4. Branches = Model States

```
main                    # Production model (inference-ready)
├── training            # Active training state
├── checkpoint-1000     # Training checkpoint at step 1000
├── checkpoint-5000     # Training checkpoint at step 5000
├── fine-tune/erebus    # Fine-tuning branch for Erebus
├── quantized/int8      # INT8 quantized version
└── experimental/lora   # LoRA adapter experiments
```

**Branch Protection Rules = Model Integrity:**
- `main` requires passing inference tests
- No force pushes (preserve weight history)
- Require reviews for weight updates

---

### 5. Commits = Gradient Updates / Weight Changes

Each training step that updates weights maps to a commit:

```
commit abc123 (training)
Author: Optimizer <adam@training.local>
Date:   Training Step 1000

    Update weights: loss=2.341 → 2.298

    - Updated 331M parameters
    - Learning rate: 1e-4
    - Batch size: 32
    - Gradient norm: 0.847

 model/layers/0/self_attn/q_proj.weight | Binary change
 model/layers/0/self_attn/k_proj.weight | Binary change
 ...
 234 files changed
```

**Commit Message Convention:**
```
<optimizer>(<layer>): <metric_change>

- Loss: <before> → <after>
- Perplexity: <value>
- Gradient norm: <value>
- Learning rate: <value>
```

---

### 6. Pull Requests = Proposed Weight Updates (Training Batches)

A pull request represents a proposed gradient update before it's applied:

```markdown
## PR #1000: Gradient Update Batch 1000

**Source:** `training` → **Target:** `main`

### Changes
- Processes batch of 32 samples
- Computes gradients via backpropagation
- Proposes weight updates across all layers

### Metrics
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Loss | 2.341 | 2.298 | -0.043 |
| Perplexity | 10.39 | 9.95 | -0.44 |

### Review Checklist
- [ ] Gradient magnitude within bounds
- [ ] No NaN values in weights
- [ ] Loss decreased
- [ ] Validation metrics stable

### Auto-merge Criteria
Merge automatically if:
- Loss improvement > 0.01
- No gradient explosion detected
- All validation checks pass
```

---

### 7. GitHub Actions = Forward/Backward Passes

#### Forward Pass Workflow

```yaml
# .github/workflows/forward_pass.yml
name: Forward Pass (Inference)

on:
  workflow_dispatch:
    inputs:
      input_tokens:
        description: 'Input token IDs'
        required: true
      max_length:
        description: 'Maximum generation length'
        default: '100'

jobs:
  embedding:
    name: "Layer: Token + Position Embedding"
    runs-on: ubuntu-latest
    outputs:
      hidden_state: ${{ steps.embed.outputs.hidden_state }}
    steps:
      - name: Token Embedding Lookup
        id: token_embed
        run: |
          # hidden = embed_tokens(input_ids)
          # Shape: [batch, seq_len, 512]

      - name: Position Embedding
        id: pos_embed
        run: |
          # positions = embed_positions(position_ids)
          # hidden = hidden + positions

  decoder_layers:
    name: "Decoder Layer ${{ matrix.layer }}"
    needs: [embedding]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        layer: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
      max-parallel: 1  # Sequential processing
    steps:
      - name: Self-Attention
        uses: ./.github/actions/self-attention
        with:
          hidden_state: ${{ needs.embedding.outputs.hidden_state }}
          layer: ${{ matrix.layer }}

      - name: Feed-Forward Network
        uses: ./.github/actions/feed-forward
        with:
          hidden_state: ${{ steps.self-attention.outputs.hidden_state }}

  output:
    name: "Layer: LM Head (Output)"
    needs: [decoder_layers]
    runs-on: ubuntu-latest
    steps:
      - name: Final Layer Norm
        run: |
          # hidden = layer_norm(hidden)

      - name: Project to Vocabulary
        run: |
          # logits = lm_head(hidden)
          # Shape: [batch, seq_len, 50272]

      - name: Sample Next Token
        run: |
          # next_token = sample(logits, temperature=0.8)
```

#### Backward Pass Workflow (Training)

```yaml
# .github/workflows/backward_pass.yml
name: Backward Pass (Training)

on:
  schedule:
    - cron: '*/5 * * * *'  # Every training step

jobs:
  forward:
    uses: ./.github/workflows/forward_pass.yml

  compute_loss:
    needs: [forward]
    runs-on: ubuntu-latest
    outputs:
      loss: ${{ steps.loss.outputs.value }}
    steps:
      - name: Cross-Entropy Loss
        id: loss
        run: |
          # loss = cross_entropy(logits, targets)

  backpropagate:
    needs: [compute_loss]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        layer: [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    steps:
      - name: Compute Gradients
        run: |
          # grad = backward(loss, layer_params)

  optimizer_step:
    needs: [backpropagate]
    runs-on: ubuntu-latest
    steps:
      - name: Adam Optimizer Update
        run: |
          # params = params - lr * adam_update(grads)

      - name: Commit Weight Updates
        run: |
          git add model/
          git commit -m "Training step: loss=${{ needs.compute_loss.outputs.loss }}"
```

---

### 8. GitHub Agents = Attention Heads

Multi-head attention parallelism maps to GitHub Copilot Agents or parallel workflow jobs:

```yaml
# .github/actions/multi-head-attention/action.yml
name: Multi-Head Attention
description: 'Parallel attention computation across 16 heads'

runs:
  using: composite
  steps:
    - name: Split into Heads
      run: |
        # Q, K, V each split into 16 heads
        # Each head: [batch, seq_len, 32] (512/16 = 32)

attention_heads:
  strategy:
    matrix:
      head: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    max-parallel: 16  # All heads in parallel
  steps:
    - name: "Attention Head ${{ matrix.head }}"
      run: |
        # scores = Q @ K.T / sqrt(32)
        # weights = softmax(scores + mask)
        # output = weights @ V

concatenate_heads:
  needs: [attention_heads]
  steps:
    - name: Concatenate Head Outputs
      run: |
        # output = concat(head_outputs, dim=-1)
        # output = out_proj(output)
```

**Agent Mapping:**
| Attention Head | GitHub Agent Role |
|----------------|-------------------|
| Head 0 | Syntax patterns |
| Head 1 | Semantic relationships |
| Head 2 | Positional dependencies |
| Head 3-15 | Specialized pattern detectors |

---

### 9. Repository Secrets = Hidden States

Hidden states (intermediate activations) are stored as encrypted repository secrets:

```yaml
# Hidden states flow through the model
secrets:
  HIDDEN_STATE_L0: # Output of layer 0 [batch, seq, 512]
  HIDDEN_STATE_L1: # Output of layer 1
  # ...
  HIDDEN_STATE_L23: # Final layer output

  # Attention caches for generation
  KEY_CACHE_L0: # Cached keys for layer 0
  VALUE_CACHE_L0: # Cached values for layer 0
  # ...
```

**Environment Variables = Hyperparameters:**
```yaml
env:
  HIDDEN_SIZE: 512
  NUM_LAYERS: 24
  NUM_HEADS: 16
  HEAD_DIM: 32
  FFN_DIM: 2048
  VOCAB_SIZE: 50272
  MAX_POSITION: 2048
  DROPOUT: 0.1
```

---

### 10. Releases = Model Checkpoints

```
v1.0.0 - Initial OPT-350M-Erebus release
├── model.safetensors (660 MB)
├── config.json
├── tokenizer.json
├── vocab.json
└── merges.txt

v1.1.0 - OpenCog Optimized (Cog)
├── model.safetensors (660 MB) - Optimized tensors
├── config.json
├── tokenizer.json
├── opencog_metadata.json
└── transformation_report.txt

v2.0.0-quantized - INT8 Quantized
├── model-int8.safetensors (165 MB)
├── quantization_config.json
└── ...
```

---

### 11. Issues = Loss Signals / Errors to Minimize

```markdown
## Issue #1: High Perplexity on Code Samples

**Labels:** `loss-signal`, `domain:code`, `priority:high`

### Description
Model shows perplexity of 45.2 on code completion tasks,
compared to 12.3 baseline on natural language.

### Metrics
- Natural language perplexity: 12.3
- Code perplexity: 45.2
- Target: < 20.0

### Proposed Fix
Fine-tune on code corpus for 10k steps.

### Resolution
Closed by PR #500 - Code fine-tuning reduces perplexity to 18.7
```

---

### 12. Projects/Milestones = Training Epochs

```
Project: Training OPT-350M-Erebus
├── Epoch 1 (100%)
│   ├── Steps 0-10000 ✓
│   ├── Validation loss: 3.2 → 2.8 ✓
│   └── Checkpoint saved: v0.1.0 ✓
├── Epoch 2 (100%)
│   ├── Steps 10001-20000 ✓
│   ├── Validation loss: 2.8 → 2.4 ✓
│   └── Checkpoint saved: v0.2.0 ✓
└── Epoch 3 (In Progress)
    ├── Steps 20001-30000 (65%)
    ├── Current loss: 2.35
    └── ETA: 4 hours
```

---

## Complete Repository Structure

```
OPT-350M-Erebus-Cog/
├── .github/
│   ├── workflows/
│   │   ├── forward_pass.yml        # Inference pipeline
│   │   ├── backward_pass.yml       # Training pipeline
│   │   ├── benchmark.yml           # Performance evaluation
│   │   ├── validate_weights.yml    # Weight integrity checks
│   │   └── quantize.yml            # Model quantization
│   ├── actions/
│   │   ├── self-attention/         # Attention computation
│   │   ├── feed-forward/           # FFN computation
│   │   ├── layer-norm/             # Normalization
│   │   ├── embedding/              # Token/position embedding
│   │   └── sampling/               # Token sampling strategies
│   ├── ISSUE_TEMPLATE/
│   │   ├── loss_signal.md          # Report high loss
│   │   ├── gradient_issue.md       # Report gradient problems
│   │   └── inference_bug.md        # Report inference issues
│   └── CODEOWNERS                  # Tokenizer (who handles what)
├── model/
│   ├── embed_tokens/
│   │   └── weight.safetensor       # [50272, 512]
│   ├── embed_positions/
│   │   └── weight.safetensor       # [2050, 512]
│   ├── layers/
│   │   ├── 0/...                   # Layer 0 weights
│   │   ├── 1/...                   # Layer 1 weights
│   │   └── 23/...                  # Layer 23 weights
│   ├── final_layer_norm/
│   │   ├── weight.safetensor
│   │   └── bias.safetensor
│   └── lm_head/
│       └── weight.safetensor       # [50272, 512]
├── config/
│   ├── model_config.json           # Architecture config
│   ├── training_config.json        # Training hyperparameters
│   └── generation_config.json      # Inference parameters
├── tokenizer/
│   ├── tokenizer.json
│   ├── vocab.json
│   └── merges.txt
├── checkpoints/                    # Training snapshots
│   ├── step-10000/
│   ├── step-20000/
│   └── best/
└── README.md                       # Model card
```

---

## Functional Equivalence Table

| Neural Network Operation | GitHub Operation |
|--------------------------|------------------|
| `model.load()` | `git clone` |
| `model.save()` | `git commit && git push` |
| `optimizer.step()` | Merge PR |
| `loss.backward()` | Create PR with gradient changes |
| `model.eval()` | Switch to `main` branch |
| `model.train()` | Switch to `training` branch |
| `torch.no_grad()` | Read-only repository access |
| `checkpoint.save()` | Create release/tag |
| `model.to(device)` | Deploy to environment |
| `tokenizer.encode()` | CODEOWNERS file parsing |
| `attention(Q, K, V)` | Parallel agent jobs |
| `F.softmax()` | Webhook probability routing |
| `nn.Dropout()` | Random file exclusion in builds |
| `nn.LayerNorm()` | Linting/formatting actions |

---

## Next Steps

See detailed documentation:
- [TENSOR_MAPPING.md](./TENSOR_MAPPING.md) - Detailed tensor-to-file mapping
- [WORKFLOW_ACTIONS.md](./WORKFLOW_ACTIONS.md) - GitHub Actions for neural ops
- [AGENTS_ATTENTION.md](./AGENTS_ATTENTION.md) - Agents as attention heads
- [TRAINING_AS_CI.md](./TRAINING_AS_CI.md) - Training loop as CI/CD
