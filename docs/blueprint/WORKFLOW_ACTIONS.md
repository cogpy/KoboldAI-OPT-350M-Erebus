# GitHub Workflows & Actions as Neural Network Operations

## Overview

This document maps transformer neural network operations to GitHub Actions workflows and reusable actions. Each computational step in the forward/backward pass is represented as a workflow job or composite action.

---

## Workflow Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE WORKFLOW                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   trigger: workflow_dispatch (input tokens)                                │
│       │                                                                    │
│       ▼                                                                    │
│   ┌─────────────────┐                                                      │
│   │   Tokenization  │  Action: tokenize                                    │
│   └────────┬────────┘                                                      │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────┐                                                      │
│   │   Embedding     │  Action: embed                                       │
│   └────────┬────────┘                                                      │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    DECODER LAYERS (0-23)                            │  │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │  │
│   │   │  LayerNorm  │→ │  Attention  │→ │  Residual   │                │  │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                │  │
│   │          │                                                          │  │
│   │          ▼                                                          │  │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │  │
│   │   │  LayerNorm  │→ │     FFN     │→ │  Residual   │                │  │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────┐                                                      │
│   │  Final Norm     │  Action: layer-norm                                  │
│   └────────┬────────┘                                                      │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────┐                                                      │
│   │    LM Head      │  Action: lm-head                                     │
│   └────────┬────────┘                                                      │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────┐                                                      │
│   │    Sampling     │  Action: sample-token                                │
│   └─────────────────┘                                                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Main Inference Workflow

```yaml
# .github/workflows/inference.yml
name: Inference Pipeline (Forward Pass)

on:
  workflow_dispatch:
    inputs:
      prompt:
        description: 'Input text prompt'
        required: true
        type: string
      max_tokens:
        description: 'Maximum tokens to generate'
        required: false
        default: '100'
        type: string
      temperature:
        description: 'Sampling temperature (0.0-2.0)'
        required: false
        default: '0.8'
        type: string
      top_p:
        description: 'Nucleus sampling threshold'
        required: false
        default: '0.95'
        type: string

env:
  HIDDEN_SIZE: 512
  NUM_LAYERS: 24
  NUM_HEADS: 16
  VOCAB_SIZE: 50272

jobs:
  # ============================================================
  # STAGE 1: Tokenization
  # ============================================================
  tokenize:
    name: "Tokenize Input"
    runs-on: ubuntu-latest
    outputs:
      input_ids: ${{ steps.tokenize.outputs.input_ids }}
      attention_mask: ${{ steps.tokenize.outputs.attention_mask }}
      seq_length: ${{ steps.tokenize.outputs.seq_length }}
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
          sparse-checkout: |
            tokenizer/

      - name: Tokenize prompt
        id: tokenize
        uses: ./.github/actions/tokenize
        with:
          text: ${{ inputs.prompt }}
          tokenizer_path: tokenizer/

  # ============================================================
  # STAGE 2: Embedding Layer
  # ============================================================
  embedding:
    name: "Token + Position Embedding"
    needs: [tokenize]
    runs-on: ubuntu-latest
    outputs:
      hidden_state: ${{ steps.embed.outputs.hidden_state }}
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
          sparse-checkout: |
            model/embed_tokens/
            model/embed_positions/

      - name: Compute embeddings
        id: embed
        uses: ./.github/actions/embedding
        with:
          input_ids: ${{ needs.tokenize.outputs.input_ids }}
          seq_length: ${{ needs.tokenize.outputs.seq_length }}

  # ============================================================
  # STAGE 3: Decoder Layers (Sequential)
  # ============================================================
  decoder_layer_0:
    name: "Decoder Layer 0"
    needs: [embedding]
    runs-on: ubuntu-latest
    outputs:
      hidden_state: ${{ steps.layer.outputs.hidden_state }}
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
          sparse-checkout: |
            model/layers/0/

      - name: Process layer 0
        id: layer
        uses: ./.github/actions/decoder-layer
        with:
          hidden_state: ${{ needs.embedding.outputs.hidden_state }}
          layer_idx: 0

  decoder_layer_1:
    name: "Decoder Layer 1"
    needs: [decoder_layer_0]
    runs-on: ubuntu-latest
    outputs:
      hidden_state: ${{ steps.layer.outputs.hidden_state }}
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
          sparse-checkout: |
            model/layers/1/

      - name: Process layer 1
        id: layer
        uses: ./.github/actions/decoder-layer
        with:
          hidden_state: ${{ needs.decoder_layer_0.outputs.hidden_state }}
          layer_idx: 1

  # ... Layers 2-22 follow same pattern ...

  decoder_layer_23:
    name: "Decoder Layer 23 (Final)"
    needs: [decoder_layer_22]
    runs-on: ubuntu-latest
    outputs:
      hidden_state: ${{ steps.layer.outputs.hidden_state }}
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
          sparse-checkout: |
            model/layers/23/

      - name: Process layer 23
        id: layer
        uses: ./.github/actions/decoder-layer
        with:
          hidden_state: ${{ needs.decoder_layer_22.outputs.hidden_state }}
          layer_idx: 23

  # ============================================================
  # STAGE 4: Output Processing
  # ============================================================
  output_projection:
    name: "Output Layer (LM Head)"
    needs: [decoder_layer_23]
    runs-on: ubuntu-latest
    outputs:
      logits: ${{ steps.project.outputs.logits }}
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
          sparse-checkout: |
            model/final_layer_norm/
            model/lm_head/

      - name: Final layer norm
        id: norm
        uses: ./.github/actions/layer-norm
        with:
          hidden_state: ${{ needs.decoder_layer_23.outputs.hidden_state }}
          weight_path: model/final_layer_norm/weight.safetensor
          bias_path: model/final_layer_norm/bias.safetensor

      - name: Project to vocabulary
        id: project
        uses: ./.github/actions/linear
        with:
          input: ${{ steps.norm.outputs.output }}
          weight_path: model/lm_head/weight.safetensor

  # ============================================================
  # STAGE 5: Token Sampling
  # ============================================================
  sample:
    name: "Sample Next Token"
    needs: [output_projection, tokenize]
    runs-on: ubuntu-latest
    outputs:
      next_token: ${{ steps.sample.outputs.token_id }}
      token_text: ${{ steps.sample.outputs.token_text }}
    steps:
      - uses: actions/checkout@v4

      - name: Sample from logits
        id: sample
        uses: ./.github/actions/sample-token
        with:
          logits: ${{ needs.output_projection.outputs.logits }}
          temperature: ${{ inputs.temperature }}
          top_p: ${{ inputs.top_p }}

      - name: Decode token
        id: decode
        uses: ./.github/actions/detokenize
        with:
          token_id: ${{ steps.sample.outputs.token_id }}

  # ============================================================
  # STAGE 6: Autoregressive Loop (Recursive Call)
  # ============================================================
  continue_generation:
    name: "Continue Generation"
    needs: [sample, tokenize]
    if: ${{ needs.sample.outputs.next_token != '[EOS]' && github.event.inputs.max_tokens > 1 }}
    uses: ./.github/workflows/inference.yml
    with:
      prompt: "${{ inputs.prompt }}${{ needs.sample.outputs.token_text }}"
      max_tokens: ${{ fromJSON(inputs.max_tokens) - 1 }}
      temperature: ${{ inputs.temperature }}
      top_p: ${{ inputs.top_p }}
```

---

## Reusable Actions

### Action: Tokenize

```yaml
# .github/actions/tokenize/action.yml
name: 'Tokenize'
description: 'Convert text to token IDs using BPE tokenizer'

inputs:
  text:
    description: 'Input text to tokenize'
    required: true
  tokenizer_path:
    description: 'Path to tokenizer files'
    required: true
  max_length:
    description: 'Maximum sequence length'
    default: '2048'

outputs:
  input_ids:
    description: 'Token ID sequence'
    value: ${{ steps.run.outputs.input_ids }}
  attention_mask:
    description: 'Attention mask (1s for real tokens)'
    value: ${{ steps.run.outputs.attention_mask }}
  seq_length:
    description: 'Sequence length'
    value: ${{ steps.run.outputs.seq_length }}

runs:
  using: 'composite'
  steps:
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install dependencies
      shell: bash
      run: pip install transformers

    - name: Tokenize
      id: run
      shell: python
      run: |
        import os
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("${{ inputs.tokenizer_path }}")
        text = """${{ inputs.text }}"""

        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=${{ inputs.max_length }},
            truncation=True
        )

        input_ids = tokens.input_ids[0].tolist()
        attention_mask = tokens.attention_mask[0].tolist()

        # Write outputs
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"input_ids={input_ids}\n")
            f.write(f"attention_mask={attention_mask}\n")
            f.write(f"seq_length={len(input_ids)}\n")
```

### Action: Embedding

```yaml
# .github/actions/embedding/action.yml
name: 'Embedding Layer'
description: 'Compute token and position embeddings'

inputs:
  input_ids:
    description: 'Token ID sequence'
    required: true
  seq_length:
    description: 'Sequence length'
    required: true

outputs:
  hidden_state:
    description: 'Combined embeddings [seq_len, hidden_size]'
    value: ${{ steps.compute.outputs.hidden_state }}

runs:
  using: 'composite'
  steps:
    - name: Compute embeddings
      id: compute
      shell: python
      run: |
        import torch
        import safetensors.torch
        import json
        import os

        # Load embedding weights
        embed_tokens = safetensors.torch.load_file(
            "model/embed_tokens/weight.safetensor"
        )["weight"]  # [50272, 512]

        embed_positions = safetensors.torch.load_file(
            "model/embed_positions/weight.safetensor"
        )["weight"]  # [2050, 512]

        # Parse input IDs
        input_ids = torch.tensor(${{ inputs.input_ids }})
        seq_len = ${{ inputs.seq_length }}

        # Token embedding lookup
        token_embeds = embed_tokens[input_ids]  # [seq_len, 512]

        # Position embedding (with offset of 2 for OPT)
        position_ids = torch.arange(2, seq_len + 2)
        position_embeds = embed_positions[position_ids]  # [seq_len, 512]

        # Combine embeddings
        hidden_state = token_embeds + position_embeds  # [seq_len, 512]

        # Serialize output (base64 encoded tensor)
        import base64
        import io
        buffer = io.BytesIO()
        torch.save(hidden_state, buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode()

        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"hidden_state={encoded}\n")
```

### Action: Layer Norm

```yaml
# .github/actions/layer-norm/action.yml
name: 'Layer Normalization'
description: 'Apply layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta'

inputs:
  hidden_state:
    description: 'Input hidden state (base64 encoded tensor)'
    required: true
  weight_path:
    description: 'Path to gamma (scale) weights'
    required: true
  bias_path:
    description: 'Path to beta (shift) weights'
    required: true
  eps:
    description: 'Epsilon for numerical stability'
    default: '1e-5'

outputs:
  output:
    description: 'Normalized hidden state'
    value: ${{ steps.compute.outputs.output }}

runs:
  using: 'composite'
  steps:
    - name: Apply LayerNorm
      id: compute
      shell: python
      run: |
        import torch
        import safetensors.torch
        import base64
        import io
        import os

        # Load input
        encoded = "${{ inputs.hidden_state }}"
        buffer = io.BytesIO(base64.b64decode(encoded))
        hidden_state = torch.load(buffer)  # [seq_len, 512]

        # Load parameters
        gamma = safetensors.torch.load_file("${{ inputs.weight_path }}")["weight"]
        beta = safetensors.torch.load_file("${{ inputs.bias_path }}")["weight"]

        # Layer normalization
        eps = ${{ inputs.eps }}
        mean = hidden_state.mean(dim=-1, keepdim=True)
        var = hidden_state.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (hidden_state - mean) / torch.sqrt(var + eps)
        output = normalized * gamma + beta

        # Serialize output
        buffer = io.BytesIO()
        torch.save(output, buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode()

        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"output={encoded}\n")
```

### Action: Self-Attention

```yaml
# .github/actions/self-attention/action.yml
name: 'Multi-Head Self-Attention'
description: 'Compute scaled dot-product attention with 16 parallel heads'

inputs:
  hidden_state:
    description: 'Input hidden state'
    required: true
  layer_idx:
    description: 'Layer index (0-23)'
    required: true
  causal:
    description: 'Apply causal mask'
    default: 'true'

outputs:
  output:
    description: 'Attention output'
    value: ${{ steps.attention.outputs.output }}
  attention_weights:
    description: 'Attention weight matrices (optional)'
    value: ${{ steps.attention.outputs.weights }}

runs:
  using: 'composite'
  steps:
    - name: Multi-Head Attention
      id: attention
      shell: python
      run: |
        import torch
        import torch.nn.functional as F
        import safetensors.torch
        import base64
        import io
        import os
        import math

        layer_idx = ${{ inputs.layer_idx }}
        num_heads = 16
        head_dim = 32
        hidden_size = 512

        # Load input
        encoded = "${{ inputs.hidden_state }}"
        buffer = io.BytesIO(base64.b64decode(encoded))
        hidden = torch.load(buffer)  # [seq_len, 512]
        seq_len = hidden.shape[0]

        # Load projection weights
        base_path = f"model/layers/{layer_idx}/self_attn"

        q_weight = safetensors.torch.load_file(f"{base_path}/q_proj/weight.safetensor")["weight"]
        q_bias = safetensors.torch.load_file(f"{base_path}/q_proj/bias.safetensor")["weight"]
        k_weight = safetensors.torch.load_file(f"{base_path}/k_proj/weight.safetensor")["weight"]
        k_bias = safetensors.torch.load_file(f"{base_path}/k_proj/bias.safetensor")["weight"]
        v_weight = safetensors.torch.load_file(f"{base_path}/v_proj/weight.safetensor")["weight"]
        v_bias = safetensors.torch.load_file(f"{base_path}/v_proj/bias.safetensor")["weight"]
        out_weight = safetensors.torch.load_file(f"{base_path}/out_proj/weight.safetensor")["weight"]
        out_bias = safetensors.torch.load_file(f"{base_path}/out_proj/bias.safetensor")["weight"]

        # Compute Q, K, V projections
        Q = F.linear(hidden, q_weight, q_bias)  # [seq_len, 512]
        K = F.linear(hidden, k_weight, k_bias)
        V = F.linear(hidden, v_weight, v_bias)

        # Reshape for multi-head attention
        Q = Q.view(seq_len, num_heads, head_dim).transpose(0, 1)  # [16, seq_len, 32]
        K = K.view(seq_len, num_heads, head_dim).transpose(0, 1)
        V = V.view(seq_len, num_heads, head_dim).transpose(0, 1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [16, seq_len, seq_len]

        # Apply causal mask
        if ${{ inputs.causal }}:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [16, seq_len, seq_len]

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [16, seq_len, 32]

        # Concatenate heads
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, hidden_size)

        # Output projection
        output = F.linear(attn_output, out_weight, out_bias)  # [seq_len, 512]

        # Serialize outputs
        buffer = io.BytesIO()
        torch.save(output, buffer)
        encoded_output = base64.b64encode(buffer.getvalue()).decode()

        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"output={encoded_output}\n")
```

### Action: Feed-Forward Network

```yaml
# .github/actions/feed-forward/action.yml
name: 'Feed-Forward Network'
description: 'Two-layer MLP with ReLU activation: FFN(x) = ReLU(xW1 + b1)W2 + b2'

inputs:
  hidden_state:
    description: 'Input hidden state'
    required: true
  layer_idx:
    description: 'Layer index (0-23)'
    required: true

outputs:
  output:
    description: 'FFN output'
    value: ${{ steps.ffn.outputs.output }}

runs:
  using: 'composite'
  steps:
    - name: FFN Forward
      id: ffn
      shell: python
      run: |
        import torch
        import torch.nn.functional as F
        import safetensors.torch
        import base64
        import io
        import os

        layer_idx = ${{ inputs.layer_idx }}

        # Load input
        encoded = "${{ inputs.hidden_state }}"
        buffer = io.BytesIO(base64.b64decode(encoded))
        hidden = torch.load(buffer)  # [seq_len, 512]

        # Load weights
        base_path = f"model/layers/{layer_idx}/ffn"

        fc1_weight = safetensors.torch.load_file(f"{base_path}/fc1/weight.safetensor")["weight"]
        fc1_bias = safetensors.torch.load_file(f"{base_path}/fc1/bias.safetensor")["weight"]
        fc2_weight = safetensors.torch.load_file(f"{base_path}/fc2/weight.safetensor")["weight"]
        fc2_bias = safetensors.torch.load_file(f"{base_path}/fc2/bias.safetensor")["weight"]

        # FFN computation
        # Up projection: [seq_len, 512] → [seq_len, 2048]
        intermediate = F.linear(hidden, fc1_weight, fc1_bias)

        # ReLU activation
        intermediate = F.relu(intermediate)

        # Down projection: [seq_len, 2048] → [seq_len, 512]
        output = F.linear(intermediate, fc2_weight, fc2_bias)

        # Serialize output
        buffer = io.BytesIO()
        torch.save(output, buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode()

        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"output={encoded}\n")
```

### Action: Decoder Layer (Composite)

```yaml
# .github/actions/decoder-layer/action.yml
name: 'Decoder Layer'
description: 'Complete decoder layer: LN → Attention → Residual → LN → FFN → Residual'

inputs:
  hidden_state:
    description: 'Input hidden state'
    required: true
  layer_idx:
    description: 'Layer index (0-23)'
    required: true

outputs:
  hidden_state:
    description: 'Output hidden state'
    value: ${{ steps.final.outputs.hidden_state }}

runs:
  using: 'composite'
  steps:
    # Pre-attention LayerNorm
    - name: Self-Attention LayerNorm
      id: attn_ln
      uses: ./.github/actions/layer-norm
      with:
        hidden_state: ${{ inputs.hidden_state }}
        weight_path: model/layers/${{ inputs.layer_idx }}/self_attn_layer_norm/weight.safetensor
        bias_path: model/layers/${{ inputs.layer_idx }}/self_attn_layer_norm/bias.safetensor

    # Multi-Head Self-Attention
    - name: Self-Attention
      id: attention
      uses: ./.github/actions/self-attention
      with:
        hidden_state: ${{ steps.attn_ln.outputs.output }}
        layer_idx: ${{ inputs.layer_idx }}

    # Residual connection after attention
    - name: Attention Residual
      id: attn_residual
      shell: python
      run: |
        import torch
        import base64
        import io
        import os

        # Load original input
        buffer = io.BytesIO(base64.b64decode("${{ inputs.hidden_state }}"))
        residual = torch.load(buffer)

        # Load attention output
        buffer = io.BytesIO(base64.b64decode("${{ steps.attention.outputs.output }}"))
        attn_out = torch.load(buffer)

        # Add residual
        hidden = residual + attn_out

        # Output
        buffer = io.BytesIO()
        torch.save(hidden, buffer)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"hidden_state={base64.b64encode(buffer.getvalue()).decode()}\n")

    # Post-attention / Pre-FFN LayerNorm
    - name: Final LayerNorm
      id: ffn_ln
      uses: ./.github/actions/layer-norm
      with:
        hidden_state: ${{ steps.attn_residual.outputs.hidden_state }}
        weight_path: model/layers/${{ inputs.layer_idx }}/final_layer_norm/weight.safetensor
        bias_path: model/layers/${{ inputs.layer_idx }}/final_layer_norm/bias.safetensor

    # Feed-Forward Network
    - name: FFN
      id: ffn
      uses: ./.github/actions/feed-forward
      with:
        hidden_state: ${{ steps.ffn_ln.outputs.output }}
        layer_idx: ${{ inputs.layer_idx }}

    # Residual connection after FFN
    - name: FFN Residual
      id: final
      shell: python
      run: |
        import torch
        import base64
        import io
        import os

        # Load post-attention hidden state
        buffer = io.BytesIO(base64.b64decode("${{ steps.attn_residual.outputs.hidden_state }}"))
        residual = torch.load(buffer)

        # Load FFN output
        buffer = io.BytesIO(base64.b64decode("${{ steps.ffn.outputs.output }}"))
        ffn_out = torch.load(buffer)

        # Add residual
        hidden = residual + ffn_out

        # Output
        buffer = io.BytesIO()
        torch.save(hidden, buffer)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"hidden_state={base64.b64encode(buffer.getvalue()).decode()}\n")
```

### Action: Sample Token

```yaml
# .github/actions/sample-token/action.yml
name: 'Sample Token'
description: 'Sample next token from logits distribution'

inputs:
  logits:
    description: 'Output logits [seq_len, vocab_size]'
    required: true
  temperature:
    description: 'Sampling temperature'
    default: '1.0'
  top_p:
    description: 'Nucleus sampling threshold'
    default: '1.0'
  top_k:
    description: 'Top-k sampling (0 = disabled)'
    default: '0'

outputs:
  token_id:
    description: 'Sampled token ID'
    value: ${{ steps.sample.outputs.token_id }}
  probability:
    description: 'Token probability'
    value: ${{ steps.sample.outputs.probability }}

runs:
  using: 'composite'
  steps:
    - name: Sample
      id: sample
      shell: python
      run: |
        import torch
        import torch.nn.functional as F
        import base64
        import io
        import os

        # Load logits
        buffer = io.BytesIO(base64.b64decode("${{ inputs.logits }}"))
        logits = torch.load(buffer)  # [seq_len, 50272]

        # Get last position logits
        next_token_logits = logits[-1, :]  # [50272]

        # Temperature scaling
        temperature = ${{ inputs.temperature }}
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Top-k filtering
        top_k = ${{ inputs.top_k }}
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        top_p = ${{ inputs.top_p }}
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        token_prob = probs[next_token].item()

        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"token_id={next_token.item()}\n")
            f.write(f"probability={token_prob}\n")
```

---

## Training Workflow (Backward Pass)

```yaml
# .github/workflows/training.yml
name: Training Pipeline (Backward Pass)

on:
  workflow_dispatch:
    inputs:
      batch_data:
        description: 'Training batch (JSON)'
        required: true
      learning_rate:
        description: 'Learning rate'
        default: '1e-4'

jobs:
  forward_pass:
    name: "Forward Pass"
    uses: ./.github/workflows/inference.yml
    with:
      prompt: ${{ fromJSON(inputs.batch_data).input }}
      max_tokens: '0'  # Don't generate, just compute

  compute_loss:
    name: "Compute Cross-Entropy Loss"
    needs: [forward_pass]
    runs-on: ubuntu-latest
    outputs:
      loss: ${{ steps.loss.outputs.value }}
      gradients: ${{ steps.loss.outputs.gradients }}
    steps:
      - name: Cross-Entropy Loss
        id: loss
        run: |
          # loss = -sum(target * log(softmax(logits)))
          # Compute gradients via autograd

  backprop_layer_23:
    name: "Backprop Layer 23"
    needs: [compute_loss]
    runs-on: ubuntu-latest
    outputs:
      gradients: ${{ steps.backprop.outputs.gradients }}
    steps:
      - name: Compute layer gradients
        id: backprop
        run: |
          # d_loss/d_weights for layer 23

  # ... Backprop through layers 22-0 ...

  optimizer_step:
    name: "Adam Optimizer Step"
    needs: [backprop_layer_0]
    runs-on: ubuntu-latest
    steps:
      - name: Update weights
        run: |
          # For each parameter:
          # m = beta1 * m + (1 - beta1) * grad
          # v = beta2 * v + (1 - beta2) * grad^2
          # param = param - lr * m / (sqrt(v) + eps)

      - name: Commit weight updates
        run: |
          git add model/
          git commit -m "Training step: loss=${{ needs.compute_loss.outputs.loss }}"
          git push

  create_checkpoint:
    name: "Save Checkpoint"
    needs: [optimizer_step]
    if: ${{ github.event.inputs.step % 1000 == 0 }}
    runs-on: ubuntu-latest
    steps:
      - name: Create release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: checkpoint-${{ github.run_number }}
          files: |
            model/**/*.safetensor
```

---

## Workflow Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GITHUB ACTIONS RUNNER POOL                         │
│                     (Analogous to GPU Compute Units)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│   │ Runner  │  │ Runner  │  │ Runner  │  │ Runner  │  │ Runner  │      │
│   │   1     │  │   2     │  │   3     │  │   4     │  │   N     │      │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
│        │            │            │            │            │           │
│        └────────────┴─────┬──────┴────────────┴────────────┘           │
│                           │                                            │
│                           ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    WORKFLOW ORCHESTRATOR                        │  │
│   │              (Analogous to CUDA Stream Manager)                 │  │
│   │                                                                 │  │
│   │   Manages job dependencies, parallel execution, data flow      │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Mapping Summary

| Neural Network Concept | GitHub Actions Equivalent |
|------------------------|---------------------------|
| Forward pass | Workflow run (sequential jobs) |
| Backward pass | Workflow with reversed job order |
| Layer computation | Job |
| Tensor operation | Step within job |
| Activation function | Step output transformation |
| Batch processing | Matrix strategy |
| Gradient accumulation | Artifact storage across runs |
| Checkpoint saving | Release creation |
| Model loading | Checkout with LFS |
| Inference endpoint | workflow_dispatch trigger |
| Training loop | Scheduled workflow + recursion |
