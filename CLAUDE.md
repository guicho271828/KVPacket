# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KV Packet is a research framework for reusing precomputed KV caches across documents in multi-document RAG settings without recomputation. Each document's KV cache is wrapped with trainable soft-token vectors (header + trailer) that absorb boundary artifacts from naive concatenation.

Paper: arXiv:2604.13226

### Problem context

In RAG, documents are often cached independently but naively concatenating their KV caches causes catastrophic performance degradation due to two issues:
1. **Positional dependency** — KV states encode absolute RoPE positions; shifted caches have wrong positional embeddings. Solved by re-rotating keys at inference (near-zero cost).
2. **Context dependency** — Each cache was computed without visibility of surrounding documents, so cross-document attention is missing. This causes attention sink artifacts at block boundaries and disrupted token distributions.

### How KV Packet solves it

Trainable header (H) and trailer (T) vectors (default: 8 tokens each, float32) are prepended/appended to each document's embeddings before generating its KV cache. The adapters absorb attention sink mass that would otherwise distort document-initial tokens, acting as smooth structural delimiters. At inference: load cached packets, re-rotate keys (RoPE shift), concatenate — no recomputation needed.

Training uses self-supervised KL distillation: the model's own full-attention output is the teacher, and only the small adapter tensors (N_h + N_t vectors × hidden_dim) receive gradients. Typical setup: 30 epochs, AdamW, linear decay, batch 64, lr 5×10⁻⁴, single A100.

### Relationship to baselines (implemented in this repo)

| Method | Approach | Overhead at inference |
|--------|----------|---------------------|
| **KV Packet** (ours) | Learned header/trailer adapters, zero recomputation | RoPE re-rotation only (~0 FLOPs) |
| **CacheBlend** | Selectively recomputes 5–18% of tokens per layer (HKVD tokens with highest KV deviation) | Partial forward pass, pipelined with loading |
| **EPIC** | Recomputes anchor tokens at document boundaries | Partial forward pass |
| **A3** | Selects tokens by real-time query-document attention scores | Partial forward pass |
| **SAM-KV** | Hierarchical compression for multi-context | Varies |
| **KVLink** (not in repo) | Trainable link tokens + base model fine-tuning (NeurIPS 2025) | Link token forward pass |

KV Packet's key differentiators: zero recomputation at inference (vs CacheBlend/EPIC/A3), and no base model fine-tuning (vs KVLink/BlockAttention). The "Universal (Mixture)" adapter trained on all datasets gives the best cross-domain generalization.

## Setup

Requires Python ≥ 3.11 with CUDA. Install with the PyTorch CUDA index:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu130
# or
pip install -r requirements.txt
```

Before running, update `model.model_path` in config files to point to your local model directory or HuggingFace identifier.

## Commands

### Train header/trailer adapters
```bash
python run_train_filler.py <config.json>
python run_train_filler.py packet_wrapper_config/llama_3_1_8b/mixture/
```

### Evaluate on benchmarks
```bash
python run_eval.py <config.json or directory> [--overwrite] [--debug]
```
Results go to `eval_results/<config_name>_result.json` next to each config file.

### Build packet from handcrafted tokens (ablation)
```bash
python run_build_packet.py <config.json>
```

## Architecture

### Core library (`kv_packet/`)

- **`packet_wrapper/`** — `PacketWrapper` class holds trainable header/trailer tensors (shape `[1, len, hidden_dim]`). `wrap()` prepends header and appends trailer to input embeddings. Saved/loaded as state dicts via `torch.save`/`torch.load`.

- **`cache/`** — KV cache storage and manipulation. `KVCache` wraps per-layer key-value pairs. Supports compression (via `kvpress` scorers), quantization, and RoPE re-rotation. `get_kv_caches()` runs a forward pass to produce caches from embeddings.

- **`cache_comb/`** — Cache combination methods (the actual evaluation strategies). Each method in `methods/` implements the `EvalCombFunc` protocol: takes model, tokenizer, pre-cached document KVs, and returns metrics (tp/fp/fn, ttft, flops). Registered in `CACHE_COMB_FUNC_DICT`. Methods: `kv_packet`, `cache_blend`, `epic`, `a3`, `sam_kv`, `no_recompute`, `rand_recompute`, `full_recompute`, `full_context`, `no_cache`, `sink`, `single_cache`.

- **`cache_comb/recompute_kv/`** — Model-specific KV re-rotation for position correction after cache concatenation. Separate implementations for Llama and Qwen3.

- **`dataset/`** — Dataset loaders producing `RetEvalEntry` iterators (preamble, documents, query, answer). Datasets: `biography`, `hotpot_qa`, `niah` (+ `musique` via eval configs). Template functions in `template.py` apply chat formatting.

- **`model/`** — Type alias `SupportedModel = LlamaForCausalLM | Qwen3ForCausalLM`. Adding a model requires implementing re-rotation in `cache_comb/recompute_kv/`.

- **`utils/`** — Config loading with `_default.json` inheritance (`broadcast_dict` merges defaults into overrides without overwriting), training loop helpers, generation cache, and token-level F1 metrics.

### Config system

Both train and eval configs use `_default.json` inheritance: a per-method override file only lists fields that differ from the default in the same directory. Fields in the override take precedence; missing fields are filled from the default.

### Training flow

1. Load model (frozen) and initialize `PacketWrapper` with random normal embeddings
2. Generate teacher signals (full-attention outputs) cached in `GenerationCache`
3. Train only header/trailer parameters via KL divergence (logits) or cross-entropy (tokens) against teacher
4. Uses custom 4D attention masks (`packet_4d_mask`) that enforce block-diagonal attention across documents
5. Checkpoints saved as `.epoch{N}` files; final result as `.pt`

### Evaluation flow

1. Load model + optional trained `PacketWrapper`
2. Generate per-document KV caches, optionally compress/quantize
3. Call the selected `cache_comb` method which concatenates/recomputes caches and generates answers
4. Compute token-level precision/recall/F1 against ground truth

### Adding a new cache combination method

1. Create a new file in `kv_packet/cache_comb/methods/` implementing the `EvalCombFunc` protocol (see `abc.py` for the signature and `no_recompute.py` for a minimal example).
2. Register it in `kv_packet/cache_comb/methods/__init__.py` by adding it to `CACHE_COMB_FUNC_DICT`.
3. Create evaluation configs in `eval_config/<model>/<dataset>/<method_name>.json` for each model/dataset combination. Use `_default.json` inheritance — only override the `cache_comb` block.
