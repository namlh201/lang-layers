# Manual: lang-layers

Logit lens hooks for MLX and PyTorch models, in the TransformerLens style.
Run translation tasks across FLORES+ and WMT24++ datasets, extract
per-token logit lens detail, and analyze layer-by-layer language specificity.

---

## 1. GPU Auto-Detection

All scripts auto-detect the best available GPU on startup:

| Priority | Backend | Hardware | Libraries used |
|----------|---------|----------|----------------|
| 1 | `cuda` | NVIDIA GPU | `vllm` or `torch` + `transformers` |
| 2 | `mlx` | Apple Silicon (M1-M5) | `vllm-mlx` or `mlx-lm` / `mlx-vlm` |
| 3 | `mps` | Apple Silicon via PyTorch | `torch` + `transformers` |
| 4 | `cpu` | CPU fallback | `torch` + `transformers` |

Override with `--device cuda|mlx|mps|cpu`.

### Generation acceleration

| Device | Fast generation | Fallback generation |
|--------|----------------|---------------------|
| `cuda` | `vllm` | `transformers` (`model.generate`) |
| `mlx` | `vllm-mlx` | `mlx-lm` / `mlx-vlm` |
| `mps` | `transformers` | -- |
| `cpu` | `transformers` | -- |

Disable accelerated generation with `--no-vllm`.

### Installation

```bash
# Apple Silicon (MLX)
pip install -e ".[mlx,dev]"

# NVIDIA CUDA
pip install -e ".[cuda,dev]"

# Both
pip install -e ".[mlx,cuda,dev]"
```

---

## 2. Scripts

### 2.1 `main.py` -- Logit lens demo

Runs the logit lens on a single prompt and prints per-layer predictions.

```
python main.py [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/gemma-4-12B-it-4bit` | Model path or HF repo |
| `--prompt` | `The capital of France is` | Prompt text |
| `--top-k` | `10` | Top tokens per layer |
| `--pos` | `-1` | Sequence position to inspect (-1 = last) |
| `--tiny` | off | Use a tiny random gemma4 model (offline, no download) |
| `--grid` | off | Show per-token grid (all positions x all layers) |

**Examples:**

```bash
# Offline smoke test (no weights downloaded)
python main.py --tiny --prompt "The capital of France is"

# Real model, single-position lens
python main.py --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --prompt "The capital of France is" --top-k 5

# Per-token grid
python main.py --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --prompt "The capital of France is" --grid
```

---

### 2.2 `generate.py` -- Text generation

Load a model and generate text. Auto-detects GPU and chooses the best backend.

```
python generate.py --prompt TEXT [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/gemma-4-12B-it-4bit` | Model path or HF repo |
| `--prompt` | (required) | Prompt text |
| `--image` | `None` | Image path or URL (VLM only) |
| `--max-tokens` | `512` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--verbose` | off | Show generation progress |
| `--backend` | auto | Force MLX sub-backend: `lm` or `vlm` |
| `--use-vllm` / `--no-vllm` | enabled | Use vllm/vllm-mlx for generation |
| `--device` | auto | Force device: `cuda`, `mlx`, `mps`, `cpu` |

**Examples:**

```bash
# Apple Silicon (auto-detected)
python generate.py --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --prompt "The capital of France is"

# NVIDIA CUDA (auto-detected)
python generate.py --model meta-llama/Llama-3.2-3B-Instruct \
    --prompt "The capital of France is"

# Force CPU
python generate.py --model meta-llama/Llama-3.2-3B-Instruct \
    --prompt "Hello" --device cpu --no-vllm

# VLM with image (MLX only)
python generate.py --model mlx-community/gemma-4-12B-it-4bit \
    --prompt "Describe this image" --image photo.jpg
```

---

### 2.3 `translate_lens.py` -- Translation + logit lens extraction

Run few-shot translation tasks from FLORES+ or WMT24++, generate a
translation, then run a cached forward pass and extract per-token logit
lens detail for every generated token.

```
python translate_lens.py --dataset DATASET --tgt LANG [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | `flores` or `wmt24pp` |
| `--src` | `eng_Latn` (FLORES) / `en` (WMT24pp) | Source language |
| `--tgt` | (required) | Target language |
| `--model` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | Model path |
| `--backend` | auto | Force MLX sub-backend: `lm` or `vlm` |
| `--n-shot` | `3` | Few-shot examples |
| `--n-samples` | `5` | Test samples |
| `--max-tokens` | `10` | Max generation tokens |
| `--top-k` | `10` | Top-k tokens per layer |
| `--grid` | off | Also output per-token grid |
| `--output-dir` | `lens_output` | Output directory |
| `--use-vllm` / `--no-vllm` | enabled | Use vllm/vllm-mlx for generation |
| `--device` | auto | Force device: `cuda`, `mlx`, `mps`, `cpu` |

**Output files** (in `--output-dir`):

| File | Content |
|------|---------|
| `lens_{tag}.txt` | Combined text output (all samples) |
| `lens_{tag}_sample{i}.json` | Structured JSON per sample |
| `lens_{tag}_sample{i}_grid.txt` | Per-token grid (if `--grid`) |

Where `tag = {dataset}_{src}_{tgt}` (e.g. `wmt24pp_en_de_DE`).

**JSON structure:**

```json
{
  "sample_id": "3",
  "source_lang": "en",
  "target_lang": "de_DE",
  "source": "Original text...",
  "reference": "Reference translation...",
  "generated": "Model translation...",
  "prompt": "Full few-shot prompt...",
  "tokens": [
    {
      "token": "Die",
      "token_id": 1234,
      "layers": [
        {
          "layer": "embed",
          "top1": "Die",
          "prob": 0.54,
          "top5": [
            {"token": "Die", "prob": 0.54},
            {"token": "The", "prob": 0.20},
            ...
          ]
        },
        {"layer": "layer 0", ...},
        ...
      ]
    },
    ...
  ]
}
```

**Examples:**

```bash
# FLORES+ English -> Chinese, 3-shot, 5 samples (Apple Silicon)
python translate_lens.py --dataset flores \
    --src eng_Latn --tgt cmn_Hans \
    --n-shot 3 --n-samples 5 \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit

# WMT24++ English -> German (NVIDIA CUDA)
python translate_lens.py --dataset wmt24pp --tgt de_DE \
    --n-shot 3 --n-samples 3 \
    --model meta-llama/Llama-3.2-3B-Instruct

# FLORES+ with VLM model (gemma4)
python translate_lens.py --dataset flores \
    --src fra_Latn --tgt cmn_Hans \
    --model mlx-community/gemma-4-12B-4bit

# With per-token grid
python translate_lens.py --dataset wmt24pp --tgt zh_CN \
    --grid --max-tokens 10
```

**FLORES+ language codes:** Uses ISO 639-3 (e.g. `cmn_Hans` for Mandarin
Simplified, `fra_Latn` for French). Common aliases are auto-mapped:
`zho_Hans` -> `cmn_Hans`, `zh` -> `cmn_Hans`, `fra` -> `fra_Latn`, etc.

**WMT24++ language codes:** Uses locale suffixes (e.g. `zh_CN`, `de_DE`,
`fr_FR`, `ar_EG`). Source is always `en`.

---

### 2.4 `translate_all.py` -- Batch all language pairs

Run the translation + logit lens pipeline across ALL language pairs in
FLORES+ and/or WMT24++. Loads the model once and iterates all pairs.

```
python translate_all.py --dataset DATASET [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | `flores`, `wmt24pp`, or `both` |
| `--src` | `eng_Latn` | Source language(s), comma-separated (FLORES+ only) |
| `--tgt-filter` | `None` | Comma-separated targets to restrict to |
| `--model` | `mlx-community/gemma-4-12B-4bit` | Model path |
| `--backend` | auto | Force MLX sub-backend: `lm` or `vlm` |
| `--n-shot` | `3` | Few-shot examples |
| `--n-samples` | `3` | Test samples per pair |
| `--max-tokens` | `10` | Max generation tokens |
| `--top-k` | `10` | Top-k tokens per layer |
| `--grid` | off | Also output per-token grid |
| `--limit` | `None` | Limit number of pairs (for testing) |
| `--output-dir` | `lens_output_all` | Root output directory |
| `--resume` | off | Skip pairs that already have output |
| `--use-vllm` / `--no-vllm` | enabled | Use vllm/vllm-mlx for generation |
| `--device` | auto | Force device: `cuda`, `mlx`, `mps`, `cpu` |

**Output structure:**

```
lens_output_all/
  summary.json                         # Global summary
  wmt24pp_en_de_DE/
    lens_wmt24pp_en_de_DE.txt          # All samples text
    lens_wmt24pp_en_de_DE_sample0.json # Per-sample JSON
    lens_wmt24pp_en_de_DE_sample1.json
    lens_wmt24pp_en_de_DE_sample2.json
  wmt24pp_en_zh_CN/
    ...
  flores_eng_Latn_cmn_Hans/
    ...
```

**Examples:**

```bash
# All WMT24++ pairs (55 en->X), 3 samples each
python translate_all.py --dataset wmt24pp --n-samples 3

# All FLORES+ pairs from English (230 targets)
python translate_all.py --dataset flores --src eng_Latn --n-samples 3

# Both datasets
python translate_all.py --dataset both --n-samples 1

# FLORES+ from multiple source languages
python translate_all.py --dataset flores \
    --src eng_Latn,fra_Latn,deu_Latn --n-samples 2

# Specific targets only (fast test)
python translate_all.py --dataset flores --src eng_Latn \
    --tgt-filter cmn_Hans,jpn_Jpan,kor_Hang

# Limit to 5 pairs for testing
python translate_all.py --dataset both --limit 5

# Resume interrupted run
python translate_all.py --dataset both --resume

# NVIDIA CUDA with vLLM
python translate_all.py --dataset wmt24pp \
    --model meta-llama/Llama-3.2-3B-Instruct --device cuda

# Apple Silicon without vllm-mlx
python translate_all.py --dataset flores --no-vllm
```

**Summary JSON:**

```json
{
  "total_pairs": 285,
  "completed": 285,
  "pairs": [
    {
      "dataset": "wmt24pp",
      "src_lang": "en",
      "tgt_lang": "de_DE",
      "src_name": "English",
      "tgt_name": "Deutsch",
      "n_samples": 3,
      "n_tokens_total": 30,
      "elapsed_s": 12.5,
      "error": null,
      "samples": [...]
    },
    ...
  ]
}
```

---

### 2.5 `analyze_all_lens.py` -- Layer analysis

Analyze all lens output JSONs in `lens_output_all/` to classify layers as
language-specific or language-neutral. Detects the dominant script (Arabic,
Cyrillic, CJK, Hangul, Latin, etc.) at each layer for each language pair.

```
python analyze_all_lens.py
```

**Output:**

1. **Overall per-layer summary**: Target%, Latin%, Other% with
   classification (TARGET-SPECIFIC, TRANSITIONAL, SOURCE, NOISE)
2. **Per-language-pair analysis**: Script breakdown per layer for each
   target language
3. **Dominant language per layer**: Aggregated across all non-Latin targets
4. **Phase classification**: Per-layer phase (TARGET DOMINANT, TARGET
   EMERGING, SOURCE DOMINANT, NOISE)

**Key finding (16-layer Llama-3.2-1B):**

| Phase | Layers | Target% | Classification |
|-------|--------|---------|----------------|
| Embedding | 0-1 | ~77% | Target-dominant (reflects few-shot context) |
| Middle | 2-13 | ~42% | Language-neutral (Latin/English-centric) |
| Final | 14-15 | ~78% | Target-specific (language selection) |

---

### 2.6 `analyze_lens.py` -- Single-file lens parser

Parses a `lens.txt` grid file (from `main.py --grid`) and classifies layers.

```
python analyze_lens.py lens.txt
```

---

## 3. Core library: `logit_lens.py`

### Classes

#### `HookedMLXModel(model, tokenizer)`

Wraps an MLX model (`mlx.nn.Module`) with TransformerLens-style hooks.
Patches `__call__` on each decoder-layer class to capture
`hook_resid_pre` / `hook_resid_post`.

#### `HookedTorchModel(model, tokenizer)`

Wraps a PyTorch model (`torch.nn.Module`) with the same interface.
Uses `register_forward_pre_hook` / `register_forward_hook` on each decoder
layer. Works with any HuggingFace `transformers` model on CUDA, MPS, or CPU.

#### `make_hooked_model(model, tokenizer)`

Factory function: detects whether the model is MLX or PyTorch and returns
the appropriate hooked wrapper. **Use this in all new code.**

#### `ActivationCache(dict)`

Dict of cached activations keyed by hook names (`blocks.0.hook_resid_pre`,
etc.). Holds a back-reference to the hooked model for lens computation.

Methods: `logit_lens(pos, top_k, final_logits)`,
`logit_lens_per_token(input_ids, final_logits, top_k)`,
`apply_final_ln(residual)`, `lens_logits(residual)`.

### Shared interface

Both `HookedMLXModel` and `HookedTorchModel` implement:

| Method | Description |
|--------|-------------|
| `run_with_cache(input, names)` | Forward pass, returns `(logits, cache)` |
| `lens_logits(residual)` | Apply final-norm + unembed + softcap |
| `logit_lens(cache, pos, top_k)` | Per-layer top-k predictions at one position |
| `logit_lens_per_token(cache, input_ids, top_k)` | Per-token grid |
| `remove_hooks()` | Clean up hooks |
| `restore()` | Revert any class-level patches (MLX only) |
| `.layers` | List of decoder layers |
| `.n_layers` | Number of layers |
| `._tokenizer` | Unwrapped tokenizer |
| `._encode(text)` | Encode text to token ids |

### Data classes

- `LayerLens` -- one layer's lens prediction at a position
- `LogitLensResult` -- all layers at one position
- `LayerTokenLens` -- one layer's top-k predictions at every position
- `LogitLensGrid` -- all layers x all positions

### Format functions

- `format_logit_lens(result, tokenizer, show_tokens)` -- readable table
- `format_logit_lens_grid(grid, tokenizer, col_width, show_probs)` -- grid table

---

## 4. Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (for gated models) | HuggingFace access token |
| `HF_HOME` | No | HuggingFace cache directory |

Store in `.env` file:

```
HF_TOKEN=hf_your_token_here
```

---

## 5. Datasets

### FLORES+ (`openlanguagedata/flores_plus`)

- 231 configs (230 languages + `default`)
- Config names use ISO 639-3: `cmn_Hans`, `fra_Latn`, `jpn_Jpan`, etc.
- `devtest` split used for evaluation
- Column: `text` (aligned by `id` across languages)

### WMT24++ (`google/wmt24pp`)

- 55 configs, all `en-{locale}` (e.g. `en-zh_CN`, `en-de_DE`)
- English -> X direction only
- `train` split used (first rows filtered by `is_bad_source`)
- Columns: `source`, `target`, `segment_id`, `is_bad_source`

---

## 6. Backend compatibility matrix

| Feature | MLX (`mlx-lm`) | MLX+`vllm-mlx` | CUDA (`transformers`) | CUDA+`vllm` |
|---------|:-:|:-:|:-:|:-:|
| Text generation | Yes | Yes (faster) | Yes | Yes (faster) |
| VLM generation | Yes | Yes (faster) | No | No |
| Logit lens | Yes | Yes | Yes | Yes |
| Per-token grid | Yes | Yes | Yes | Yes |
| `--resume` | Yes | Yes | Yes | Yes |
| Few-shot prompting | Yes | Yes | Yes | Yes |

> **Note:** vLLM (NVIDIA) loads the HF model internally. The logit lens
> forward pass uses this model via `HookedTorchModel`. Generation uses
> vLLM's optimized `llm.generate()`.

---

## 7. Tips

- **Faster runs**: Use `--max-tokens 10` (default) for quick lens extraction.
- **Resume**: Use `--resume` to skip completed pairs in `translate_all.py`.
- **Testing**: Use `--limit 5` to process only 5 pairs.
- **Specific languages**: Use `--tgt-filter cmn_Hans,jpn_Jpan,kor_Hang`.
- **Grid output**: Use `--grid` for a full layers x positions visualization.
- **CPU-only**: Use `--device cpu --no-vllm` to force CPU inference.
- **Tiny model test**: `python main.py --tiny` runs offline with a random
  4-layer gemma4 model (no weight download).
