# lang-layers

Logit lens **hooks** for MLX models, implemented in the
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) style.

TransformerLens is PyTorch-based and cannot run MLX-quantized checkpoints such
as `mlx-community/gemma-4-12B-it-4bit`. This project ports TransformerLens's
logit-lens technique — its hook points, `run_with_cache`, and the
`apply_ln_to_stack(..., recompute_ln=True)` lens — to models loaded with
[`mlx_lm`](https://github.com/ml-explore/mlx-lm), so you can run the lens on
Apple Silicon without leaving MLX.

## How it works

`HookedMLXModel` wraps any `mlx_lm` model and instruments every decoder layer
with TransformerLens-style forward hooks on the residual stream:

| Hook point                       | Meaning                                  |
| -------------------------------- | ---------------------------------------- |
| `blocks.{i}.hook_resid_pre`      | Residual stream entering block `i`       |
| `blocks.{i}.hook_resid_post`     | Residual stream leaving block `i`        |

`run_with_cache(...)` runs a forward pass and returns the real logits plus an
`ActivationCache` of every residual. The **logit lens** then applies the
model's final layer-norm (with recomputed statistics) + unembed (+ logit
soft-cap, for Gemma) to each cached residual — i.e. "what would the model
predict if it stopped here?" — exactly mirroring TransformerLens's
`ActivationCache.apply_ln_to_stack(residual, layer=n_layers, recompute_ln=True)`
followed by the unembed.

A hook may return a replacement activation that flows forward (ablation /
patching), just like TransformerLens `add_hook`.

## Usage

```python
from mlx_vlm import load
from logit_lens import HookedMLXModel, format_logit_lens

model, processor = load("mlx-community/gemma-4-12B-it-4bit")
hooked = HookedMLXModel(model, processor)

logits, cache = hooked.run_with_cache("The capital of France is")
result = cache.logit_lens(pos=-1, top_k=10)
print(format_logit_lens(result, processor))

hooked.restore()
```

Register your own forward hook (e.g. zero out an early layer's residual):

```python
def ablate(activation, hook):
    return mx.zeros_like(activation) if hook.layer == 0 else None

hooked.add_hook("blocks.0.hook_resid_pre", ablate)
hooked.run_with_cache("...")
```

## CLI

```bash
# Real model (downloads weights on first run):
python main.py --model mlx-community/gemma-4-12B-it-4bit \
    --prompt "The capital of France is"

# Offline smoke test with a tiny random gemma4 model:
python main.py --tiny
```

## Tests

```bash
python test_logit_lens.py
```

The tests instantiate the *real* `mlx_lm` gemma4 architecture at tiny scale
(no weights downloaded) and assert the core invariant:
`lens(resid_post[-1]) == model output logits`.

## Notes

- The example model in the prompt, `mlx-community/gemma-4-12B-4bit`, is the
  instruction-tuned `mlx-community/gemma-4-12B-it-4bit` checkpoint; pass any MLX
  repo to `--model` / `load()`.
- Gated models (Gemma) require `HF_TOKEN` in your environment.
- Models are loaded via `mlx_vlm` which returns a `(model, processor)` tuple.
  `HookedMLXModel` automatically detects VLM models and traverses the
  `language_model` submodule to find the text decoder's residual stream, final
  norm, and unembed.
