"""Demo / CLI: run the logit lens on an MLX model and print per-layer predictions.

Examples::

    # Real model (downloads weights on first run):
    python main.py --model mlx-community/gemma-4-12B-it-4bit \
        --prompt "The capital of France is"

    # Offline smoke test with a tiny random gemma4 model (no download):
    python main.py --tiny --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
from dotenv import load_dotenv

from logit_lens import (
    format_logit_lens,
    format_logit_lens_grid,
    make_hooked_model,
)

load_dotenv()


def _load_real(model_name: str):
    from mlx_vlm import load

    model, processor = load(model_name)
    return model, processor


def _tiny_model():
    from mlx_lm.models.gemma4_text import Model, ModelArgs

    args = ModelArgs(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_key_value_heads=1,
        num_kv_shared_layers=1,
        vocab_size=256,
        vocab_size_per_layer_input=256,
        hidden_size_per_layer_input=16,
        tie_word_embeddings=True,
        final_logit_softcapping=30.0,
        sliding_window=16,
        sliding_window_pattern=2,
        max_position_embeddings=256,
        use_double_wide_mlp=False,
    )
    model = Model(args)
    mx.eval(model.parameters())

    class _Tok:
        def decode(self, ids, **kwargs):
            return "".join(f"<{i}>" for i in ids)

        def encode(self, text, **kwargs):
            return [(ord(c) % 250) + 1 for c in text[:8]]

    return model, _Tok()


def run_lens(model, tokenizer, prompt: str, top_k: int, pos: int, grid: bool):
    hooked = make_hooked_model(model, tokenizer)
    logits, cache = hooked.run_with_cache(prompt)
    print(f"Prompt: {prompt!r}")
    if grid:
        input_ids = hooked._encode(prompt)
        grid_result = cache.logit_lens_per_token(input_ids, final_logits=logits)
        print(format_logit_lens_grid(grid_result, tokenizer))
    else:
        lens_result = cache.logit_lens(pos=pos, top_k=top_k, final_logits=logits)
        print(format_logit_lens(lens_result, tokenizer, show_tokens=min(5, top_k)))
    hooked.restore()
    return hooked


def main() -> None:
    parser = argparse.ArgumentParser(description="Logit lens for MLX models.")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-12B-it-4bit",
        help="MLX model path or HF repo (default: mlx-community/gemma-4-12B-it-4bit)",
    )
    parser.add_argument(
        "--prompt", default="The capital of France is", help="Prompt text"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top tokens per layer")
    parser.add_argument(
        "--pos",
        type=int,
        default=-1,
        help="Sequence position to inspect (-1 = last token)",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a tiny random gemma4 model for an offline smoke test",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Show per-token logit lens grid (all positions x all layers)",
    )
    args = parser.parse_args()

    try:
        if args.tiny:
            model, tokenizer = _tiny_model()
        else:
            model, tokenizer = _load_real(args.model)
    except Exception as exc:  # pragma: no cover - network / auth failures
        print(f"Failed to load model {args.model!r}: {exc}", file=sys.stderr)
        print(
            "Tip: pass --tiny for an offline smoke test, or --model <mlx-repo>.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_lens(model, tokenizer, args.prompt, args.top_k, args.pos, args.grid)


if __name__ == "__main__":
    main()
