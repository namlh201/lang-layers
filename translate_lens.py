"""Run few-shot translation tasks from FLORES+ / WMT24++ and extract per-token
logit lens detail for each translation.

For each sample:
1. Build a few-shot prompt from dataset examples.
2. Generate the model's translation.
3. Run a cached forward pass over prompt + generated translation.
4. Output the logit lens at every layer for every generated token position.

Usage::

    # FLORES+ French → Chinese, 3-shot, 5 samples (mlx_lm model)
    python translate_lens.py --dataset flores \\
        --src fra_Latn --tgt cmn_Hans \\
        --n-shot 3 --n-samples 5 \\
        --model mlx-community/Llama-3.2-1B-Instruct-4bit

    # FLORES+ with a VLM model (gemma4)
    python translate_lens.py --dataset flores \\
        --src fra_Latn --tgt cmn_Hans \\
        --model mlx-community/gemma-4-12B-4bit

    # WMT24++ English → Chinese
    python translate_lens.py --dataset wmt24pp \\
        --tgt zh_CN --n-shot 3 --n-samples 5 \\
        --model mlx-community/Llama-3.2-1B-Instruct-4bit

    # Also output per-token grid (layers × positions)
    python translate_lens.py ... --grid

FLORES+ uses ISO 639-3 config names (e.g. ``cmn_Hans`` for Mandarin Simplified,
``fra_Latn`` for French).  Common aliases are mapped automatically::

    zho_Hans → cmn_Hans     zho_Hant → cmn_Hant
    zh       → cmn_Hans     fra      → fra_Latn
    eng      → eng_Latn     deu      → deu_Latn
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from logit_lens import (
    format_logit_lens,
    format_logit_lens_grid,
    make_hooked_model,
)
from utils.load import (
    TranslationSample,
    detect_gpu,
    load_flores,
    load_model,
    load_wmt24pp,
)
from utils.generation import build_few_shot_prompt, generate_translation

load_dotenv()


# --------------------------------------------------------------------------- #
#  Lens extraction
# --------------------------------------------------------------------------- #


@dataclass
class TokenLensDetail:
    """Per-token logit lens detail for one generated translation token."""

    token: str
    token_id: int
    layer_predictions: list[dict] = field(default_factory=list)


@dataclass
class SampleLensResult:
    """Full lens result for one translation sample."""

    sample: TranslationSample
    prompt: str
    generated: str
    # Per-token lens at the position predicting each generated token
    token_details: list[TokenLensDetail] = field(default_factory=list)
    # Single-position lens at the very last prompt position
    last_pos_lens: str = ""
    # Per-token grid (all positions × all layers)
    grid_text: str = ""


def run_lens_on_sample(
    hooked,
    model,
    tokenizer,
    sample: TranslationSample,
    few_shot: list[TranslationSample],
    max_gen_tokens: int,
    top_k: int,
    do_grid: bool,
    is_vlm: bool = False,
    vllm_wrapper=None,
    backend_kind: str = "mlx-lm",
) -> SampleLensResult:
    """Run the full lens extraction pipeline on one sample."""

    # 1. Build prompt
    prompt = build_few_shot_prompt(few_shot, sample.source)

    # 2. Generate translation
    generated = generate_translation(
        model, tokenizer, prompt,
        max_tokens=max_gen_tokens,
        is_vlm=is_vlm,
        vllm_wrapper=vllm_wrapper,
        backend_kind=backend_kind,
    )

    # 3. Run cached forward pass over prompt + generated translation
    full_text = prompt + generated
    logits, cache = hooked.run_with_cache(full_text)

    # 4. Single-position lens at the last prompt token (predicts first gen token)
    prompt_ids = hooked._encode(prompt)
    prompt_len = prompt_ids.shape[-1]
    full_ids = hooked._encode(full_text)

    # Use the unwrapped tokenizer for decode (handles VLM processors)
    raw_tok = hooked._tokenizer

    lens_result = cache.logit_lens(
        pos=prompt_len - 1,
        top_k=top_k,
        final_logits=logits,
    )
    last_pos_lens = format_logit_lens(lens_result, tokenizer, show_tokens=min(5, top_k))

    # 5. Per-token lens at each generated token position
    token_details: list[TokenLensDetail] = []
    gen_ids = full_ids[0, prompt_len:].tolist() if full_ids.shape[-1] > prompt_len else []

    for offset, tid in enumerate(gen_ids):
        pos = prompt_len + offset - 1  # position that predicts this token
        if pos < 0:
            continue
        tok_str = raw_tok.decode([tid]) if raw_tok else str(tid)

        layer_preds: list[dict] = []

        # Re-run lens at this position
        pos_lens = cache.logit_lens(
            pos=pos,
            top_k=top_k,
            final_logits=logits,
        )
        for row in pos_lens.rows:
            if row.top_tokens:
                top_tok = row.top_tokens[0]
                layer_preds.append({
                    "layer": row.label,
                    "top1": top_tok[1],
                    "prob": round(top_tok[3], 4),
                    "top5": [
                        {"token": t[1], "prob": round(t[3], 4)}
                        for t in row.top_tokens[:5]
                    ],
                })

        token_details.append(TokenLensDetail(
            token=tok_str,
            token_id=tid,
            layer_predictions=layer_preds,
        ))

    # 6. Optional per-token grid
    grid_text = ""
    if do_grid:
        grid_result = cache.logit_lens_per_token(
            full_ids, final_logits=logits, top_k=1
        )
        grid_text = format_logit_lens_grid(grid_result, tokenizer)

    hooked.remove_hooks()

    return SampleLensResult(
        sample=sample,
        prompt=prompt,
        generated=generated,
        token_details=token_details,
        last_pos_lens=last_pos_lens,
        grid_text=grid_text,
    )


# --------------------------------------------------------------------------- #
#  Output formatting
# --------------------------------------------------------------------------- #


def format_token_detail(detail: TokenLensDetail, top_k: int) -> str:
    """Format per-token lens detail as a readable table."""
    lines: list[str] = []
    lines.append(f"  Token: {detail.token!r} (id={detail.token_id})")
    lines.append(f"  {'Layer':<12} {'top1':<16} {'p(top1)':>8}  top-{top_k}")
    lines.append("  " + "-" * 70)

    for pred in detail.layer_predictions:
        top1 = pred["top1"]
        prob = pred["prob"]
        topk_strs = [
            f'{t["token"][:12]:<12}{t["prob"]:6.1%}'
            for t in pred[f"top{top_k}"]
        ]
        lines.append(
            f"  {pred['layer']:<12} {top1[:14]:<16} {prob:>8.1%}  "
            + "  ".join(topk_strs)
        )
    lines.append("")
    return "\n".join(lines)


def format_sample_result(result: SampleLensResult, idx: int, top_k: int) -> str:
    """Format the full result for one sample."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"SAMPLE {idx}")
    lines.append("=" * 80)
    lines.append(f"Source ({result.sample.source_lang}): {result.sample.source}")
    lines.append(f"Reference ({result.sample.target_lang}): {result.sample.target}")
    lines.append(f"Generated: {result.generated}")
    lines.append("")
    lines.append("--- Last-position lens (first generated token) ---")
    lines.append(result.last_pos_lens)
    lines.append("")
    lines.append("--- Per-token lens detail ---")
    for i, detail in enumerate(result.token_details):
        lines.append(f"[Token {i}]")
        lines.append(format_token_detail(detail, top_k))
    return "\n".join(lines)


def save_json(result: SampleLensResult, path: Path) -> None:
    """Save structured lens result as JSON."""
    data = {
        "sample_id": result.sample.sample_id,
        "source_lang": result.sample.source_lang,
        "target_lang": result.sample.target_lang,
        "source": result.sample.source,
        "reference": result.sample.target,
        "generated": result.generated,
        "prompt": result.prompt,
        "tokens": [
            {
                "token": d.token,
                "token_id": d.token_id,
                "layers": d.layer_predictions,
            }
            for d in result.token_details
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translation + logit lens extraction from FLORES+/WMT24++.",
    )
    parser.add_argument(
        "--dataset",
        choices=["flores", "wmt24pp"],
        required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--src",
        default=None,
        help="Source language (FLORES: e.g. fra_Latn; WMT24pp: always en)",
    )
    parser.add_argument(
        "--tgt",
        required=True,
        help="Target language (FLORES: e.g. zho_Hans; WMT24pp: e.g. zh_CN)",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="MLX model path or HF repo",
    )
    parser.add_argument(
        "--backend",
        choices=["lm", "vlm"],
        default=None,
        help="Force backend (auto-detected by default)",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=True,
        help="Use vllm-mlx (Apple) or vllm (NVIDIA) for generation "
        "(default: True)",
    )
    parser.add_argument(
        "--no-vllm",
        dest="use_vllm",
        action="store_false",
        help="Disable vllm/vllm-mlx, use mlx_lm/mlx_vlm or "
        "transformers directly",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mlx", "mps", "cpu"],
        help="Force device (auto-detected by default)",
    )
    parser.add_argument("--n-shot", type=int, default=3, help="Few-shot examples")
    parser.add_argument("--n-samples", type=int, default=5, help="Test samples")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max gen tokens")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k tokens per layer")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also output per-token grid (layers × positions)",
    )
    parser.add_argument(
        "--output-dir",
        default="lens_output",
        help="Directory for output files",
    )
    args = parser.parse_args()

    # --- Load dataset ---
    print(f"Loading {args.dataset} dataset...", file=sys.stderr)
    if args.dataset == "flores":
        src_lang = args.src or "eng_Latn"
        tgt_lang = args.tgt
        few_shot, test = load_flores(
            src_lang, tgt_lang, args.n_shot, args.n_samples
        )
    else:
        # WMT24pp: always en→X
        src_lang = "en"
        tgt_lang = args.tgt
        few_shot, test = load_wmt24pp(tgt_lang, args.n_shot, args.n_samples)

    print(
        f"  {len(few_shot)} few-shot examples, {len(test)} test samples",
        file=sys.stderr,
    )

    # --- Load model ---
    print(f"Loading model {args.model}...", file=sys.stderr)
    gpu = args.device or detect_gpu()
    print(f"  Detected GPU: {gpu}", file=sys.stderr)
    model, tokenizer, backend_kind, vllm_wrapper = load_model(
        args.model, args.backend, use_vllm=args.use_vllm, device=gpu
    )
    is_vlm = backend_kind in ("mlx-vlm", "vlm")
    gen_backend = (
        vllm_wrapper.__class__.__module__.split(".")[0]
        if vllm_wrapper is not None
        else backend_kind
    )
    print(f"  Backend: {gen_backend} ({backend_kind})", file=sys.stderr)

    hooked = make_hooked_model(model, tokenizer)

    # --- Output directory ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.dataset}_{src_lang}_{tgt_lang}"
    all_text_path = out_dir / f"lens_{tag}.txt"

    with open(all_text_path, "w") as all_f:
        for i, sample in enumerate(test):
            print(
                f"\n[{i + 1}/{len(test)}] Processing sample {sample.sample_id}...",
                file=sys.stderr,
            )
            print(f"  Source: {sample.source[:80]}...", file=sys.stderr)

            result = run_lens_on_sample(
                hooked,
                model,
                tokenizer,
                sample,
                few_shot,
                max_gen_tokens=args.max_tokens,
                top_k=args.top_k,
                do_grid=args.grid,
                is_vlm=is_vlm,
                vllm_wrapper=vllm_wrapper,
                backend_kind=backend_kind,
            )

            # Print and save
            text = format_sample_result(result, i)
            # print(text)
            all_f.write(text)
            all_f.write("\n\n")

            # Save JSON
            json_path = out_dir / f"lens_{tag}_sample{i}.json"
            save_json(result, json_path)

            # Save grid if requested
            if args.grid and result.grid_text:
                grid_path = out_dir / f"lens_{tag}_sample{i}_grid.txt"
                grid_path.write_text(result.grid_text)

    print(f"\nAll results saved to {out_dir}/", file=sys.stderr)
    print(f"Combined text: {all_text_path}", file=sys.stderr)

    hooked.restore()


if __name__ == "__main__":
    main()
