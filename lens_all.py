"""Run logit lens on pre-generated translations.

Reads translation JSON files produced by ``translate_all.py``, loads the
model in HuggingFace (transformers) format or via nnsight's vLLM backend,
and runs the logit lens on each sample's prompt + generated translation.

Usage::

    # Run lens with HuggingFace backend (default)
    python lens_all.py --model google/gemma-4-12B --input-dir translations

    # Run lens with nnsight/vLLM backend
    python lens_all.py --model google/gemma-4-12B \\
        --input-dir translations --backend nnsight

    # Limit to specific pairs
    python lens_all.py --model google/gemma-4-12b \\
        --input-dir translations --limit 5

    # Resume (skip pairs that already have lens output)
    python lens_all.py --model google/gemma-4-12b \\
        --input-dir translations --resume

    # Also output per-token grid (layers x positions)
    python lens_all.py --model google/gemma-4-12b \\
        --input-dir translations --grid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from safetensors.torch import save_file

from dotenv import load_dotenv

from logit_lens import (
    format_logit_lens,
    format_logit_lens_grid,
    grid_to_lens_result,
    make_hooked_model,
    LogitLensGrid,
    LayerTokenLens,
    _decode_ids,
)
from utils.load import TranslationSample, load_hf_model, load_nnsight_model
from translate_lens import (
    TokenLensDetail,
    SampleLensResult,
    format_sample_result,
    save_json,
)

load_dotenv()


# --------------------------------------------------------------------------- #
#  Logit lens on a pre-generated translation
# --------------------------------------------------------------------------- #


def run_lens_on_batch(
    hooked,
    tokenizer,
    samples: list[dict],
    top_k: int,
    do_grid: bool,
    src_lang: str = "",
    tgt_lang: str = "",
    generation_only: bool = True,
    save_logits: bool = False,
) -> tuple[list[SampleLensResult], list[Any]]:
    """Run the logit lens on a batch of pre-generated translations.

    Encodes all ``prompt + translation`` texts with padding, runs one
    forward pass, and computes the logit lens for every layer in a
    single batched pass.  Padded positions are excluded.

    When ``generation_only=True`` (default), lens results are only
    saved for generated token positions.  When ``False``, lens results
    are saved for all token positions (prompt + generation).
    """
    texts = [s["prompt"] + s["translation"] for s in samples]
    if save_logits:
        grids_and_logits = hooked.run_lens_batch(
            texts, top_k=top_k, return_logits=True,
        )
        grids = [g for g, _ in grids_and_logits]
        logits_list = [l for _, l in grids_and_logits]
    else:
        grids = hooked.run_lens_batch(texts, top_k=top_k)
        logits_list = [None] * len(grids)

    raw_tok = tokenizer
    if raw_tok is not None and hasattr(raw_tok, "tokenizer"):
        raw_tok = raw_tok.tokenizer

    results: list[SampleLensResult] = []
    for i, (s, grid) in enumerate(zip(samples, grids)):
        prompt = s["prompt"]
        generated = s["translation"]

        prompt_ids = hooked._encode(prompt)
        prompt_len = prompt_ids.shape[-1]

        # Last-position lens (predicts first generated token)
        lens_result = grid_to_lens_result(
            grid, pos=prompt_len - 1, top_k=top_k, tokenizer=tokenizer,
        )
        last_pos_lens = format_logit_lens(
            lens_result, tokenizer, show_tokens=min(10, top_k),
        )

        # Per-token lens at each token position
        token_details: list[TokenLensDetail] = []
        if generation_only:
            token_start = prompt_len
        else:
            token_start = 0

        for token_idx in range(token_start, grid.n_positions):
            pos = token_idx - 1
            if pos < 0:
                continue

            tid = grid.input_ids[token_idx]
            tok_str = (
                raw_tok.decode([tid]) if raw_tok else str(tid)
            )

            layer_preds: list[dict] = []
            for layer in grid.layers:
                if layer.label == "final":
                    continue
                if layer.top_ids is None or pos >= layer.top_ids.shape[0]:
                    continue

                ids = layer.top_ids[pos].tolist()
                probs = (
                    layer.probs[pos].tolist()
                    if layer.probs is not None
                    else [0.0] * len(ids)
                )

                top1_str = (
                    layer.tokens[pos]
                    if pos < len(layer.tokens)
                    else str(ids[0])
                )

                topk_str: list[dict] = []
                for j in range(min(top_k, len(ids))):
                    t_str = (
                        raw_tok.decode([ids[j]])
                        if raw_tok
                        else str(ids[j])
                    )
                    topk_str.append(
                        {"token": t_str, "prob": round(probs[j], 4)}
                    )

                layer_preds.append({
                    "layer": layer.label,
                    "top1": top1_str,
                    "prob": round(probs[0], 4),
                    f"top{top_k}": topk_str,
                })

            token_details.append(TokenLensDetail(
                token=tok_str,
                token_id=tid,
                layer_predictions=layer_preds,
            ))

        grid_text = ""
        if do_grid:
            grid_text = format_logit_lens_grid(grid, tokenizer)

        sample = TranslationSample(
            source=s.get("source", ""),
            target=s.get("reference", ""),
            source_lang=src_lang,
            target_lang=tgt_lang,
            sample_id=s.get("sample_id", str(i)),
        )

        results.append(SampleLensResult(
            sample=sample,
            prompt=prompt,
            generated=generated,
            token_details=token_details,
            last_pos_lens=last_pos_lens,
            grid_text=grid_text,
        ))

    return results, logits_list


# --------------------------------------------------------------------------- #
#  Process one translation JSON file
# --------------------------------------------------------------------------- #


def run_lens_on_file(
    hooked,
    tokenizer,
    json_path: Path,
    out_dir: Path,
    top_k: int,
    do_grid: bool,
    batch_size: int = 4,
    resume: bool = False,
    generation_only: bool = True,
    save_logits: bool = False,
    logits_out_dir: Path | None = None,
) -> dict:
    """Read a translations JSON file and run the logit lens on each sample.

    Samples are processed in batches of ``batch_size`` for faster
    inference.  Each batch is padded, run through one forward pass,
    and lens-computed in a single batched pass.

    When ``resume=True``, samples whose output JSON already exists
    are skipped, and the text file is appended to (not truncated).
    """

    data = json.loads(json_path.read_text())
    dataset = data.get("dataset", "")
    src_lang = data.get("src_lang", "")
    tgt_lang = data.get("tgt_lang", "")
    samples = data.get("samples", [])
    n_samples = len(samples)

    tag = json_path.stem
    if tag.startswith("translations_"):
        tag = tag[len("translations_"):]
    pair_dir = out_dir / tag
    pair_dir.mkdir(parents=True, exist_ok=True)

    logits_pair_dir = (
        logits_out_dir / tag if logits_out_dir else None
    )
    if logits_pair_dir is not None:
        logits_pair_dir.mkdir(parents=True, exist_ok=True)

    # Determine which samples already have output
    done_indices: set[int] = set()
    if resume:
        for idx in range(n_samples):
            lens_done = (
                pair_dir / f"lens_{tag}_sample{idx}.json"
            ).exists()
            logits_done = True
            if save_logits and logits_pair_dir is not None:
                logits_done = (
                    logits_pair_dir
                    / f"logits_{tag}_sample{idx}.safetensors"
                ).exists()
            if lens_done and logits_done:
                done_indices.add(idx)

    to_process = [
        (idx, s) for idx, s in enumerate(samples) if idx not in done_indices
    ]

    # Build summary for already-done samples (from existing JSONs)
    n_tokens_total = 0
    sample_summaries: list[dict] = []

    for idx in sorted(done_indices):
        s = samples[idx]
        json_out = pair_dir / f"lens_{tag}_sample{idx}.json"
        try:
            existing = json.loads(json_out.read_text())
            n_tokens = len(existing.get("tokens", []))
        except Exception:
            n_tokens = 0
        n_tokens_total += n_tokens
        sample_summaries.append({
            "sample_id": s.get("sample_id", str(idx)),
            "source": s.get("source", "")[:120],
            "reference": s.get("reference", "")[:120],
            "translation": s.get("translation", "")[:120],
            "n_tokens": n_tokens,
        })

    if not to_process:
        if done_indices:
            print(f"    All {n_samples} samples already done.",
                  file=sys.stderr)
        return {
            "dataset": dataset,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "n_samples": n_samples,
            "n_tokens_total": n_tokens_total,
            "samples": sample_summaries,
        }

    n_skipped = len(done_indices)
    if n_skipped:
        print(f"    Resuming: {n_skipped} done, "
              f"{len(to_process)} remaining", file=sys.stderr)

    all_text_path = pair_dir / f"lens_{tag}.txt"
    text_mode = "a" if (resume and done_indices) else "w"
    with open(all_text_path, text_mode) as all_f:
        for batch_start in range(0, len(to_process), batch_size):
            batch = to_process[batch_start:batch_start + batch_size]
            batch_samples = [s for _, s in batch]
            batch_indices = [idx for idx, _ in batch]

            print(
                f"    [{batch_indices[0] + 1}-{batch_indices[-1] + 1}/{n_samples}] "
                f"batch of {len(batch)}...",
                file=sys.stderr,
            )

            batch_results, batch_logits = run_lens_on_batch(
                hooked,
                tokenizer,
                batch_samples,
                top_k=top_k,
                do_grid=do_grid,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                generation_only=generation_only,
                save_logits=save_logits,
            )

            for i, result in enumerate(batch_results):
                global_idx = batch_indices[i]
                s = batch_samples[i]

                text = format_sample_result(result, global_idx, top_k)
                all_f.write(text)
                all_f.write("\n\n")

                json_path_out = (
                    pair_dir / f"lens_{tag}_sample{global_idx}.json"
                )
                save_json(result, json_path_out)

                if do_grid and result.grid_text:
                    grid_path = (
                        pair_dir
                        / f"lens_{tag}_sample{global_idx}_grid.txt"
                    )
                    grid_path.write_text(result.grid_text)

                if (
                    save_logits
                    and batch_logits[i] is not None
                    and logits_pair_dir is not None
                ):
                    logits_path = (
                        logits_pair_dir
                        / f"logits_{tag}_sample{global_idx}.safetensors"
                    )
                    lg = batch_logits[i]
                    metadata = {
                        "sample_id": str(
                            s.get("sample_id", str(global_idx))
                        ),
                        "n_layers": str(lg.shape[0]),
                        "seq_len": str(lg.shape[1]),
                        "vocab_size": str(lg.shape[2]),
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                    }
                    save_file(
                        {"logits": lg.contiguous()},
                        str(logits_path),
                        metadata=metadata,
                    )

                n_tokens_total += len(result.token_details)
                sample_summaries.append({
                    "sample_id": s.get("sample_id", str(global_idx)),
                    "source": s.get("source", "")[:120],
                    "reference": s.get("reference", "")[:120],
                    "translation": s.get("translation", "")[:120],
                    "n_tokens": len(result.token_details),
                })

    return {
        "dataset": dataset,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "n_samples": n_samples,
        "n_tokens_total": n_tokens_total,
        "samples": sample_summaries,
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run logit lens on pre-generated translations.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model path or repo",
    )
    parser.add_argument(
        "--input-dir",
        default="translations",
        help="Directory containing translation JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="lens_output_all",
        help="Root output directory for lens results",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k tokens per layer")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also output per-token grid (layers x positions)",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "nnsight"],
        default="hf",
        help="Model backend: 'hf' (transformers, default) or 'nnsight' (vLLM via nnsight)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force device (auto-detected by default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of translation files to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip individual samples that already have lens output JSONs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples per batch for lens computation (default 4)",
    )
    parser.add_argument(
        "--generation-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only compute lens on generated tokens (default: True). "
        "Use --no-generation-only to also include prompt tokens.",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Save full per-layer logits as safetensors files "
        "(shape: [n_layers, seq_len, vocab_size])",
    )
    parser.add_argument(
        "--logits-output-dir",
        default="logits_output_all",
        help="Root output directory for logits safetensors files",
    )
    args = parser.parse_args()
    print(args)

    sanitized_model = "__".join(args.model.split("/"))
    input_dir = args.input_dir + "/" + sanitized_model
    output_dir = args.output_dir + "/" + sanitized_model
    logits_output_dir = args.logits_output_dir + "/" + sanitized_model

    # --- Find all translation JSON files ---
    in_dir = Path(input_dir)
    if not in_dir.exists():
        print(f"Error: input directory {in_dir} does not exist",
              file=sys.stderr)
        sys.exit(1)

    # Look for translations_*.json in pair subdirectories
    json_files = sorted(in_dir.rglob("translations_*.json"))

    if not json_files:
        print(f"No translation JSON files found in {in_dir}",
              file=sys.stderr)
        sys.exit(1)

    if args.limit:
        json_files = json_files[: args.limit]

    total = len(json_files)

    out_dir = Path(output_dir)
    logits_out_dir = Path(logits_output_dir) if args.save_logits else None

    # --- Pre-check: if resuming, filter out fully-done pairs ---
    if args.resume:
        pending: list[Path] = []
        for jp in json_files:
            t = jp.stem
            if t.startswith("translations_"):
                t = t[len("translations_"):]
            pd = out_dir / t
            data = json.loads(jp.read_text())
            expected = len(data.get("samples", []))
            existing_lens = (
                len(list(pd.glob(f"lens_{t}_sample*.json")))
                if pd.exists() else 0
            )
            lens_ok = existing_lens >= expected
            logits_ok = True
            if args.save_logits and logits_out_dir is not None:
                lpd = logits_out_dir / t
                existing_logits = (
                    len(list(lpd.glob(f"logits_{t}_sample*.safetensors")))
                    if lpd.exists() else 0
                )
                logits_ok = existing_logits >= expected
            if lens_ok and logits_ok:
                print(f"  SKIP {t} (all {expected} samples done)",
                      file=sys.stderr)
            else:
                pending.append(jp)
        json_files = pending
        total = len(json_files)
        if total == 0:
            print("All pairs already done. Nothing to do.", file=sys.stderr)
            return

    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"Translation files to process: {total}", file=sys.stderr)
    print(f"  top_k={args.top_k}  grid={args.grid}  backend={args.backend}"
          f"  generation_only={args.generation_only}"
          f"  save_logits={args.save_logits}", file=sys.stderr)
    print(f"  Model: {args.model}", file=sys.stderr)
    print(f"  Input: {input_dir}/", file=sys.stderr)
    print(f"  Output: {output_dir}/", file=sys.stderr)
    if args.save_logits:
        print(f"  Logits: {logits_output_dir}/", file=sys.stderr)
    print(f"{'=' * 70}\n", file=sys.stderr)

    # --- Load model (once) ---
    print(f"Loading model {args.model} (backend={args.backend})", file=sys.stderr)

    if args.backend == "nnsight":
        hooked, tokenizer = load_nnsight_model(args.model, args.device)
    else:
        model, tokenizer, device = load_hf_model(args.model, args.device)
        hooked = make_hooked_model(model, tokenizer)

    # --- Output directory ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if logits_out_dir is not None:
        logits_out_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each translation file ---
    summaries: list[dict] = []
    global_t0 = time.time()

    for idx, json_path in enumerate(json_files):
        tag = json_path.stem
        if tag.startswith("translations_"):
            tag = tag[len("translations_"):]
        pair_dir = out_dir / tag

        print(
            f"[{idx + 1}/{total}] {tag}",
            file=sys.stderr,
        )

        pair_t0 = time.time()
        try:
            summary = run_lens_on_file(
                hooked,
                tokenizer,
                json_path,
                out_dir,
                args.top_k,
                args.grid,
                batch_size=args.batch_size,
                resume=args.resume,
                generation_only=args.generation_only,
                save_logits=args.save_logits,
                logits_out_dir=logits_out_dir,
            )
            summaries.append(summary)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            summaries.append({
                "error": str(exc),
                "src_lang": "",
                "tgt_lang": "",
            })

        elapsed = time.time() - global_t0
        pair_elapsed = time.time() - pair_t0
        done = idx + 1
        remaining = (elapsed / done) * (total - done) if done > 0 else 0
        print(
            f"  -> {summary.get('n_samples', 0)} samples, "
            f"{summary.get('n_tokens_total', 0)} tokens"
            + (f"  ERROR: {summary.get('error', '')}"
               if summary.get("error") else ""),
            file=sys.stderr,
        )
        print(
            f"  Progress: {done}/{total}  "
            f"Pair: {pair_elapsed:.1f}s  "
            f"Elapsed: {elapsed:.0f}s  ETA: {remaining:.0f}s",
            file=sys.stderr,
        )

        # Save running summary
        summary_data = {
            "total_files": total,
            "completed": done,
            "model": args.model,
            "pairs": summaries,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2)
        )

    # --- Final ---
    hooked.restore()

    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"DONE. {len(summaries)} files processed in "
          f"{time.time() - global_t0:.1f}s", file=sys.stderr)
    print(f"Results in {out_dir}/", file=sys.stderr)
    print(f"Summary: {out_dir / 'summary.json'}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)


if __name__ == "__main__":
    main()
