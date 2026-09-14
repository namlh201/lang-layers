#!/usr/bin/env python
"""Build per-token language probability distribution using fasttext.

Procedure:
1. Load FLORES+ (all language configs) and WMT24++ (all configs)
2. For each sentence, detect language with fasttext (lid.176.bin)
3. Tokenize the text into tokens with the model tokenizer(s)
4. Accumulate language counts per token ID (each token in the sentence
   gets +1 for the predicted language)
5. Normalize counts to probability distributions per token
6. Save as JSON for easy future loading

Multiple tokenizers can be specified (--model m1 m2 ...) and each gets
its own output JSON, allowing one pass over the data to build
distributions for several tokenizers.

Output JSON structure:
{
  "metadata": {
    "tokenizer": "google/gemma-4-12B",
    "ft_model": "lid.176.bin",
    "datasets": ["flores", "wmt24pp"],
    "total_sentences": 500000,
    "vocab_size": 256000,
    "num_tokens_with_data": 200000,
    "num_languages_detected": 87,
    "languages_detected": ["eng", "fra", ...]
  },
  "tokens": {
    "0":  {"token": "<pad>", "total_count": 0,    "langs": {}},
    "1":  {"token": "the",   "total_count": 50000, "langs": {"eng": 0.95, "fra": 0.05}},
    ...
  }
}
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-token language probability distribution using fasttext",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["google/gemma-4-12B"],
        help="HuggingFace model path(s) for tokenizer(s) (default: google/gemma-4-12B)",
    )
    parser.add_argument(
        "--ft-model",
        default="lid.176.bin",
        help="Path to fasttext language identification model (default: lid.176.bin)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: token_lang_dist_<sanitized_model>.json). "
             "If multiple models are given, this is ignored and one file per model is written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["flores", "wmt24pp"],
        choices=["flores", "wmt24pp"],
        help="Datasets to process (default: flores wmt24pp)",
    )
    parser.add_argument(
        "--flores-split",
        default="devtest",
        help="FLORES+ split to use (default: devtest)",
    )
    parser.add_argument(
        "--wmt-split",
        default="train",
        help="WMT24++ split to use (default: train)",
    )
    parser.add_argument(
        "--max-per-config",
        type=int,
        default=None,
        help="Max sentences per dataset config (default: None = all)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N sentences (default: 1000)",
    )
    args = parser.parse_args()

    # ── Load fasttext model ──────────────────────────────────────────────
    print(f"Loading fasttext model from {args.ft_model} ...", flush=True)
    import fasttext

    ft_model = fasttext.load_model(args.ft_model)
    ft_labels = ft_model.get_labels()
    print(f"  {len(ft_labels)} labels", flush=True)

    # ── Load tokenizers ──────────────────────────────────────────────────
    from transformers import AutoTokenizer

    tokenizers: list[tuple[str, any, int]] = []  # (model_name, tokenizer, vocab_size)
    for model_name in args.model:
        print(f"Loading tokenizer from {model_name} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(tok, "tokenizer"):
            tok = tok.tokenizer
        vs = len(tok)
        print(f"  Vocab size: {vs}", flush=True)
        tokenizers.append((model_name, tok, vs))

    # ── Per-tokenizer accumulators ───────────────────────────────────────
    n_tok = len(tokenizers)
    # token_lang_counts[model_idx][token_id] = Counter({lang: count, ...})
    token_lang_counts: list[dict[int, Counter]] = [
        defaultdict(Counter) for _ in range(n_tok)
    ]
    total_sentences = 0
    all_langs: set[str] = set()

    def detect_and_count(text: str) -> None:
        """Detect language with fasttext, tokenize with all tokenizers, accumulate counts."""
        nonlocal total_sentences
        if not text or not text.strip():
            return
        text_clean = text.replace("\n", " ").strip()
        labels_pred, _ = ft_model.predict(text_clean)
        lang = labels_pred[0].replace("__label__", "")
        all_langs.add(lang)
        for idx, (_, tok, _) in enumerate(tokenizers):
            token_ids = tok.encode(text_clean, add_special_tokens=False)
            for tid in token_ids:
                token_lang_counts[idx][tid][lang] += 1
        total_sentences += 1
        if total_sentences % args.progress_every == 0:
            print(
                f"  [{total_sentences:>10,}] "
                f"{len(token_lang_counts[0]):>8,} tokens (tok[0]) | "
                f"{len(all_langs):>3} langs",
                flush=True,
            )

    # ── Process FLORES+ ──────────────────────────────────────────────────
    if "flores" in args.datasets:
        from datasets import get_dataset_config_names, load_dataset

        print("\n=== FLORES+ ===", flush=True)
        configs = [
            c for c in get_dataset_config_names("openlanguagedata/flores_plus")
            if c != "default"
        ]
        print(f"Found {len(configs)} configs", flush=True)
        for i, config in enumerate(configs):
            try:
                ds = load_dataset(
                    "openlanguagedata/flores_plus",
                    config,
                    split=args.flores_split,
                )
            except Exception as e:
                print(f"  [{i + 1}/{len(configs)}] SKIP {config}: {e}", flush=True)
                continue
            n = 0
            for record in ds:
                if args.max_per_config and n >= args.max_per_config:
                    break
                detect_and_count(record["text"])
                n += 1
            print(f"  [{i + 1}/{len(configs)}] {config}: {n:,} sentences", flush=True)

    # ── Process WMT24++ ──────────────────────────────────────────────────
    if "wmt24pp" in args.datasets:
        from datasets import get_dataset_config_names, load_dataset

        print("\n=== WMT24++ ===", flush=True)
        configs = [
            c for c in get_dataset_config_names("google/wmt24pp")
            if c != "default"
        ]
        print(f"Found {len(configs)} configs", flush=True)

        wmt_source_done = False
        for i, config in enumerate(configs):
            try:
                ds = load_dataset("google/wmt24pp", config, split=args.wmt_split)
            except Exception as e:
                print(f"  [{i + 1}/{len(configs)}] SKIP {config}: {e}", flush=True)
                continue
            n = 0
            for record in ds:
                if args.max_per_config and n >= args.max_per_config:
                    break
                if record.get("is_bad_source", False):
                    continue
                if not wmt_source_done:
                    text = record.get("source", "")
                    if text:
                        detect_and_count(text)
                text = record.get("target", "")
                if text:
                    detect_and_count(text)
                n += 1
            wmt_source_done = True
            print(f"  [{i + 1}/{len(configs)}] {config}: {n:,} sentences", flush=True)

    # ── Normalize and save per tokenizer ─────────────────────────────────
    for idx, (model_name, tok, vocab_size) in enumerate(tokenizers):
        counts_map = token_lang_counts[idx]
        print(f"\nNormalizing {len(counts_map):,} tokens with data "
              f"({model_name}) ...", flush=True)

        all_ids = list(range(vocab_size))
        token_strings = tok.convert_ids_to_tokens(all_ids)

        token_dist: dict[str, dict] = {}
        for tid in range(vocab_size):
            if tid in counts_map:
                counts = counts_map[tid]
                total = sum(counts.values())
                langs = {lang: count / total for lang, count in counts.most_common()}
            else:
                langs = {}
            token_dist[str(tid)] = {
                "token": token_strings[tid],
                "total_count": sum(counts_map[tid].values()) if tid in counts_map else 0,
                "langs": langs,
            }

        # Determine output path
        if args.output and n_tok == 1:
            out_path = args.output
        else:
            sanitized = "__".join(model_name.split("/"))
            out_path = f"token_lang_dist_{sanitized}.json"

        output = {
            "metadata": {
                "tokenizer": model_name,
                "ft_model": args.ft_model,
                "datasets": args.datasets,
                "flores_split": args.flores_split,
                "wmt_split": args.wmt_split,
                "total_sentences": total_sentences,
                "vocab_size": vocab_size,
                "num_tokens_with_data": len(counts_map),
                "num_languages_detected": len(all_langs),
                "languages_detected": sorted(all_langs),
            },
            "tokens": token_dist,
        }

        print(f"Saving to {out_path} ...", flush=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        file_mb = Path(out_path).stat().st_size / (1024 * 1024)
        print(
            f"Done ({model_name}). {total_sentences:,} sentences | "
            f"{len(counts_map):,}/{vocab_size:,} tokens with data | "
            f"{len(all_langs)} languages | "
            f"File: {file_mb:.1f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
