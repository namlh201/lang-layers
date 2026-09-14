#!/usr/bin/env python
"""Build per-token language probability distribution using fasttext on FineWeb + FineWeb2.

Procedure:
1. Stream FineWeb (English, sample-10BT) and FineWeb2 (languages present in
   FLORES+ and WMT24++ only)
2. For each document, detect language with fasttext (lid.176.bin)
3. Tokenize the text into tokens with the model tokenizer(s)
4. Accumulate language counts per token ID
5. Normalize counts to probability distributions per token
6. Save as JSON for easy future loading

--max-rows (default 1,000,000) rows are streamed *per config* (FineWeb and
each FineWeb2 config) via streaming, so the full dataset is never downloaded.

Multiple tokenizers can be specified (--model m1 m2 ...) and each gets
its own output JSON, allowing one pass over the data to build
distributions for several tokenizers.

Multiprocessing: the main process streams batches of documents from the
dataset (I/O-bound) while a pool of worker processes runs fasttext
prediction + tokenization (CPU-bound) in parallel.  Each worker loads
its own copy of fasttext and the tokenizers once at startup via an
initializer; batches are dispatched via ``imap_unordered`` so results are
merged as soon as any worker finishes, regardless of order.

Output JSON structure (same as build_token_lang_dist.py):
{
  "metadata": { ... },
  "tokens": {
    "0":  {"token": "<pad>", "total_count": 0, "langs": {}},
    "1":  {"token": "the",   "total_count": 50000, "langs": {"eng": 0.95, "fra": 0.05}},
    ...
  }
}
"""

import os
import argparse
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

from langcodes import Language

from dotenv import load_dotenv
load_dotenv()


# ── Worker-process globals (loaded once per worker, never pickled) ────────────

_w_ft = None
_w_toks: list = []          # list of (name, tokenizer, vocab_size)
_w_max_chars: int = 2000


def _worker_init(ft_model_path: str, model_names: list[str], max_chars: int) -> None:
    """Load fasttext + tokenizers once per worker process."""
    global _w_ft, _w_toks, _w_max_chars
    import fasttext
    from transformers import AutoTokenizer

    _w_ft = fasttext.load_model(ft_model_path)
    _w_max_chars = max_chars
    _w_toks = []
    for name in model_names:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if hasattr(tok, "tokenizer"):
            tok = tok.tokenizer
        _w_toks.append((name, tok, len(tok)))

    pid = os.getpid()
    print(f"  [worker {pid}] ready ({len(_w_toks)} tokenizers, "
          f"{len(_w_ft.get_labels())} fasttext labels)", flush=True)


def _worker_process_batch(
    texts: list[str],
) -> tuple[list[dict[int, Counter]], set[str], int, int]:
    """Process a batch of texts.

    Returns ``(per_tokenizer_counts, langs_detected, n_processed, n_total)``.

    fasttext prediction runs per-document (no batch API), but tokenization
    is batched via ``encode_batch`` so the Rust backend parallelizes
    internally and avoids Python-loop overhead.
    """
    global _w_ft, _w_toks, _w_max_chars

    n_tok = len(_w_toks)

    # ── Clean texts + detect languages ────────────────────────────────
    clean_texts: list[str] = []
    langs_list: list[str] = []
    langs: set[str] = set()

    for text in texts:
        if not text or not text.strip():
            continue
        text_clean = text.replace("\n", " ").strip()
        if len(text_clean) > _w_max_chars:
            text_clean = text_clean[: _w_max_chars]
        labels_pred, _ = _w_ft.predict(text_clean)
        lang = labels_pred[0].replace("__label__", "")
        langs.add(lang)
        clean_texts.append(text_clean)
        langs_list.append(lang)

    n_processed = len(clean_texts)

    # ── Batch-tokenize per tokenizer ───────────────────────────────────
    results: list[dict[int, Counter]] = [defaultdict(Counter) for _ in range(n_tok)]

    for idx, (_, tok, _) in enumerate(_w_toks):
        encodings = tok.encode(
            clean_texts, add_special_tokens=False,
        )
        # encodings = tok.batch_encode_plus(
        #     clean_texts, add_special_tokens=False,
        # )
        for enc, lang in zip(encodings, langs_list):
            # for tid in enc.ids:
            for tid in enc:
                results[idx][tid][lang] += 1

    return results, langs, n_processed, len(texts)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_lang_set() -> set[str]:
    """Collect the set of lang_Script codes from FLORES+ and WMT24++.

    FLORES+ configs are already in lang_Script format (e.g. fra_Latn).
    WMT24++ configs are en-<locale> (e.g. en-zh_CN); we extract the 2-letter
    code, resolve common aliases, and convert to lang_Script.
    """
    from datasets import get_dataset_config_names

    lang_set: set[str] = set()

    # FLORES+ configs — already lang_Script
    for c in get_dataset_config_names("openlanguagedata/flores_plus"):
        if c != "default" and "_" in c:
            lang_set.add(c)

    # WMT24++ configs — en-<locale>
    aliases = {
        "zh": "cmn_Hans", "zh_TW": "cmn_Hant",
        "nb": "nob_Latn", "nn": "nno_Latn",
    }
    for c in get_dataset_config_names("google/wmt24pp"):
        if "-" not in c:
            continue
        tgt = c.split("-", 1)[1]       # e.g. zh_CN
        code2 = tgt.split("_")[0]      # e.g. zh
        if tgt in aliases:
            lang_set.add(aliases[tgt])
        elif code2 in aliases:
            lang_set.add(aliases[code2])
        else:
            try:
                alpha3 = Language.get(code2).to_alpha3()
                script = Language.get(tgt).script or "Latn"
                lang_set.add(f"{alpha3}_{script}")
            except Exception:
                pass

    return lang_set


def _batched(iterable, batch_size: int, max_rows: int | None = None):
    """Yield lists of at most *batch_size* items from *iterable*."""
    batch: list = []
    count = 0
    for item in iterable:
        if max_rows is not None and count >= max_rows:
            break
        batch.append(item)
        count += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _merge_counts(
    token_lang_counts: list[dict[int, Counter]],
    worker_counts: list[dict[int, Counter]],
    n_tok: int,
) -> None:
    """Merge worker-produced per-tokenizer counts into the global accumulators."""
    for idx in range(n_tok):
        for tid, counter in worker_counts[idx].items():
            token_lang_counts[idx][tid].update(counter)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-token language probability distribution using fasttext on FineWeb + FineWeb2",
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
        help="Output JSON path (default: token_lang_dist_fineweb_<sanitized_model>.json). "
             "If multiple models are given, this is ignored and one file per model is written.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1_000_000,
        help="Rows to stream per config — FineWeb and each FineWeb2 config (default: 1,000,000)",
    )
    parser.add_argument(
        "--fineweb-config",
        default="sample-10BT",
        help="FineWeb config to stream (default: sample-10BT)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Truncate each document to this many chars before langid (default: 2000)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N documents (default: 1000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
        help="Number of worker processes (default: SLURM_CPUS_PER_TASK or CPU count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Documents per batch sent to each worker (default: 500)",
    )
    args = parser.parse_args()
    print(args)

    # ── Load tokenizers in main process (for final save) ─────────────────
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

    n_tok = len(tokenizers)

    # ── Per-tokenizer accumulators ───────────────────────────────────────
    token_lang_counts: list[dict[int, Counter]] = [
        defaultdict(Counter) for _ in range(n_tok)
    ]
    total_docs = 0
    all_langs: set[str] = set()

    # ── Build FineWeb2 config list ───────────────────────────────────────
    from datasets import get_dataset_config_names, load_dataset

    fw2_all = [
        c for c in get_dataset_config_names("HuggingFaceFW/fineweb-2")
        if c != "default"
    ]
    lang_set = _build_lang_set()
    fw2_configs = [c for c in fw2_all if c in lang_set]

    # ── Create worker pool ──────────────────────────────────────────────
    print(
        f"Starting pool: {args.workers} workers, batch_size={args.batch_size}",
        flush=True,
    )
    pool = mp.Pool(
        args.workers,
        initializer=_worker_init,
        initargs=(args.ft_model, args.model, args.max_chars),
    )

    try:
        # ── Stream FineWeb (English) ────────────────────────────────────
        print(
            f"\n=== FineWeb (English, {args.fineweb_config}) "
            f"— up to {args.max_rows:,} rows ===",
            flush=True,
        )
        fw_count = 0
        try:
            ds = load_dataset(
                "HuggingFaceFW/fineweb",
                args.fineweb_config,
                split="train",
                streaming=True,
            )
            for worker_counts, worker_langs, n_proc, n_total in pool.imap_unordered(
                _worker_process_batch,
                _batched(
                    (r.get("text", "") for r in ds),
                    args.batch_size,
                    args.max_rows,
                ),
            ):
                _merge_counts(token_lang_counts, worker_counts, n_tok)
                all_langs.update(worker_langs)
                total_docs += n_proc
                fw_count += n_total
                if total_docs // args.progress_every != (total_docs - n_proc) // args.progress_every:
                    print(
                        f"  [{total_docs:>10,}] "
                        f"{len(token_lang_counts[0]):>8,} tokens (tok[0]) | "
                        f"{len(all_langs):>3} langs",
                        flush=True,
                    )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FineWeb error: {e}", flush=True)
        print(f"  FineWeb: {fw_count:,} records ({total_docs:,} non-empty)", flush=True)

        # ── Stream FineWeb2 (filtered configs) ──────────────────────────
        fw2_count = 0
        print(
            f"\n=== FineWeb2 — up to {args.max_rows:,} rows per config ===",
            flush=True,
        )
        print(
            f"  {len(fw2_configs)}/{len(fw2_all)} configs "
            f"(filtered to FLORES+ & WMT24++ languages)",
            flush=True,
        )

        for i, config in enumerate(fw2_configs):
            try:
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-2",
                    config,
                    split="train",
                    streaming=True,
                )
                config_count = 0
                config_proc = 0
                for worker_counts, worker_langs, n_proc, n_total in pool.imap_unordered(
                    _worker_process_batch,
                    _batched(
                        (r.get("text", "") for r in ds),
                        args.batch_size,
                        args.max_rows,
                    ),
                ):
                    _merge_counts(token_lang_counts, worker_counts, n_tok)
                    all_langs.update(worker_langs)
                    total_docs += n_proc
                    config_count += n_total
                    config_proc += n_proc
                    fw2_count += n_total
                    if total_docs // args.progress_every != (total_docs - n_proc) // args.progress_every:
                        print(
                            f"  [{total_docs:>10,}] "
                            f"{len(token_lang_counts[0]):>8,} tokens (tok[0]) | "
                            f"{len(all_langs):>3} langs",
                            flush=True,
                        )
                print(
                    f"  [{i + 1}/{len(fw2_configs)}] {config}: "
                    f"{config_count:,} records ({config_proc:,} non-empty) "
                    f"(running total: {fw2_count:,})",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{i + 1}/{len(fw2_configs)}] SKIP {config}: {e}", flush=True)
                continue

        print(f"  FineWeb2 total: {fw2_count:,} records", flush=True)

    finally:
        pool.close()
        pool.join()

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

        if args.output and n_tok == 1:
            out_path = args.output
        else:
            sanitized = "__".join(model_name.split("/"))
            out_path = f"token_lang_dist_fineweb_{sanitized}.json"

        output = {
            "metadata": {
                "tokenizer": model_name,
                "tokenizer_class": type(tok).__name__,
                "ft_model": args.ft_model,
                "sources": ["fineweb", "fineweb2"],
                "fineweb_config": args.fineweb_config,
                "fineweb_docs": fw_count,
                "fineweb2_docs": fw2_count,
                "total_docs": total_docs,
                "max_rows_per_config": args.max_rows,
                "vocab_size": vocab_size,
                "num_tokens_with_data": len(counts_map),
                "num_languages_detected": len(all_langs),
                "languages_detected": sorted(all_langs),
                "workers": args.workers,
                "batch_size": args.batch_size,
            },
            "tokens": token_dist,
        }

        print(f"Saving to {out_path} ...", flush=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        file_mb = Path(out_path).stat().st_size / (1024 * 1024)
        print(
            f"Done ({model_name}). {total_docs:,} docs | "
            f"{len(counts_map):,}/{vocab_size:,} tokens with data | "
            f"{len(all_langs)} languages | "
            f"File: {file_mb:.1f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
