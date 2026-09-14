#!/usr/bin/env python
"""Build per-token language probability distribution using dataset config
labels on FineWeb + FineWeb2.

Instead of fasttext prediction, the language of each document is taken
directly from the dataset config name:
  - FineWeb (sample-10BT): all documents are English → "eng_Latn"
  - FineWeb2: config name IS the language, e.g. "fra_Latn", "cmn_Hans"

Procedure:
1. Stream FineWeb (English, sample-10BT) and FineWeb2 (languages present
   in FLORES+ and WMT24++ only)
2. For each document, use the known language label from the config name
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
dataset (I/O-bound) while a pool of worker processes runs tokenization
(CPU-bound) in parallel.  Each worker loads the tokenizers once at
startup via an initializer; batches are dispatched via
``imap_unordered`` so results are merged as soon as any worker finishes,
regardless of order.

Output JSON structure (same as build_token_lang_dist_fineweb-fasttext.py):
{
  "metadata": { ... },
  "tokens": {
    "0":  {"token": "<pad>", "total_count": 0, "langs": {}},
    "1":  {"token": "the",   "total_count": 50000, "langs": {"eng_Latn": 0.95, "fra_Latn": 0.05}},
    ...
  }
}
"""

import os
import argparse
import json
import multiprocessing as mp
import queue
import string
import threading
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from langcodes import Language

from dotenv import load_dotenv
load_dotenv()


# ── FineWeb2 config → FLORES+ language code remapping ─────────────────────────
# FineWeb2 sometimes uses a different script tag than FLORES+ for the same
# language.  When streaming a FineWeb2 config listed here, the lang_label
# stored in the distribution is the remapped (FLORES+) code.
FW2_CONFIG_REMAP: dict[str, str] = {
    "cmn_Hani": "cmn_Hans",   # FineWeb2 Hani → FLORES+ Hans (Simplified)
    "wuu_Hani": "wuu_Hans",
    "yue_Hani": "yue_Hant",
    "khk_Cyrl": "khk_Mong",   # FineWeb2 Cyrillic → FLORES+ Mongolian script
}


# ── Worker-process globals (loaded once per worker, never pickled) ────────────

_w_toks: list = []          # list of (name, tokenizer, vocab_size)


def _worker_init(model_names: list[str]) -> None:
    """Load tokenizers once per worker process."""
    global _w_toks
    from transformers import AutoTokenizer

    _w_toks = []
    for name in model_names:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if hasattr(tok, "tokenizer"):
            tok = tok.tokenizer
        _w_toks.append((name, tok, len(tok)))

    pid = os.getpid()
    print(f"  [worker {pid}] ready ({len(_w_toks)} tokenizers)", flush=True)


def _worker_process_batch(
    batch: tuple[list[str], str],
) -> tuple[list[dict[int, int]], str, int, int]:
    """Process a batch of texts with a known language label.

    Args:
        batch: ``(texts, lang_label)`` where *lang_label* is the same
        for all texts in the batch (e.g. "eng_Latn", "fra_Latn").

    Returns ``(per_tokenizer_counts, lang_label, n_processed, n_total)``.
    """
    global _w_toks

    texts, lang_label = batch
    n_tok = len(_w_toks)

    clean_texts: list[str] = []
    for text in texts:
        if not text or not text.strip():
            continue
        clean_texts.append(text.replace("\n", " ").strip())

    n_processed = len(clean_texts)

    results: list[dict[int, int]] = []

    for idx, (_, tok, _) in enumerate(_w_toks):
        if not clean_texts:
            results.append({})
            continue
        encodings = tok.encode(
            clean_texts, add_special_tokens=False,
        )
        arrays = [np.array(enc, dtype=np.int64) for enc in encodings if enc]
        if arrays:
            all_tids = np.concatenate(arrays)
            counts = np.bincount(all_tids)
            nz = counts.nonzero()[0]
            results.append(dict(zip(nz.tolist(), counts[nz].tolist())))
        else:
            results.append({})

    return results, lang_label, n_processed, len(texts)


# ── Helpers ───────────────────────────────────────────────────────────────────


def is_language_agnostic(token_str: str) -> bool:
    """Check if a token is language-agnostic (punctuation, digits, whitespace).

    These tokens get uniform P(L|t) = 1/M regardless of observed frequency,
    because their raw counts are heavily skewed by dataset formatting
    rather than genuine language signal (notes2.md §Edge Cases).
    """
    # Strip BPE/SentencePiece space markers and regular whitespace
    clean = token_str.replace("Ġ", "").replace("▁", "").replace(" ", "").strip()
    if not clean:
        return True  # Pure whitespace
    # Pure punctuation or digits
    if all(c in string.punctuation or c.isdigit() for c in clean):
        return True
    return False


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


def _batched(
    iterable,
    batch_size: int,
    max_rows: int | None = None,
    lang_label: str = "",
):
    """Yield ``(list_of_texts, lang_label)`` from *iterable*."""
    batch: list = []
    count = 0
    for item in iterable:
        if max_rows is not None and count >= max_rows:
            break
        batch.append(item)
        count += 1
        if len(batch) >= batch_size:
            yield batch, lang_label
            batch = []
    if batch:
        yield batch, lang_label


def _stream_parquet_texts(
    repo: str, config: str, max_rows: int | None = None,
) -> str:
    """Stream text rows from parquet files via pyarrow.

    Works for both FineWeb (``HuggingFaceFW/fineweb``) and FineWeb2
    (``HuggingFaceFW/fineweb-2``).  Bypasses the ``datasets`` library
    to avoid schema mismatch errors and streaming overhead.
    """
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    base = f"datasets/{repo}/data/{config}/train"
    try:
        files = fs.ls(base, detail=False)
    except Exception:
        return
    parquet_files = sorted(f for f in files if f.endswith(".parquet"))

    count = 0
    for pf_path in parquet_files:
        f = fs.open(pf_path)
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["text"]):
            for text in batch.column("text").to_pylist():
                if text:
                    yield text
                    count += 1
                    if max_rows is not None and count >= max_rows:
                        f.close()
                        return
        f.close()


def _merge_counts(
    token_lang_counts: list[dict[int, Counter]],
    worker_counts: list[dict[int, int]],
    lang_label: str,
    n_tok: int,
) -> None:
    """Merge worker-produced per-tokenizer counts into the global accumulators."""
    for idx in range(n_tok):
        for tid, count in worker_counts[idx].items():
            token_lang_counts[idx][tid][lang_label] += count


def _prefetched(iterable, maxsize: int):
    """Wrap an iterable with a thread-based prefetch buffer.

    A background thread consumes *iterable* and fills a bounded queue,
    overlapping I/O with computation in worker processes.
    """
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    _DONE = object()
    _err: list = [None]

    def _producer():
        try:
            for item in iterable:
                q.put(item)
        except Exception as e:
            _err[0] = e
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is _DONE:
            if _err[0] is not None:
                raise _err[0]
            break
        yield item


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-token language probability distribution using"
        " dataset config labels on FineWeb + FineWeb2",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["google/gemma-4-12B"],
        help="HuggingFace model path(s) for tokenizer(s) (default: google/gemma-4-12B)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: token_lang_dist_fineweb_dataset_<sanitized_model>.json). "
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
        "--progress-every",
        type=int,
        default=1_000,
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
        default=1_000,
        help="Documents per batch sent to each worker (default: 5000)",
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

    # Build list of (fw2_config, lang_label) pairs.
    # Directly matching configs use their own name as lang_label.
    # Configs needing remapping (e.g. cmn_Hani → cmn_Hans) use the remapped label.
    fw2_configs: list[tuple[str, str]] = []
    for c in fw2_all:
        if c in lang_set:
            fw2_configs.append((c, c))
        elif c in FW2_CONFIG_REMAP and FW2_CONFIG_REMAP[c] in lang_set:
            fw2_configs.append((c, FW2_CONFIG_REMAP[c]))

    # ── Create worker pool ──────────────────────────────────────────────
    print(
        f"Starting pool: {args.workers} workers, batch_size={args.batch_size}",
        flush=True,
    )
    pool = mp.Pool(
        args.workers,
        initializer=_worker_init,
        initargs=(args.model,),
    )

    fw_count = 0
    fw2_count = 0
    total_tokens = 0

    try:
        # ── Stream FineWeb (English) ────────────────────────────────────
        print(
            f"\n=== FineWeb (English, {args.fineweb_config}) "
            f"— up to {args.max_rows:,} rows ===",
            flush=True,
        )
        try:
            ds = load_dataset(
                "HuggingFaceFW/fineweb",
                args.fineweb_config,
                split="train",
                streaming=True,
            )
            for worker_counts, lang_label, n_proc, n_total in pool.imap_unordered(
                _worker_process_batch,
                _prefetched(
                    _batched(
                        (r.get("text", "") for r in ds),
                        args.batch_size,
                        args.max_rows,
                        lang_label="eng_Latn",
                    ),
                    maxsize=args.workers * 2,
                ),
            ):
                _merge_counts(token_lang_counts, worker_counts, lang_label, n_tok)
                if n_proc:
                    all_langs.add(lang_label)
                total_docs += n_proc
                fw_count += n_total
                batch_tokens = sum(worker_counts[0].values())
                total_tokens += batch_tokens
                if total_docs // args.progress_every != (total_docs - n_proc) // args.progress_every:
                    print(
                        f"  [{total_docs:>10,} docs | {total_tokens:>12,} tokens] "
                        f"{len(token_lang_counts[0]):>8,} detected tokens | "
                        f"{len(all_langs):>3} langs",
                        flush=True,
                    )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FineWeb error: {e}", flush=True)
        print(f"  FineWeb: {fw_count:,} records ({total_docs:,} non-empty, {total_tokens:,} tokens)", flush=True)

        # ── Stream FineWeb2 (filtered configs) ──────────────────────────
        print(
            f"\n=== FineWeb2 — up to {args.max_rows:,} rows per config ===",
            flush=True,
        )
        print(
            f"  {len(fw2_configs)}/{len(fw2_all)} configs "
            f"(filtered to FLORES+ & WMT24++ languages)",
            flush=True,
        )

        for i, (config, lang_label) in enumerate(fw2_configs):
            try:
                config_count = 0
                config_proc = 0
                config_tokens = 0
                for worker_counts, lang_label, n_proc, n_total in pool.imap_unordered(
                    _worker_process_batch,
                    _prefetched(
                        _batched(
                            _stream_parquet_texts(
                                "HuggingFaceFW/fineweb-2", config,
                                args.max_rows,
                            ),
                            args.batch_size,
                            args.max_rows,
                            lang_label=lang_label,
                        ),
                        maxsize=args.workers * 2,
                    ),
                ):
                    _merge_counts(token_lang_counts, worker_counts, lang_label, n_tok)
                    if n_proc:
                        all_langs.add(lang_label)
                    total_docs += n_proc
                    config_count += n_total
                    config_proc += n_proc
                    fw2_count += n_total
                    batch_tokens = sum(worker_counts[0].values())
                    total_tokens += batch_tokens
                    config_tokens += batch_tokens
                    if total_docs // args.progress_every != (total_docs - n_proc) // args.progress_every:
                        print(
                            f"  [{total_docs:>10,} docs | {total_tokens:>12,} tokens] "
                            f"{len(token_lang_counts[0]):>8,} detected tokens | "
                            f"{len(all_langs):>3} langs",
                            flush=True,
                        )
                print(
                    f"  [{i + 1}/{len(fw2_configs)}] {config}→{lang_label}: "
                    f"{config_count:,} records ({config_proc:,} non-empty, {config_tokens:,} tokens) "
                    f"(running total: {fw2_count:,})",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{i + 1}/{len(fw2_configs)}] SKIP {config}: {e}", flush=True)
                continue

        print(f"  FineWeb2 total: {fw2_count:,} records, {total_tokens:,} tokens", flush=True)

    finally:
        pool.close()
        pool.join()

    # ── Normalize and save per tokenizer ─────────────────────────────────
    sorted_langs = sorted(all_langs)
    num_langs = len(sorted_langs)

    for idx, (model_name, tok, vocab_size) in enumerate(tokenizers):
        counts_map = token_lang_counts[idx]
        print(f"\nNormalizing {len(counts_map):,} tokens with data "
              f"({model_name}) ...", flush=True)

        all_ids = list(range(vocab_size))
        token_strings = tok.convert_ids_to_tokens(all_ids)

        agnostic_ids = {
            tid for tid, ts in enumerate(token_strings)
            if is_language_agnostic(ts)
        }
        n_agnostic = len(agnostic_ids)
        print(f"  Language-agnostic tokens (uniform): {n_agnostic:,}", flush=True)

        token_dist: dict[str, dict] = {}
        for tid in range(vocab_size):
            token_str = token_strings[tid]

            if tid in agnostic_ids:
                uniform = 1.0 / num_langs if num_langs > 0 else 0.0
                langs = {lang: uniform for lang in sorted_langs}
            elif tid in counts_map:
                counts = counts_map[tid]
                total = sum(counts.values())
                langs = {lang: count / total for lang, count in counts.most_common()}
            else:
                langs = {}
            token_dist[str(tid)] = {
                "token": token_str,
                "total_count": sum(counts_map[tid].values()) if tid in counts_map else 0,
                "langs": langs,
            }

        if args.output and n_tok == 1:
            out_path = args.output
        else:
            sanitized = "__".join(model_name.split("/"))
            out_path = f"token_lang_dist_fineweb_dataset_{sanitized}.json"

        output = {
            "metadata": {
                "tokenizer": model_name,
                "tokenizer_class": type(tok).__name__,
                "method": "dataset_label",
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
