"""Generate translations across ALL language pairs in FLORES+ and/or WMT24++.

For each language pair, the script:
1. Loads few-shot translation examples.
2. Generates a translation for each test sample.
3. Saves a JSON file with source text, reference translation, generated
   translation, and the prompt used.

The output JSON files can later be consumed by ``lens_all.py`` to run the
logit lens with a HuggingFace model.

Usage::

    # All WMT24++ pairs (55 en-X), 3 samples each
    python translate_all.py --dataset wmt24pp --n-samples 3

    # All FLORES+ pairs from English to every language (230 targets)
    python translate_all.py --dataset flores --src eng_Latn --n-samples 3

    # Both datasets, limited pairs for testing
    python translate_all.py --dataset both --limit 5

    # FLORES+ from multiple source languages
    python translate_all.py --dataset flores \\
        --src eng_Latn,fra_Latn,deu_Latn --n-samples 2

    # Specific targets only
    python translate_all.py --dataset flores --src eng_Latn \\
        --tgt-filter cmn_Hans,jpn_Jpan,kor_Hang

    # Skip pairs already completed (resume)
    python translate_all.py --dataset both --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from utils.load import (
    detect_gpu,
    detect_backend,
    lang_name,
    load_flores,
    load_model,
    load_wmt24pp,
    _resolve_flores_lang,
)
from utils.generation import build_few_shot_prompt, generate_translations

load_dotenv()


# --------------------------------------------------------------------------- #
#  Config discovery
# --------------------------------------------------------------------------- #

# 10 languages for non-English cross-lingual pairs (all 90 ordered pairs)
CROSS_LINGUAL_LANGS = [
    "ces_Latn",   # Czech
    "slk_Latn",   # Slovak
    "deu_Latn",   # German
    "nld_Latn",   # Dutch
    "vie_Latn",   # Vietnamese
    "tha_Thai",   # Thai
    "cmn_Hans",   # Chinese
    "jpn_Jpan",   # Japanese
    "kor_Hang",   # Korean
    "hin_Deva",   # Hindi
]

# Map FLORES+ lang_Script → ISO 639-1 two-letter code for WMT24++ config matching
_CROSS_LINGUAL_FLORES_TO_ISO2 = {
    "ces_Latn": "cs",
    "slk_Latn": "sk",
    "deu_Latn": "de",
    "nld_Latn": "nl",
    "vie_Latn": "vi",
    "tha_Thai": "th",
    "cmn_Hans": "zh",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "hin_Deva": "hi",
}


def get_wmt24pp_pairs() -> list[tuple[str, str]]:
    """Return all (source, target) pairs from WMT24++ configs.

    Source is always ``en``; target is the locale suffix (e.g. ``zh_CN``).
    """
    from datasets import get_dataset_config_names

    configs = get_dataset_config_names("google/wmt24pp")
    pairs: list[tuple[str, str]] = []
    for cfg in configs:
        if cfg.startswith("en-"):
            tgt = cfg[3:]
            pairs.append(("en", tgt))
    return pairs


def get_wmt24pp_same_lang_pairs() -> list[tuple[str, str]]:
    """Return (tgt, tgt) pairs for every WMT24++ target language."""
    from datasets import get_dataset_config_names

    configs = get_dataset_config_names("google/wmt24pp")
    pairs: list[tuple[str, str]] = []
    for cfg in configs:
        if cfg.startswith("en-"):
            tgt = cfg[3:]
            pairs.append((tgt, tgt))
    return pairs


def get_flores_pairs(
    src_langs: list[str],
    tgt_filter: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return all (source, target) pairs for FLORES+.

    Args:
        src_langs: List of source language codes (e.g. ``["eng_Latn"]``).
        tgt_filter: Optional set of target codes to restrict to.
    """
    from datasets import get_dataset_config_names

    all_configs = get_dataset_config_names("openlanguagedata/flores_plus")
    # Skip "default" which is the combined config
    all_langs = [c for c in all_configs if c != "default"]

    # Resolve aliases
    from utils.load import _resolve_flores_lang

    src_resolved = [_resolve_flores_lang(s) for s in src_langs]

    pairs: list[tuple[str, str]] = []
    for src in src_resolved:
        for tgt in all_langs:
            if tgt == src:
                continue
            if tgt_filter:
                tgt_r = _resolve_flores_lang(tgt)
                if tgt_r not in tgt_filter and tgt not in tgt_filter:
                    continue
            pairs.append((src, tgt))
    return pairs


def get_flores_same_lang_pairs() -> list[tuple[str, str]]:
    """Return (src, tgt) pairs where src == tgt for every FLORES+ language.

    This covers all languages in both FLORES+ and WMT24++ (the latter's
    target languages are a subset of FLORES+).
    """
    from datasets import get_dataset_config_names

    all_configs = get_dataset_config_names("openlanguagedata/flores_plus")
    all_langs = [c for c in all_configs if c != "default"]
    return [(lang, lang) for lang in all_langs]


def get_cross_lingual_pairs() -> list[tuple[str, str]]:
    """Return all 90 ordered pairs among CROSS_LINGUAL_LANGS (excl. self-pairs)."""
    pairs = []
    for src in CROSS_LINGUAL_LANGS:
        for tgt in CROSS_LINGUAL_LANGS:
            if src != tgt:
                pairs.append((src, tgt))
    return pairs


def _resolve_wmt24pp_locale(flores_lang: str) -> str | None:
    """Find the WMT24++ locale code for a FLORES+ lang_Script code."""
    from datasets import get_dataset_config_names

    iso2 = _CROSS_LINGUAL_FLORES_TO_ISO2.get(flores_lang)
    if iso2 is None:
        return None
    for cfg in get_dataset_config_names("google/wmt24pp"):
        if cfg.startswith("en-"):
            locale = cfg[3:]
            if locale.split("_")[0].lower() == iso2:
                return locale
    return None


def get_wmt24pp_cross_lingual_pairs() -> list[tuple[str, str]]:
    """Return all 90 ordered cross-lingual pairs among WMT24++ target languages.

    Each pair uses WMT24++ locale codes (e.g. ``cs_CZ``, ``de_DE``).
    """
    locales: dict[str, str] = {}  # flores_lang → wmt24pp locale
    for flores_lang in CROSS_LINGUAL_LANGS:
        locale = _resolve_wmt24pp_locale(flores_lang)
        if locale:
            locales[flores_lang] = locale

    pairs: list[tuple[str, str]] = []
    for src_flores in locales:
        for tgt_flores in locales:
            if src_flores != tgt_flores:
                pairs.append((locales[src_flores], locales[tgt_flores]))
    return pairs


# --------------------------------------------------------------------------- #
#  Summary
# --------------------------------------------------------------------------- #


class PairSummary:
    """Summary of one language pair run."""

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        dataset: str,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.dataset = dataset
        self.n_samples = 0
        self.samples: list[dict] = []
        self.elapsed_s: float = 0.0
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "src_name": lang_name(self.src_lang),
            "tgt_name": lang_name(self.tgt_lang),
            "n_samples": self.n_samples,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "samples": self.samples,
        }


# --------------------------------------------------------------------------- #
#  Translation per pair
# --------------------------------------------------------------------------- #


def run_one_pair(
    model,
    tokenizer,
    dataset: str,
    src_lang: str,
    tgt_lang: str,
    n_shot: int,
    n_samples: int,
    max_gen_tokens: int,
    out_dir: Path,
    is_vlm: bool = False,
    vllm_wrapper=None,
    backend_kind: str = "mlx-lm",
) -> PairSummary:
    """Generate translations for one language pair and save as JSON."""

    summary = PairSummary(src_lang, tgt_lang, dataset)
    t0 = time.time()

    tag = f"{dataset}_{src_lang}_{tgt_lang}"
    pair_dir = out_dir / tag
    pair_dir.mkdir(parents=True, exist_ok=True)

    try:
        if dataset == "flores":
            few_shot, test = load_flores(src_lang, tgt_lang, n_shot, n_samples)
        else:
            same_lang = src_lang == tgt_lang
            cross_lang = src_lang != "en" and not same_lang
            few_shot, test = load_wmt24pp(
                tgt_lang, n_shot, n_samples,
                same_lang=same_lang,
                src_lang=src_lang if cross_lang else None,
            )
    except Exception as exc:
        summary.error = f"load_failed: {exc!s}"
        summary.elapsed_s = time.time() - t0
        return summary

    samples_json: list[dict] = []

    try:
        prompts = [
            build_few_shot_prompt(few_shot, sample)
            for sample in test
        ]

        print(
            f"    Generating {len(test)} translations (batch)...",
            file=sys.stderr,
        )

        translations = generate_translations(
            model, tokenizer, prompts,
            max_tokens=max_gen_tokens,
            is_vlm=is_vlm,
            vllm_wrapper=vllm_wrapper,
            backend_kind=backend_kind,
        )

        for i, (sample, prompt, generated) in enumerate(
            zip(test, prompts, translations)
        ):
            samples_json.append({
                "sample_id": sample.sample_id,
                "source": sample.source,
                "reference": sample.target,
                "translation": generated,
                "prompt": prompt,
            })

            summary.n_samples += 1
            summary.samples.append({
                "sample_id": sample.sample_id,
                "source": sample.source[:120],
                "reference": sample.target[:120],
                "translation": generated[:120],
            })
    except Exception as exc:
        summary.error = f"run_failed: {exc!s}"
        import traceback
        traceback.print_exc()

    # Save translations JSON for this pair
    json_path = pair_dir / f"translations_{tag}.json"
    json_data = {
        "dataset": dataset,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "n_shot": n_shot,
        "max_gen_tokens": max_gen_tokens,
        "samples": samples_json,
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))

    summary.elapsed_s = time.time() - t0
    return summary


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def write_summary(
    out_dir: Path,
    summaries: list[PairSummary],
    total: int,
    done: int,
) -> None:
    """Write the global summary JSON."""
    data = {
        "total_pairs": total,
        "completed": done,
        "pairs": [s.to_dict() for s in summaries],
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate translations across ALL language pairs.",
    )
    parser.add_argument(
        "--dataset",
        choices=["flores", "wmt24pp", "both"],
        default=None,
        help="Dataset(s) to use (not required if --same-lang or --cross-lingual)",
    )
    parser.add_argument(
        "--same-lang",
        action="store_true",
        help="Include same-language pairs (X→X) for all FLORES+ languages",
    )
    parser.add_argument(
        "--cross-lingual",
        action="store_true",
        help="Include 90 cross-lingual pairs among 10 non-English languages "
        "(Czech, Slovak, German, Dutch, Vietnamese, Thai, Chinese, "
        "Japanese, Korean, Hindi)",
    )
    parser.add_argument(
        "--src",
        default="eng_Latn",
        help="Source language(s) for FLORES+, comma-separated "
        "(default: eng_Latn). Ignored for WMT24++ (always en).",
    )
    parser.add_argument(
        "--tgt-filter",
        default=None,
        help="Comma-separated target languages to restrict to (optional)",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-E2B-4bit",
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
    parser.add_argument("--n-samples", type=int, default=-1, help="Test samples per pair")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max gen tokens")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pairs (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        default="translations",
        help="Root output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pairs that already have translation JSON files",
    )
    args = parser.parse_args()

    # --- Discover all pairs ---
    all_pairs: list[tuple[str, str, str]] = []  # (dataset, src, tgt)

    if args.dataset in ("wmt24pp", "both"):
        for src, tgt in get_wmt24pp_pairs():
            all_pairs.append(("wmt24pp", src, tgt))

    if args.dataset in ("flores", "both"):
        src_langs = [s.strip() for s in args.src.split(",")]
        tgt_filter = None
        if args.tgt_filter:
            tgt_filter = {t.strip() for t in args.tgt_filter.split(",")}
        for src, tgt in get_flores_pairs(src_langs, tgt_filter):
            all_pairs.append(("flores", src, tgt))

    if args.same_lang:
        flo_same = get_flores_same_lang_pairs()
        for src, tgt in flo_same:
            all_pairs.append(("flores", src, tgt))
        print(f"  --same-lang: {len(flo_same)} FLORES+ same-language pairs", file=sys.stderr)

        wmt_same = get_wmt24pp_same_lang_pairs()
        for src, tgt in wmt_same:
            all_pairs.append(("wmt24pp", src, tgt))
        print(f"  --same-lang: {len(wmt_same)} WMT24++ same-language pairs", file=sys.stderr)

    if args.cross_lingual:
        cross_pairs = get_cross_lingual_pairs()
        for src, tgt in cross_pairs:
            all_pairs.append(("flores", src, tgt))
        print(f"  --cross-lingual: {len(cross_pairs)} FLORES+ cross-lingual pairs", file=sys.stderr)

        wmt_cross_pairs = get_wmt24pp_cross_lingual_pairs()
        for src, tgt in wmt_cross_pairs:
            all_pairs.append(("wmt24pp", src, tgt))
        print(f"  --cross-lingual: {len(wmt_cross_pairs)} WMT24++ cross-lingual pairs", file=sys.stderr)

    if not all_pairs:
        parser.error(
            "No pairs to process. Specify --dataset and/or --same-lang/--cross-lingual."
        )

    if args.limit:
        all_pairs = all_pairs[: args.limit]

    sanitized_model = "__".join(args.model.split("/"))
    output_dir = args.output_dir + "/" + sanitized_model

    total = len(all_pairs)
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"Total language pairs: {total}", file=sys.stderr)
    print(f"  n_shot={args.n_shot}  n_samples={args.n_samples}  "
          f"max_tokens={args.max_tokens}", file=sys.stderr)
    print(f"  Model: {args.model}", file=sys.stderr)
    print(f"  Output: {output_dir}/", file=sys.stderr)
    print(f"{'=' * 70}\n", file=sys.stderr)

    # --- Load model (once) ---
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

    # --- Output directory ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume check ---
    summaries: list[PairSummary] = []
    global_t0 = time.time()

    for idx, (dataset, src_lang, tgt_lang) in enumerate(all_pairs):
        tag = f"{dataset}_{src_lang}_{tgt_lang}"
        pair_dir = out_dir / tag
        json_path = pair_dir / f"translations_{tag}.json"

        if args.resume and json_path.exists():
            print(
                f"[{idx + 1}/{total}] SKIP {tag} (already done)",
                file=sys.stderr,
            )
            continue

        print(
            f"[{idx + 1}/{total}] {tag}  "
            f"({lang_name(src_lang)} -> {lang_name(tgt_lang)})",
            file=sys.stderr,
        )

        # model, tokenizer, is_vlm, vllm_wrapper, backend_kind = None, None, None, None, None

        summary = run_one_pair(
            model,
            tokenizer,
            dataset,
            src_lang,
            tgt_lang,
            args.n_shot,
            args.n_samples,
            args.max_tokens,
            out_dir,
            is_vlm=is_vlm,
            vllm_wrapper=vllm_wrapper,
            backend_kind=backend_kind,
        )

        summaries.append(summary)

        elapsed = time.time() - global_t0
        done = idx + 1
        remaining = (elapsed / done) * (total - done) if done > 0 else 0
        print(
            f"  -> {summary.n_samples} samples, "
            f"{summary.elapsed_s:.1f}s"
            + (f"  ERROR: {summary.error}" if summary.error else ""),
            file=sys.stderr,
        )
        print(
            f"  Progress: {done}/{total}  "
            f"Pair: {summary.elapsed_s:.1f}s  "
            f"Elapsed: {elapsed:.0f}s  ETA: {remaining:.0f}s",
            file=sys.stderr,
        )

        # Save running summary
        write_summary(out_dir, summaries, total, done)

    # --- Final summary ---
    write_summary(out_dir, summaries, total, total)

    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"DONE. {len(summaries)} pairs processed in "
          f"{time.time() - global_t0:.1f}s", file=sys.stderr)
    print(f"Translations in {out_dir}/", file=sys.stderr)
    print(f"Summary: {out_dir / 'summary.json'}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)


if __name__ == "__main__":
    main()
