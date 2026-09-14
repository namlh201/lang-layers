"""Analyze per-layer working language in LLMs using logit lens + token
language distributions.

For each layer and token position, computes a per-language probability
vector by weighting the **top-10** logit lens predictions by their
per-token language distributions (from a ``token_lang_dist`` JSON).

This reveals:

  * **Language trajectory** -- how the model's "working language" shifts
    layer-by-layer from source to target.
  * **Pivot languages** -- non-source/non-target languages that spike in
    intermediate layers (e.g.  Japanese appearing in Eng->Chinese).
  * **Transition metrics** -- onset, crossover, stable dominance, and
    transition sharpness.

Requires a token language distribution JSON (from
``build_token_lang_dist*.py``).

Usage::

    python analyze_working_language.py \\
        --lang-dist token_lang_dist_dataset.json

    python analyze_working_language.py \\
        --lang-dist token_lang_dist_fineweb_utter-project__EuroLLM-9B-2512.json \\
        --dir lens_output_all/utter-project__EuroLLM-9B-2512

    python analyze_working_language.py \\
        --lang-dist ... --workers 16 --batch-size 500
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    import langcodes as _langcodes
except ImportError:
    _langcodes = None

LENS_DIR = Path(__file__).parent / "lens_output_all"
LANG_DIST_DIR = Path(__file__).parent

MACROLANGUAGE_MEMBERS = {
    "ara": ["arb", "arz", "ary", "arq", "ars", "acm", "acq", "aeb",
            "apc", "apd", "ajp"],
    "zho": ["cmn", "yue", "wuu", "hak", "nan", "hsn", "mnp", "cdo"],
    "nor": ["nob", "nno"],
    "swa": ["swh", "swc"],
    "lav": ["lvs"],
    "est": ["ekk"],
}


# --------------------------------------------------------------------------- #
#  Language code helpers
# --------------------------------------------------------------------------- #


def normalize_lang_code(lang: str) -> str:
    """Convert FLORES+, WMT24++, or ISO 639-3 codes to ISO 639-3."""
    first = lang.split("_")[0]
    if len(first) == 3:
        return first
    if _langcodes is not None:
        try:
            return _langcodes.Language.get(lang).to_alpha3()
        except Exception:
            pass
    return first


def _lookup_lang_prob(
    lang_accum: dict[str, float], code: str
) -> float:
    """Look up summed language probability, with macrolanguage fallback."""
    if not code:
        return 0.0
    if code in lang_accum:
        return lang_accum[code]
    if code in MACROLANGUAGE_MEMBERS:
        return sum(
            lang_accum.get(m, 0.0) for m in MACROLANGUAGE_MEMBERS[code]
        )
    return 0.0


def _is_english_source_dir(dirname: str) -> bool:
    """Check if a pair directory name indicates English as source."""
    if dirname.startswith("flores_"):
        return dirname[len("flores_"):].startswith("eng_Latn_")
    if dirname.startswith("wmt24pp_"):
        return dirname[len("wmt24pp_"):].startswith("en_")
    return False


# --------------------------------------------------------------------------- #
#  Token language distribution loading
# --------------------------------------------------------------------------- #

_TOKEN_LANG_DIST: dict[str, dict[str, float]] | None = None


def load_token_lang_dist(path: Path) -> dict[str, dict[str, float]]:
    """Load token language distribution JSON.

    Returns a mapping ``token_str -> {iso3_code: prob}``.
    Converts SentencePiece ``\u2581`` to space to match lens output format.
    """
    print(f"Loading token language distribution from {path}...", file=sys.stderr)
    with open(path) as f:
        data = json.load(f)

    mapping: dict[str, dict[str, float]] = {}
    n_tokens = 0
    for tid, tdata in data.get("tokens", {}).items():
        token_str = tdata.get("token", "")
        if not token_str:
            continue
        token_str = token_str.replace("\u2581", " ")
        raw_langs = tdata.get("langs", {})
        if not raw_langs:
            continue
        langs: dict[str, float] = {}
        for lang_code, prob in raw_langs.items():
            norm = normalize_lang_code(lang_code)
            langs[norm] = langs.get(norm, 0.0) + prob
        if langs:
            mapping[token_str] = langs
            n_tokens += 1

    print(f"Loaded {n_tokens} tokens with language data", file=sys.stderr)
    return mapping


# --------------------------------------------------------------------------- #
#  Multiprocessing worker
# --------------------------------------------------------------------------- #


def _process_file_batch(
    file_paths: list[str],
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, dict]],
    dict[tuple[str, str], dict],
    int,
]:
    """Process a batch of lens JSON files in a worker process.

    For each file, for each token, for each layer, computes a per-language
    probability vector by weighting the top-10 predictions by their
    per-token language distributions.

    Returns ``(layer_names, pair_stats, pair_meta, n_processed)`` where:
      - pair_stats: ``{(src, tgt): {layer_name: {"langs": {lang: sum}, "n": int}}}``
      - pair_meta: ``{(src, tgt): {"n_tokens": int, "src_code": str, "tgt_code": str}}``
    """
    layer_names: list[str] = []
    pairs: dict[tuple, dict[str, dict]] = {}
    pair_meta: dict[tuple, dict] = {}
    n_processed = 0

    for fpath_str in file_paths:
        try:
            data = json.loads(Path(fpath_str).read_text())
        except Exception:
            continue
        if "tokens" not in data or not data["tokens"]:
            continue
        if not data["tokens"][0].get("layers"):
            continue

        n_processed += 1
        src_lang = data.get("source_lang", "")
        tgt_lang = data.get("target_lang", "")
        src_code = normalize_lang_code(src_lang)
        tgt_code = normalize_lang_code(tgt_lang)

        pair_key = (src_lang, tgt_lang)
        if pair_key not in pairs:
            pairs[pair_key] = {}
        if pair_key not in pair_meta:
            pair_meta[pair_key] = {
                "n_tokens": 0,
                "src_code": src_code,
                "tgt_code": tgt_code,
            }

        if not layer_names:
            layer_names = [
                l["layer"] for l in data["tokens"][0]["layers"]
            ]

        pstats = pairs[pair_key]
        pmeta = pair_meta[pair_key]

        for tok_data in data["tokens"]:
            if not tok_data.get("layers"):
                continue
            pmeta["n_tokens"] += 1

            for layer_entry in tok_data["layers"]:
                lname = layer_entry["layer"]

                top10 = layer_entry.get("top10", [])
                if not top10:
                    top10 = [{
                        "token": layer_entry.get("top1", ""),
                        "prob": layer_entry.get("prob", 1.0),
                    }]

                total_lp = sum(t["prob"] for t in top10)
                if total_lp > 0:
                    weights = [t["prob"] / total_lp for t in top10]
                else:
                    weights = [1.0 / len(top10)] * len(top10)

                if lname not in pstats:
                    pstats[lname] = {"langs": {}, "n": 0}
                ls = pstats[lname]
                ls["n"] += 1

                for t, w in zip(top10, weights):
                    lang_dist = _TOKEN_LANG_DIST.get(t["token"])
                    if lang_dist is not None:
                        for lang_code, prob in lang_dist.items():
                            ls["langs"][lang_code] = (
                                ls["langs"].get(lang_code, 0.0)
                                + w * prob
                            )
                    else:
                        ls["langs"]["__unknown__"] = (
                            ls["langs"].get("__unknown__", 0.0) + w
                        )

    return layer_names, pairs, pair_meta, n_processed


# --------------------------------------------------------------------------- #
#  Statistics aggregation
# --------------------------------------------------------------------------- #


def compute_stats_from_dir(
    lens_dir: Path,
    eng_only: bool = True,
    progress_every: int = 5000,
    workers: int | None = None,
    batch_size: int = 200,
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, dict]],
    dict[tuple[str, str], dict],
]:
    """Stream over all JSON files and accumulate per-layer language stats.

    Returns:
        layer_names: ordered layer labels
        pairs: ``{(src, tgt): {layer_name: {"langs": {...}, "n": int}}}``
        pair_meta: ``{(src, tgt): {"n_tokens": int, "src_code": str, "tgt_code": str}}``
    """
    files = sorted(
        f for f in lens_dir.rglob("*.json")
        if f.name != "summary.json"
        and (not eng_only or _is_english_source_dir(f.parent.name))
    )
    total_files = len(files)

    file_batches: list[list[str]] = []
    batch: list[str] = []
    for f in files:
        batch.append(str(f))
        if len(batch) >= batch_size:
            file_batches.append(batch)
            batch = []
    if batch:
        file_batches.append(batch)

    layer_names: list[str] = []
    pairs: dict[tuple, dict[str, dict]] = {}
    pair_meta: dict[tuple, dict] = {}

    n_workers = workers or os.cpu_count() or 1
    processed_files = 0

    with mp.Pool(n_workers) as pool:
        for w_layers, w_pairs, w_pmeta, w_n in pool.imap_unordered(
            _process_file_batch, file_batches
        ):
            if not layer_names and w_layers:
                layer_names = w_layers

            for pair_key, layer_dict in w_pairs.items():
                if pair_key not in pairs:
                    pairs[pair_key] = {}
                pstats = pairs[pair_key]
                for lname, ldata in layer_dict.items():
                    if lname not in pstats:
                        pstats[lname] = {"langs": {}, "n": 0}
                    target = pstats[lname]
                    target["n"] += ldata["n"]
                    for lang, val in ldata["langs"].items():
                        target["langs"][lang] = (
                            target["langs"].get(lang, 0.0) + val
                        )

            for pair_key, wmeta in w_pmeta.items():
                if pair_key not in pair_meta:
                    pair_meta[pair_key] = {
                        "n_tokens": 0,
                        "src_code": wmeta["src_code"],
                        "tgt_code": wmeta["tgt_code"],
                    }
                pair_meta[pair_key]["n_tokens"] += wmeta["n_tokens"]

            processed_files += w_n
            if progress_every and (
                processed_files // progress_every
                != (processed_files - w_n) // progress_every
            ):
                print(
                    f"  ...{processed_files}/{total_files} files",
                    file=sys.stderr,
                )

    return layer_names, pairs, pair_meta


# --------------------------------------------------------------------------- #
#  Per-layer trajectory computation
# --------------------------------------------------------------------------- #


def compute_layer_traj(
    pair_stats: dict[str, dict],
    src_code: str,
    tgt_code: str,
    layer_names: list[str],
) -> list[dict]:
    """Compute per-layer trajectory from aggregated stats.

    Returns a list of dicts (one per layer) with keys:
        layer, p_src, p_tgt, p_oth, pivots (list of (lang, prob) sorted desc)
    """
    traj: list[dict] = []

    for lname in layer_names:
        ldata = pair_stats.get(lname)
        if ldata is None or ldata["n"] == 0:
            traj.append({
                "layer": lname,
                "p_src": 0.0,
                "p_tgt": 0.0,
                "p_oth": 0.0,
                "pivots": [],
            })
            continue

        n = ldata["n"]
        langs = ldata["langs"]

        p_src = _lookup_lang_prob(langs, src_code) / n
        p_tgt = _lookup_lang_prob(langs, tgt_code) / n
        p_oth = max(0.0, 1.0 - p_src - p_tgt)

        excluded = {src_code, tgt_code, "__unknown__"}
        for m in MACROLANGUAGE_MEMBERS.get(src_code, []):
            excluded.add(m)
        for m in MACROLANGUAGE_MEMBERS.get(tgt_code, []):
            excluded.add(m)
        pivot_items = sorted(
            (
                (lang, val / n)
                for lang, val in langs.items()
                if lang not in excluded
            ),
            key=lambda x: -x[1],
        )
        top_pivots = pivot_items[:5]

        traj.append({
            "layer": lname,
            "p_src": p_src,
            "p_tgt": p_tgt,
            "p_oth": p_oth,
            "pivots": top_pivots,
        })

    return traj


# --------------------------------------------------------------------------- #
#  Transition metrics
# --------------------------------------------------------------------------- #


def compute_transition_metrics(
    traj: list[dict],
) -> dict:
    """Compute transition metrics from a per-layer trajectory.

    Returns dict with:
        source_plateau_end: last layer idx where p_src > 0.90
        transition_onset: first layer idx where p_tgt > 0.30
        crossover: first layer idx where p_tgt > p_src
        stable_dominance: first layer idx where p_tgt > 0.70 and holds
        transition_span: stable - onset (or None)
        sharpness: str
    """
    n = len(traj)
    p_src = [t["p_src"] for t in traj]
    p_tgt = [t["p_tgt"] for t in traj]

    # Source plateau end: last layer where p_src > 0.90
    source_plateau_end = None
    for i in range(n - 1, -1, -1):
        if p_src[i] > 0.90:
            source_plateau_end = i
            break

    # Transition onset: first layer where p_tgt > 0.30
    transition_onset = None
    for i in range(n):
        if p_tgt[i] > 0.30:
            transition_onset = i
            break

    # Crossover: first layer where p_tgt > p_src
    crossover = None
    for i in range(n):
        if p_tgt[i] > p_src[i]:
            crossover = i
            break

    # Stable dominance: first layer where p_tgt > 0.70 and stays
    stable_dominance = None
    for i in range(n):
        if all(p_tgt[j] > 0.70 for j in range(i, n)):
            stable_dominance = i
            break

    # Transition span
    transition_span = None
    if transition_onset is not None and stable_dominance is not None:
        transition_span = stable_dominance - transition_onset

    # Sharpness
    sharpness = "no transition"
    if transition_span is not None:
        if transition_span <= 5:
            sharpness = "sharp"
        elif transition_span <= 10:
            sharpness = "moderate"
        elif transition_span <= 15:
            sharpness = "gradual"
        else:
            sharpness = "very gradual"
    elif crossover is not None:
        sharpness = "incomplete (no stable dominance)"

    return {
        "source_plateau_end": source_plateau_end,
        "transition_onset": transition_onset,
        "crossover": crossover,
        "stable_dominance": stable_dominance,
        "transition_span": transition_span,
        "sharpness": sharpness,
    }


def compute_pivot_summary(
    traj: list[dict],
    src_code: str,
    tgt_code: str,
) -> list[tuple[str, float, int]]:
    """Find pivot languages and their peak probability/layer.

    Returns list of (lang_code, peak_prob, peak_layer_idx) sorted by peak desc.
    """
    excluded = {src_code, tgt_code, "__unknown__"}
    for m in MACROLANGUAGE_MEMBERS.get(src_code, []):
        excluded.add(m)
    for m in MACROLANGUAGE_MEMBERS.get(tgt_code, []):
        excluded.add(m)

    lang_peaks: dict[str, tuple[float, int]] = {}
    for i, t in enumerate(traj):
        for lang, prob in t["pivots"]:
            if lang in excluded:
                continue
            if lang not in lang_peaks or prob > lang_peaks[lang][0]:
                lang_peaks[lang] = (prob, i)

    result = [
        (lang, peak, layer_idx)
        for lang, (peak, layer_idx) in lang_peaks.items()
    ]
    result.sort(key=lambda x: -x[1])
    return result[:10]


# --------------------------------------------------------------------------- #
#  Output formatting
# --------------------------------------------------------------------------- #


def _fmt_layer(idx: int | None) -> str:
    if idx is None:
        return "—"
    return f"L{idx}"


def _fmt_layer(idx: int | None, layer_names: list[str] | None = None) -> str:
    if idx is None:
        return "—"
    if layer_names is not None:
        return layer_names[idx]
    return f"L{idx}"


def print_trajectory_table(
    traj: list[dict],
    layer_names: list[str],
    title: str,
):
    print(f"\n{'=' * 130}")
    print(title)
    print(f"{'=' * 130}")

    header = (
        f"{'Layer':<12} {'p_src':>7} {'p_tgt':>7} {'p_oth':>7}"
        f"  {'Bar (S=src T=tgt .=other)':<32}"
        f"  {'Top pivot languages':<50}"
    )
    print(header)
    print("-" * 130)

    for t in traj:
        ps, pt, po = t["p_src"], t["p_tgt"], t["p_oth"]
        bar_s = "S" * int(ps * 30)
        bar_t = "T" * int(pt * 30)
        bar_o = "." * int(po * 30)
        bar = f"|{bar_s}{bar_t}{bar_o}"
        pivots_str = "  ".join(
            f"{lang}({prob:.3f})" for lang, prob in t["pivots"][:3]
        )
        print(
            f"{t['layer']:<12} {ps:>6.1%} {pt:>7.1%} {po:>7.1%}"
            f"  {bar:<32}"
            f"  {pivots_str:<50}"
        )


def print_transition_metrics(
    metrics: dict,
    layer_names: list[str],
    title: str = "TRANSITION METRICS",
):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    sp = metrics["source_plateau_end"]
    on = metrics["transition_onset"]
    co = metrics["crossover"]
    sd = metrics["stable_dominance"]

    if sp is not None:
        print(
            f"  Source plateau:     {layer_names[0]} - {layer_names[sp]}"
            f"  (p_src > 0.90)"
        )
    else:
        print("  Source plateau:     (none, p_src never > 0.90)")

    if on is not None:
        print(
            f"  Transition onset:   {layer_names[on]:<12}"
            f"  (p_tgt first > 0.30)"
        )
    else:
        print("  Transition onset:   (none, p_tgt never > 0.30)")

    if co is not None:
        print(
            f"  Crossover:          {layer_names[co]:<12}"
            f"  (p_tgt > p_src)"
        )
    else:
        print("  Crossover:          (none, p_tgt never exceeds p_src)")

    if sd is not None:
        print(
            f"  Stable dominance:   {layer_names[sd]:<12}"
            f"  (p_tgt > 0.70 and holds)"
        )
    else:
        print("  Stable dominance:   (none, p_tgt never stably > 0.70)")

    span = metrics["transition_span"]
    if span is not None:
        print(
            f"  Transition span:    {span} layers"
            f"  (onset -> stable)"
        )

    print(f"  Sharpness:          {metrics['sharpness']}")


def print_pivot_summary(
    pivots: list[tuple[str, float, int]],
    layer_names: list[str],
    title: str = "PIVOT LANGUAGES",
):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    if not pivots:
        print("  No pivot languages detected.")
        return

    print(f"\n  {'#':<4} {'Lang':<8} {'Peak prob':>10} {'At layer':<12}")
    print("  " + "-" * 40)
    for rank, (lang, peak, layer_idx) in enumerate(pivots, 1):
        print(
            f"  {rank:<4} {lang:<8} {peak:>10.4f}"
            f"  {layer_names[layer_idx]:<12}"
        )


def print_per_pair_summary(
    pair_stats: dict[tuple[str, str], dict[str, dict]],
    pair_meta: dict[tuple[str, str], dict],
    layer_names: list[str],
):
    print(f"\n{'=' * 140}")
    print("PER-PAIR SUMMARY")
    print(f"{'=' * 140}")

    header = (
        f"{'Pair':<45} {'Tokens':>6}"
        f" {'Onset':>6} {'Cross':>6} {'Stable':>7}"
        f" {'Span':>5} {'Sharpness':<16}"
        f"  {'Top pivot':<30}"
    )
    print(header)
    print("-" * 140)

    for pair_key in sorted(pair_stats.keys()):
        src_lang, tgt_lang = pair_key
        meta = pair_meta[pair_key]
        src_code = meta["src_code"]
        tgt_code = meta["tgt_code"]

        if src_code == tgt_code:
            continue

        pair_str = f"{src_lang} -> {tgt_lang}"
        traj = compute_layer_traj(
            pair_stats[pair_key], src_code, tgt_code, layer_names
        )
        metrics = compute_transition_metrics(traj)
        pivots = compute_pivot_summary(traj, src_code, tgt_code)

        top_pivot_str = (
            f"{pivots[0][0]}({pivots[0][1]:.3f})"
            if pivots else "none"
        )

        print(
            f"{pair_str:<45} {meta['n_tokens']:>6}"
            f" {_fmt_layer(metrics['transition_onset'], layer_names):>6}"
            f" {_fmt_layer(metrics['crossover'], layer_names):>6}"
            f" {_fmt_layer(metrics['stable_dominance'], layer_names):>7}"
            f" {metrics['transition_span'] or '—':>5}"
            f" {metrics['sharpness']:<16}"
            f"  {top_pivot_str:<30}"
        )


# --------------------------------------------------------------------------- #
#  Per-model analysis
# --------------------------------------------------------------------------- #


def run_model_analysis(
    lens_dir: Path,
    model_name: str,
    output_path: Path,
    eng_only: bool = True,
    per_pair: bool = False,
    progress_every: int = 5000,
    workers: int | None = None,
    batch_size: int = 200,
    lang_dist_path: Path | None = None,
) -> bool:
    """Run analysis for a single model directory."""
    global _TOKEN_LANG_DIST

    if lang_dist_path is not None and _TOKEN_LANG_DIST is None:
        _TOKEN_LANG_DIST = load_token_lang_dist(lang_dist_path)

    use_lang_dist = _TOKEN_LANG_DIST is not None

    mode_str = "top-10 lang-dist" if use_lang_dist else "script-based"
    filter_str = "English-source only" if eng_only else "all pairs"

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Model: {model_name}", file=sys.stderr)
    print(f"Directory: {lens_dir}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    print(f"Mode: {mode_str}", file=sys.stderr)
    print(f"Filter: {filter_str}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    files = sorted(
        f for f in lens_dir.rglob("*.json")
        if f.name != "summary.json"
        and (not eng_only or _is_english_source_dir(f.parent.name))
    )
    if not files:
        print(f"No lens JSON files found in {lens_dir}", file=sys.stderr)
        return False

    print(f"Found {len(files)} files", file=sys.stderr)
    print("Processing files...", file=sys.stderr)

    layer_names, pairs, pair_meta = compute_stats_from_dir(
        lens_dir,
        eng_only=eng_only,
        progress_every=progress_every,
        workers=workers,
        batch_size=batch_size,
    )

    n_pairs = len(pairs)
    n_tokens = sum(m["n_tokens"] for m in pair_meta.values())
    n_distinct = sum(
        1 for m in pair_meta.values()
        if m["src_code"] != m["tgt_code"]
    )

    print(
        f"\nLoaded {len(files)} files across {n_pairs} language pairs"
        f" ({n_distinct} distinct-language, {n_tokens} tokens)",
        file=sys.stderr,
    )
    print(
        f"Layers: {len(layer_names)} "
        f"({layer_names[0]} ... {layer_names[-1]})",
        file=sys.stderr,
    )

    # Build aggregate stats across all pairs
    agg_stats: dict[str, dict] = {}
    for pair_key, layer_dict in pairs.items():
        meta = pair_meta[pair_key]
        if meta["src_code"] == meta["tgt_code"]:
            continue
        for lname, ldata in layer_dict.items():
            if lname not in agg_stats:
                agg_stats[lname] = {"langs": {}, "n": 0}
            agg_stats[lname]["n"] += ldata["n"]
            for lang, val in ldata["langs"].items():
                agg_stats[lname]["langs"][lang] = (
                    agg_stats[lname]["langs"].get(lang, 0.0) + val
                )

    # Use aggregate src/tgt codes (all English-source, so src=eng)
    agg_src = "eng"
    agg_tgt = "other"

    with open(output_path, "w") as out_f:
        old_stdout = sys.stdout
        sys.stdout = out_f
        try:
            print(f"Model: {model_name}")
            print(f"Directory: {lens_dir}")
            print(f"Files: {len(files)}")
            print(f"Mode: {mode_str}")
            print(f"Filter: {filter_str}")
            print(f"Language distribution: "
                  f"{lang_dist_path.name if lang_dist_path else 'preloaded'}")
            print()

            # ---- Aggregate trajectory ----
            # For aggregate, we compute p_eng (source) vs p_non-eng
            # by treating eng as src and everything else as "other"
            agg_traj: list[dict] = []
            for lname in layer_names:
                ldata = agg_stats.get(lname)
                if ldata is None or ldata["n"] == 0:
                    agg_traj.append({
                        "layer": lname,
                        "p_src": 0.0,
                        "p_tgt": 0.0,
                        "p_oth": 0.0,
                        "pivots": [],
                    })
                    continue
                n = ldata["n"]
                langs = ldata["langs"]
                p_eng = _lookup_lang_prob(langs, "eng") / n
                p_oth = max(0.0, 1.0 - p_eng)

                # Top non-English languages
                pivot_items = [
                    (lang, val / n)
                    for lang, val in langs.items()
                    if lang not in ("eng", "__unknown__")
                ]
                pivot_items.sort(key=lambda x: -x[1])
                agg_traj.append({
                    "layer": lname,
                    "p_src": p_eng,
                    "p_tgt": p_oth,
                    "p_oth": 0.0,
                    "pivots": pivot_items[:5],
                })

            print_trajectory_table(
                agg_traj,
                layer_names,
                title=(
                    "AGGREGATE LANGUAGE TRAJECTORY"
                    f" ({n_tokens} tokens, {n_distinct} pairs)"
                    "\n  p_src = English (source),  p_tgt = non-English"
                    " (target+other)"
                ),
            )

            # ---- Per-pair analysis ----
            print_per_pair_summary(pairs, pair_meta, layer_names)

            if per_pair:
                for pair_key in sorted(pairs.keys()):
                    src_lang, tgt_lang = pair_key
                    meta = pair_meta[pair_key]
                    if meta["src_code"] == meta["tgt_code"]:
                        continue

                    traj = compute_layer_traj(
                        pairs[pair_key],
                        meta["src_code"],
                        meta["tgt_code"],
                        layer_names,
                    )
                    metrics = compute_transition_metrics(traj)
                    pivots = compute_pivot_summary(
                        traj, meta["src_code"], meta["tgt_code"]
                    )

                    pair_str = f"{src_lang} -> {tgt_lang}"
                    print_trajectory_table(
                        traj,
                        layer_names,
                        title=(
                            f"LANGUAGE TRAJECTORY: {pair_str}"
                            f"  ({meta['n_tokens']} tokens,"
                            f" src={meta['src_code']},"
                            f" tgt={meta['tgt_code']})"
                        ),
                    )
                    print_transition_metrics(
                        metrics,
                        layer_names,
                        title=f"TRANSITION METRICS: {pair_str}",
                    )
                    print_pivot_summary(
                        pivots,
                        layer_names,
                        title=f"PIVOT LANGUAGES: {pair_str}",
                    )

            # ---- Summary ----
            print(f"\n{'=' * 100}")
            print("SUMMARY")
            print(f"{'=' * 100}")

            # Aggregate transition stats
            all_onsets: list[int] = []
            all_crosses: list[int] = []
            all_stables: list[int] = []
            all_spans: list[int] = []

            for pair_key in sorted(pairs.keys()):
                meta = pair_meta[pair_key]
                if meta["src_code"] == meta["tgt_code"]:
                    continue
                traj = compute_layer_traj(
                    pairs[pair_key],
                    meta["src_code"],
                    meta["tgt_code"],
                    layer_names,
                )
                m = compute_transition_metrics(traj)
                if m["transition_onset"] is not None:
                    all_onsets.append(m["transition_onset"])
                if m["crossover"] is not None:
                    all_crosses.append(m["crossover"])
                if m["stable_dominance"] is not None:
                    all_stables.append(m["stable_dominance"])
                if m["transition_span"] is not None:
                    all_spans.append(m["transition_span"])

            def _stats(vals: list[int], name: str) -> None:
                if not vals:
                    print(f"  {name}: no data")
                    return
                vs = sorted(vals)
                mean = sum(vals) / len(vals)
                median = vs[len(vs) // 2]
                print(
                    f"  {name}: mean=L{mean:.1f}  median=L{median}"
                    f"  range=L{min(vals)}-L{max(vals)}"
                    f"  (n={len(vals)})"
                )

            print(f"\n  Across {len(all_onsets)} pairs with transitions:")
            _stats(all_onsets, "Transition onset ")
            _stats(all_crosses, "Crossover        ")
            _stats(all_stables, "Stable dominance ")
            _stats(all_spans, "Transition span  ")

            # Sharpness distribution
            sharpness_counts: dict[str, int] = defaultdict(int)
            for pair_key in sorted(pairs.keys()):
                meta = pair_meta[pair_key]
                if meta["src_code"] == meta["tgt_code"]:
                    continue
                traj = compute_layer_traj(
                    pairs[pair_key],
                    meta["src_code"],
                    meta["tgt_code"],
                    layer_names,
                )
                m = compute_transition_metrics(traj)
                sharpness_counts[m["sharpness"]] += 1

            print(f"\n  Sharpness distribution:")
            for sharp, cnt in sorted(
                sharpness_counts.items(), key=lambda x: -x[1]
            ):
                print(f"    {sharp:<24} {cnt} pairs")

            # Top pivot languages across all pairs
            all_pivots: dict[str, tuple[float, str]] = {}
            for pair_key in sorted(pairs.keys()):
                meta = pair_meta[pair_key]
                if meta["src_code"] == meta["tgt_code"]:
                    continue
                traj = compute_layer_traj(
                    pairs[pair_key],
                    meta["src_code"],
                    meta["tgt_code"],
                    layer_names,
                )
                pivots = compute_pivot_summary(
                    traj, meta["src_code"], meta["tgt_code"]
                )
                for lang, peak, layer_idx in pivots[:3]:
                    if lang not in all_pivots or peak > all_pivots[lang][0]:
                        pair_str = f"{pair_key[0]} -> {pair_key[1]}"
                        all_pivots[lang] = (peak, pair_str)

            if all_pivots:
                print(f"\n  Top pivot languages across all pairs:")
                sorted_pivots = sorted(
                    all_pivots.items(), key=lambda x: -x[1][0]
                )
                for lang, (peak, pair_str) in sorted_pivots[:10]:
                    print(
                        f"    {lang:<8} peak={peak:.4f}"
                        f"  (in {pair_str})"
                    )

            print()
        finally:
            sys.stdout = old_stdout

    print(f"Saved: {output_path}", file=sys.stderr)
    return True


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def _sanitize_model_name(model: str) -> str:
    """Convert e.g. 'google/gemma-4-12B' to 'google__gemma-4-12B'."""
    return model.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-layer working language in LLMs using logit lens"
            " top-10 predictions weighted by per-token language"
            " distributions."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name, e.g. google/gemma-4-12B.",
    )
    parser.add_argument(
        "--eng-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only analyze English-source pairs (default: True). "
        "Use --no-eng-only to include all pairs.",
    )
    parser.add_argument(
        "--per-pair",
        action="store_true",
        help="Show detailed per-pair trajectory tables.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N files (0 to disable).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        ),
        help="Number of worker processes (default: SLURM_CPUS_PER_TASK"
        " or CPU count).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Files per batch sent to each worker (default: 200).",
    )
    args = parser.parse_args()

    sanitized = _sanitize_model_name(args.model)
    lens_dir = LENS_DIR / sanitized
    lang_dist_path = LANG_DIST_DIR / f"token_lang_dist_fineweb_{sanitized}.json"

    if not lens_dir.exists():
        print(f"Error: lens dir {lens_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    if not lang_dist_path.exists():
        print(
            f"Error: token lang dist {lang_dist_path} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = lens_dir / f"{sanitized}_working_lang_eng_src_only.txt"
    if not args.eng_only:
        output_path = lens_dir / f"{sanitized}_working_lang.txt"

    run_model_analysis(
        lens_dir,
        sanitized,
        output_path,
        eng_only=args.eng_only,
        per_pair=args.per_pair,
        progress_every=args.progress_every,
        workers=args.workers,
        batch_size=args.batch_size,
        lang_dist_path=lang_dist_path,
    )


if __name__ == "__main__":
    main()
