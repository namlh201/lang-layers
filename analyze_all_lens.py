"""Analyze all lens output JSONs to classify layers as language-specific
or language-neutral, and detect language switching transitions.

For each layer, the logit lens top-1 predictions are classified by script
(source script, target script, or other).  Layers are then labeled:

  - SOURCE-SPECIFIC: predictions predominantly in source language script
  - TARGET-SPECIFIC: predictions predominantly in target language script
  - NEUTRAL: predictions not clearly in either language script
  - TRANSITIONAL: between specific and neutral

Transition points (source→neutral, neutral→target) are reported
with a confidence probability.

Usage::

    python analyze_all_lens.py
    python analyze_all_lens.py --dir lens_output_all
    python analyze_all_lens.py --per-pair
    python analyze_all_lens.py --workers 16 --batch-size 500
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

LENS_DIR = Path(__file__).parent / "lens_output_all"

SPECIFIC_THRESHOLD = 0.70
NEUTRAL_THRESHOLD = 0.30

SCRIPT_RANGES = {
    "Arabic":     (0x0600, 0x06FF),
    "Cyrillic":   (0x0400, 0x04FF),
    "Bengali":    (0x0980, 0x09FF),
    "CJK":        (0x4E00, 0x9FFF),
    "Hangul":     (0xAC00, 0xD7AF),
    "Latin":      (0x0000, 0x024F),
    "Hiragana":   (0x3040, 0x309F),
    "Katakana":   (0x30A0, 0x30FF),
    "Ethiopic":   (0x1200, 0x137F),
    "Devanagari": (0x0900, 0x097F),
    "Tamil":      (0x0B80, 0x0BFF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Kannada":    (0x0C80, 0x0CFF),
    "Malayalam":  (0x0D00, 0x0D7F),
    "Gurmukhi":   (0x0A00, 0x0A7F),
    "Gujarati":   (0x0A80, 0x0AFF),
    "Greek":      (0x0370, 0x03FF),
    "Hebrew":     (0x0590, 0x05FF),
    "Thai":       (0x0E00, 0x0E7F),
    "Lao":        (0x0E80, 0x0EFF),
    "Khmer":      (0x1780, 0x17FF),
    "Myanmar":    (0x1000, 0x109F),
    "Sinhala":    (0x0D80, 0x0DFF),
    "Tifinagh":   (0x2D30, 0x2D7F),
    "Armenian":   (0x0530, 0x058F),
    "Georgian":   (0x10A0, 0x10FF),
    "Tibetan":    (0x0F00, 0x0FFF),
    "Mongolian":  (0x1800, 0x18AF),
}

SCRIPT_CODE_MAP = {
    "Latn": "Latin", "Cyrl": "Cyrillic", "Arab": "Arabic",
    "Hans": "CJK", "Hant": "CJK", "Jpan": "CJK",
    "Hang": "Hangul", "Beng": "Bengali", "Deva": "Devanagari",
    "Thai": "Thai", "Laoo": "Lao", "Khmr": "Khmer",
    "Mymr": "Myanmar", "Sinh": "Sinhala", "Grek": "Greek",
    "Hebr": "Hebrew", "Ethi": "Ethiopic", "Taml": "Tamil",
    "Telu": "Telugu", "Knda": "Kannada", "Mlym": "Malayalam",
    "Guru": "Gurmukhi", "Gujr": "Gujarati", "Tfng": "Tifinagh",
    "Armn": "Armenian", "Geor": "Georgian", "Tibt": "Tibetan",
    "Mong": "Mongolian",
}

LANG_SCRIPT = {
    "en": "Latin", "fr": "Latin", "de": "Latin", "es": "Latin",
    "it": "Latin", "nl": "Latin", "pl": "Latin", "tr": "Latin",
    "vi": "Latin", "id": "Latin",
    "ar_EG": "Arabic", "ar_SA": "Arabic",
    "bg_BG": "Cyrillic", "ru_RU": "Cyrillic", "uk_UA": "Cyrillic",
    "bn_IN": "Bengali", "hi_IN": "Devanagari", "th_TH": "Thai",
    "el_GR": "Greek", "he_IL": "Hebrew",
    "zh_CN": "CJK", "zh_TW": "CJK", "ja_JP": "CJK", "ko_KR": "Hangul",
    "ca_ES": "Latin", "cs_CZ": "Latin", "da_DK": "Latin",
    "fi_FI": "Latin", "hu_HU": "Latin", "no_NO": "Latin",
    "ro_RO": "Latin", "sk_SK": "Latin", "sv_SE": "Latin",
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def lang_to_script(lang: str) -> str:
    if lang in LANG_SCRIPT:
        return LANG_SCRIPT[lang]
    parts = lang.split("_")
    if len(parts) >= 2:
        script_part = parts[-1]
        if script_part in SCRIPT_CODE_MAP:
            return SCRIPT_CODE_MAP[script_part]
    return "Other"


def detect_script(token: str) -> str:
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        return "Other"
    for char in token:
        if not char.isalpha():
            continue
        cp = ord(char)
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                if name == "CJK":
                    if 0x3040 <= cp <= 0x309F:
                        return "Hiragana"
                    if 0x30A0 <= cp <= 0x30FF:
                        return "Katakana"
                return name
    return "Other"


# --------------------------------------------------------------------------- #
#  Language code normalization (for token lang dist mode)
# --------------------------------------------------------------------------- #

try:
    import langcodes as _langcodes
except ImportError:
    _langcodes = None

MACROLANGUAGE_MEMBERS = {
    "ara": ["arb", "arz", "ary", "arq", "ars", "acm", "acq", "aeb",
            "apc", "apd", "ajp"],
    "zho": ["cmn", "yue", "wuu", "hak", "nan", "hsn", "mnp", "cdo"],
    "nor": ["nob", "nno"],
    "swa": ["swh", "swc"],
    "lav": ["lvs"],
    "est": ["ekk"],
}


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
    lang_dist: dict[str, float], code: str
) -> float:
    """Look up language probability, with macrolanguage fallback."""
    if not code:
        return 0.0
    if code in lang_dist:
        return lang_dist[code]
    if code in MACROLANGUAGE_MEMBERS:
        return sum(
            lang_dist.get(m, 0.0) for m in MACROLANGUAGE_MEMBERS[code]
        )
    return 0.0


# --------------------------------------------------------------------------- #
#  Token language distribution loading
# --------------------------------------------------------------------------- #

_TOKEN_LANG_DIST: dict[str, dict[str, float]] | None = None
_NUM_TOKENS: int | None = None
_NUM_LANGS: int | None = None
_TOP_K: int = 10


def load_token_lang_dist(path: Path) -> dict[str, dict[str, float]]:
    """Load token language distribution JSON.

    Returns a mapping ``token_str -> {iso3_code: prob}``.
    Converts SentencePiece ``▁`` to space to match lens output format.
    Also sets ``_NUM_LANGS`` from ``num_languages_detected`` metadata.
    """
    global _NUM_LANGS
    print(f"Loading token language distribution from {path}...", file=sys.stderr)
    with open(path) as f:
        data = json.load(f)

    _NUM_LANGS = data.get("num_languages_detected")
    if _NUM_LANGS is not None:
        print(
            f"  num_detected_languages: {_NUM_LANGS}",
            file=sys.stderr,
        )

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

    print(
        f"Loaded {n_tokens} tokens with language data",
        file=sys.stderr,
    )
    return mapping


# --------------------------------------------------------------------------- #
#  Multiprocessing worker
# --------------------------------------------------------------------------- #


def _process_file_batch(
    file_paths: list[str],
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, list[int]]],
    dict[tuple[str, str], dict],
    int,
]:
    """Process a batch of lens JSON files in a worker process.

    Returns ``(layer_names, pair_stats, pair_meta, n_files_processed)``.
    """
    layer_names: list[str] = []
    pairs: dict[tuple, dict[str, list]] = {}
    pair_meta: dict[tuple, dict] = {}
    n_processed = 0

    use_lang_dist = _TOKEN_LANG_DIST is not None

    for fpath_str in file_paths:
        try:
            data = json.loads(Path(fpath_str).read_text())
        except Exception:
            continue
        if "tokens" not in data or not data["tokens"]:
            continue

        n_processed += 1
        src_lang = data.get("source_lang", "")
        tgt_lang = data.get("target_lang", "")

        if use_lang_dist:
            src_code = normalize_lang_code(src_lang)
            tgt_code = normalize_lang_code(tgt_lang)
            same_script = src_code == tgt_code
            src_script = src_code
            tgt_script = tgt_code
        else:
            src_script = lang_to_script(src_lang)
            tgt_script = lang_to_script(tgt_lang)
            same_script = src_script == tgt_script

        pair_key = (src_lang, tgt_lang)
        if pair_key not in pair_meta:
            pair_meta[pair_key] = {
                "src_script": src_script,
                "tgt_script": tgt_script,
                "same_script": same_script,
                "n_tokens": 0,
            }
        if pair_key not in pairs:
            pairs[pair_key] = {}

        if not layer_names:
            layer_names = [l["layer"] for l in data["tokens"][0]["layers"]]

        pstats = pairs[pair_key]
        pmeta = pair_meta[pair_key]

        for tok_data in data["tokens"]:
            pmeta["n_tokens"] += 1
            for layer_entry in tok_data["layers"]:
                lname = layer_entry["layer"]

                if use_lang_dist:
                    top_preds = layer_entry.get(f"top{_TOP_K}", [])
                    if not top_preds:
                        top_preds = [{
                            "token": layer_entry.get("top1", ""),
                            "prob": layer_entry.get("prob", 1.0),
                        }]

                    total_lp = sum(t["prob"] for t in top_preds)
                    if total_lp > 0:
                        weights = [t["prob"] / total_lp for t in top_preds]
                    elif _NUM_TOKENS:
                        weights = [1.0 / _NUM_TOKENS] * len(top_preds)
                    else:
                        weights = [1.0 / len(top_preds)] * len(top_preds)

                    p_src = 0.0
                    p_tgt = 0.0
                    p_oth = 0.0

                    for t, w in zip(top_preds, weights):
                        lang_dist = _TOKEN_LANG_DIST.get(t["token"])
                        if lang_dist is not None:
                            ps = _lookup_lang_prob(lang_dist, src_code)
                            pt = _lookup_lang_prob(lang_dist, tgt_code)
                            p_src += w * ps
                            p_tgt += w * pt
                            p_oth += w * (1.0 - ps - pt)
                        elif _NUM_LANGS:
                            uniform = 1.0 / _NUM_LANGS
                            p_src += w * uniform
                            p_tgt += w * uniform
                            p_oth += w * (1.0 - 2.0 * uniform)
                        else:
                            p_oth += w

                    if lname not in pstats:
                        pstats[lname] = [0.0, 0.0, 0.0]
                    pstats[lname][0] += p_src
                    pstats[lname][1] += p_tgt
                    pstats[lname][2] += p_oth
                else:
                    script = detect_script(layer_entry["top1"])

                    if same_script:
                        idx = 0 if script == src_script else 2
                    else:
                        if script == src_script:
                            idx = 0
                        elif script == tgt_script:
                            idx = 1
                        else:
                            idx = 2

                    if lname not in pstats:
                        pstats[lname] = [0, 0, 0]
                    pstats[lname][idx] += 1

    return layer_names, pairs, pair_meta, n_processed


# --------------------------------------------------------------------------- #
#  Core statistics
# --------------------------------------------------------------------------- #


def compute_stats_from_dir(
    lens_dir: Path,
    progress_every: int = 5000,
    workers: int | None = None,
    batch_size: int = 200,
) -> tuple[
    list[str],
    dict[str, list[int]],
    dict[tuple[str, str], dict[str, list[int]]],
    dict[tuple[str, str], dict],
]:
    """Stream over all JSON files and accumulate per-layer script counts.

    Uses a multiprocessing pool: files are batched and dispatched to
    workers via ``imap_unordered``; partial results are merged in the
    main process.

    Returns:
        layer_names: ordered layer labels
        agg: {layer_name: [source_count, target_count, other_count]}
        pairs: {(src, tgt): {layer_name: [s, t, o]}}
        pair_meta: {(src, tgt): {src_script, tgt_script, same_script, n_tokens}}
    """
    files = sorted(
        f for f in lens_dir.rglob("*.json") if f.name != "summary.json"
    )
    total_files = len(files)

    # Batch file paths for dispatch
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
    pairs: dict[tuple, dict[str, list[int]]] = {}
    pair_meta: dict[tuple, dict] = {}

    n_workers = workers or os.cpu_count() or 1
    processed_files = 0

    with mp.Pool(n_workers) as pool:
        for w_layer_names, w_pairs, w_pair_meta, w_n in pool.imap_unordered(
            _process_file_batch, file_batches
        ):
            if not layer_names and w_layer_names:
                layer_names = w_layer_names

            # Merge per-pair stats
            for pair_key, layer_dict in w_pairs.items():
                if pair_key not in pairs:
                    pairs[pair_key] = {}
                pstats = pairs[pair_key]
                for lname, counts in layer_dict.items():
                    if lname not in pstats:
                        pstats[lname] = [0, 0, 0]
                    for j in range(3):
                        pstats[lname][j] += counts[j]

            # Merge pair meta
            for pair_key, wmeta in w_pair_meta.items():
                if pair_key not in pair_meta:
                    pair_meta[pair_key] = {
                        "src_script": wmeta["src_script"],
                        "tgt_script": wmeta["tgt_script"],
                        "same_script": wmeta["same_script"],
                        "n_tokens": 0,
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

    # Compute aggregate from merged per-pair stats
    agg: dict[str, list[int]] = {}
    for layer_dict in pairs.values():
        for lname, counts in layer_dict.items():
            if lname not in agg:
                agg[lname] = [0, 0, 0]
            for j in range(3):
                agg[lname][j] += counts[j]

    return layer_names, agg, pairs, pair_meta


def compute_fractions(
    stats: dict[str, list[int]],
    layer_names: list[str],
) -> tuple[
    list[float], list[float], list[float],
    list[str], list[float],
]:
    """Compute per-layer fractions, classifications, and confidences."""
    p_s: list[float] = []
    p_t: list[float] = []
    p_o: list[float] = []
    classifications: list[str] = []
    confidences: list[float] = []

    for lname in layer_names:
        s, t, o = stats.get(lname, [0, 0, 0])
        total = s + t + o
        if total == 0:
            p_s.append(0.0)
            p_t.append(0.0)
            p_o.append(0.0)
            classifications.append("N/A")
            confidences.append(0.0)
            continue

        ps = s / total
        pt = t / total
        po = o / total
        specificity = max(ps, pt)

        if pt > SPECIFIC_THRESHOLD:
            label, conf = "TARGET-SPECIFIC", pt
        elif ps > SPECIFIC_THRESHOLD:
            label, conf = "SOURCE-SPECIFIC", ps
        elif specificity < NEUTRAL_THRESHOLD:
            label, conf = "NEUTRAL", 1.0 - specificity
        else:
            label = "TRANSITIONAL"
            conf = 1.0 - max(ps, pt, po)

        p_s.append(ps)
        p_t.append(pt)
        p_o.append(po)
        classifications.append(label)
        confidences.append(conf)

    return p_s, p_t, p_o, classifications, confidences


# --------------------------------------------------------------------------- #
#  Transition detection
# --------------------------------------------------------------------------- #


def find_phase_boundaries(
    classifications: list[str],
    layer_names: list[str],
) -> dict:
    """Find source-exit and target-entry layer indices."""
    source_exit_idx: int | None = None
    target_entry_idx: int | None = None

    was_source = False
    for i, c in enumerate(classifications):
        if c == "SOURCE-SPECIFIC":
            was_source = True
        elif was_source and source_exit_idx is None:
            source_exit_idx = i
            break

    start = source_exit_idx if source_exit_idx is not None else 0
    for i in range(start, len(classifications)):
        if classifications[i] == "TARGET-SPECIFIC":
            target_entry_idx = i
            break

    return {
        "source_exit_idx": source_exit_idx,
        "target_entry_idx": target_entry_idx,
        "source_exit": (
            layer_names[source_exit_idx]
            if source_exit_idx is not None else None
        ),
        "target_entry": (
            layer_names[target_entry_idx]
            if target_entry_idx is not None else None
        ),
    }


def detect_transitions(
    classifications: list[str],
    confidences: list[float],
    p_sources: list[float],
    p_targets: list[float],
    layer_names: list[str],
) -> list[dict]:
    """Detect all classification-change transitions."""
    transitions = []
    for i in range(1, len(classifications)):
        if classifications[i] != classifications[i - 1]:
            transitions.append({
                "from": classifications[i - 1],
                "to": classifications[i],
                "layer": layer_names[i],
                "layer_idx": i,
                "confidence": confidences[i],
                "p_source": p_sources[i],
                "p_target": p_targets[i],
            })
    return transitions


# --------------------------------------------------------------------------- #
#  Output
# --------------------------------------------------------------------------- #


def print_layer_table(
    layer_names: list[str],
    p_s: list[float],
    p_t: list[float],
    p_o: list[float],
    classifications: list[str],
    confidences: list[float],
    title: str,
):
    print(f"\n{'=' * 105}")
    print(title)
    print(f"{'=' * 105}")

    header = (
        f"{'Layer':<12} {'Source%':>8} {'Target%':>8} {'Other%':>8}"
        f"  {'Classification':<18} {'Conf':>6}  Bar"
    )
    print(header)
    print("-" * 105)

    for i, lname in enumerate(layer_names):
        ps, pt, po = p_s[i], p_t[i], p_o[i]
        cls = classifications[i]
        conf = confidences[i]

        bar_s = "S" * int(ps * 30)
        bar_t = "T" * int(pt * 30)
        bar_o = "." * int(po * 30)

        print(
            f"{lname:<12} {ps:>7.1%} {pt:>8.1%} {po:>8.1%}"
            f"  {cls:<18} {conf:>5.0%}  |{bar_s}{bar_t}{bar_o}"
        )


def print_transition_analysis(
    layer_names: list[str],
    p_s: list[float],
    p_t: list[float],
    p_o: list[float],
    classifications: list[str],
    confidences: list[float],
    title: str = "TRANSITION ANALYSIS",
):
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")

    transitions = detect_transitions(
        classifications, confidences, p_s, p_t, layer_names
    )

    if not transitions:
        print("  No transitions detected (single phase throughout).")
        return

    print(f"\n  Classification transitions ({len(transitions)}):")
    for t in transitions:
        print(
            f"    {t['from']:<18} -> {t['to']:<18}"
            f"  at {t['layer']:<12}"
            f"  conf={t['confidence']:.0%}"
            f"  (src={t['p_source']:.0%}, tgt={t['p_target']:.0%})"
        )

    b = find_phase_boundaries(classifications, layer_names)

    print(f"\n  Phase boundaries:")
    se = b["source_exit"]
    te = b["target_entry"]

    if se:
        idx = b["source_exit_idx"]
        print(
            f"    Source -> Neutral:  {se:<12}"
            f"  (conf={confidences[idx]:.0%},"
            f" src={p_s[idx]:.0%}, tgt={p_t[idx]:.0%})"
        )
    else:
        print("    Source -> Neutral:  (no source-specific phase found)")

    if te:
        idx = b["target_entry_idx"]
        print(
            f"    Neutral -> Target:  {te:<12}"
            f"  (conf={confidences[idx]:.0%},"
            f" src={p_s[idx]:.0%}, tgt={p_t[idx]:.0%})"
        )
    else:
        print("    Neutral -> Target:  (no target-specific phase found)")

    # Neutrality peak
    specificities = [max(ps, pt) for ps, pt in zip(p_s, p_t)]
    if specificities:
        peak_idx = min(
            range(len(specificities)), key=lambda i: specificities[i]
        )
        print(
            f"\n    Neutrality peak:   {layer_names[peak_idx]:<12}"
            f"  (specificity={specificities[peak_idx]:.0%},"
            f" neutrality={1 - specificities[peak_idx]:.0%})"
        )

    # Phase durations
    if se and te:
        se_idx = b["source_exit_idx"]
        te_idx = b["target_entry_idx"]
        src_dur = se_idx
        ntl_dur = te_idx - se_idx
        tgt_dur = len(layer_names) - te_idx
        print(
            f"\n    Phase durations:   "
            f"source={src_dur} layers, "
            f"neutral={ntl_dur} layers, "
            f"target={tgt_dur} layers"
        )
    elif se and not te:
        se_idx = b["source_exit_idx"]
        print(
            f"\n    Phase durations:   "
            f"source={se_idx} layers, "
            f"neutral={len(layer_names) - se_idx} layers, "
            f"target=0 layers"
        )
    elif not se and te:
        te_idx = b["target_entry_idx"]
        print(
            f"\n    Phase durations:   "
            f"source=0 layers, "
            f"neutral={te_idx} layers, "
            f"target={len(layer_names) - te_idx} layers"
        )


def print_per_pair_summary(
    pair_stats: dict,
    pair_meta: dict,
    layer_names: list[str],
):
    print(f"\n{'=' * 130}")
    print("PER-PAIR TRANSITION SUMMARY")
    print(f"{'=' * 130}")

    header = (
        f"{'Pair':<40} {'Script':<12} {'Src->Ntl':<12} {'Ntl->Tgt':<12}"
        f" {'Src_dur':>7} {'Ntl_dur':>7} {'Tgt_dur':>7}"
        f"  {'Peak_Ntl':<12} {'Peak_N%':>7}  Notes"
    )
    print(header)
    print("-" * 130)

    for pair_key in sorted(pair_stats.keys()):
        src_lang, tgt_lang = pair_key
        meta = pair_meta[pair_key]

        pair_str = f"{src_lang} -> {tgt_lang}"

        if meta["same_script"]:
            print(
                f"{pair_str:<40} {'(same)':<12} {'-':<12} {'-':<12}"
                f" {'-':>7} {'-':>7} {'-':>7}"
                f"  {'-':<12} {'-':>7}  AMBIGUOUS (same script)"
            )
            continue

        stats = pair_stats[pair_key]
        p_s, p_t, p_o, cls, conf = compute_fractions(stats, layer_names)
        b = find_phase_boundaries(cls, layer_names)

        se = b["source_exit"]
        te = b["target_entry"]
        se_idx = b["source_exit_idx"]
        te_idx = b["target_entry_idx"]

        src_dur = str(se_idx) if se_idx is not None else "-"
        ntl_dur = (
            str(te_idx - se_idx)
            if se_idx is not None and te_idx is not None
            else "-"
        )
        tgt_dur = (
            str(len(layer_names) - te_idx)
            if te_idx is not None
            else "-"
        )

        specificities = [max(ps, pt) for ps, pt in zip(p_s, p_t)]
        peak_idx = min(
            range(len(specificities)), key=lambda i: specificities[i]
        )
        peak_layer = layer_names[peak_idx]
        peak_neut = 1 - specificities[peak_idx]

        notes = []
        if se is None:
            notes.append("no src phase")
        if te is None:
            notes.append("no tgt phase")
        if se_idx is not None and te_idx is not None and te_idx <= se_idx:
            notes.append("no neutral phase")

        print(
            f"{pair_str:<40} "
            f"{meta['tgt_script']:<12} "
            f"{(se or '-'):<12} "
            f"{(te or '-'):<12} "
            f"{src_dur:>7} {ntl_dur:>7} {tgt_dur:>7}  "
            f"{peak_layer:<12} {peak_neut:>6.0%}  "
            f"{', '.join(notes)}"
        )


def print_per_pair_detail(
    pair_key: tuple,
    stats: dict,
    meta: dict,
    layer_names: list[str],
):
    src_lang, tgt_lang = pair_key
    p_s, p_t, p_o, cls, conf = compute_fractions(stats, layer_names)

    title = (
        f"DETAIL: {src_lang} -> {tgt_lang} "
        f"(script: {meta['tgt_script']}, tokens: {meta['n_tokens']})"
    )
    print_layer_table(
        layer_names, p_s, p_t, p_o, cls, conf, title
    )
    print_transition_analysis(
        layer_names, p_s, p_t, p_o, cls, conf,
        title=f"TRANSITIONS: {src_lang} -> {tgt_lang}",
    )


# --------------------------------------------------------------------------- #
#  Per-model analysis
# --------------------------------------------------------------------------- #


def run_model_analysis(
    lens_dir: Path,
    model_name: str,
    output_path: Path,
    per_pair: bool = False,
    progress_every: int = 5000,
    workers: int | None = None,
    batch_size: int = 200,
    lang_dist_path: Path | None = None,
    num_tokens: int | None = None,
    top_k: int = 10,
) -> bool:
    """Run analysis for a single model directory, write results to *output_path*.

    Progress messages go to stderr; the full analysis is written to the
    output file.  Returns True if analysis completed, False if no data.
    """
    global _TOKEN_LANG_DIST, _NUM_TOKENS, _TOP_K

    _TOP_K = top_k
    if num_tokens is not None:
        _NUM_TOKENS = num_tokens

    if lang_dist_path is not None and _TOKEN_LANG_DIST is None:
        _TOKEN_LANG_DIST = load_token_lang_dist(lang_dist_path)

    use_lang_dist = _TOKEN_LANG_DIST is not None

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Model: {model_name}", file=sys.stderr)
    print(f"Directory: {lens_dir}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    if use_lang_dist:
        print(f"Mode: language-probability (top{_TOP_K})", file=sys.stderr)
    else:
        print(f"Mode: script-based (top1)", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    files = sorted(
        f for f in lens_dir.rglob("*.json") if f.name != "summary.json"
    )
    if not files:
        print(f"No lens JSON files found in {lens_dir}", file=sys.stderr)
        return False

    print(f"Found {len(files)} files", file=sys.stderr)
    print("Processing files...", file=sys.stderr)

    layer_names, agg, pairs, pair_meta = compute_stats_from_dir(
        lens_dir,
        progress_every=progress_every,
        workers=workers,
        batch_size=batch_size,
    )

    n_pairs = len(pairs)
    n_tokens = sum(m["n_tokens"] for m in pair_meta.values())
    n_distinct_pairs = sum(
        1 for m in pair_meta.values() if not m["same_script"]
    )

    print(
        f"\nLoaded {len(files)} files across {n_pairs} language pairs"
        f" ({n_distinct_pairs} distinct-script, {n_tokens} tokens)",
        file=sys.stderr,
    )
    print(
        f"Layers: {len(layer_names)} "
        f"({layer_names[0]} ... {layer_names[-1]})",
        file=sys.stderr,
    )

    with open(output_path, "w") as out_f:
        old_stdout = sys.stdout
        sys.stdout = out_f
        try:
            print(f"Model: {model_name}")
            print(f"Directory: {lens_dir}")
            print(f"Files: {len(files)}")
            if use_lang_dist:
                print(
                    f"Mode: language-probability (top{_TOP_K}, lang dist:"
                    f" {lang_dist_path.name if lang_dist_path else 'preloaded'})"
                )
            else:
                print("Mode: script-based (top1)")
            print()

            p_s, p_t, p_o, cls, conf = compute_fractions(agg, layer_names)
            print_layer_table(
                layer_names, p_s, p_t, p_o, cls, conf,
                title="LAYER CLASSIFICATION (aggregate across all pairs)",
            )
            print_transition_analysis(
                layer_names, p_s, p_t, p_o, cls, conf,
                title="TRANSITION ANALYSIS (aggregate)",
            )

            print_per_pair_summary(pairs, pair_meta, layer_names)

            if per_pair:
                for pair_key in sorted(pairs.keys()):
                    meta = pair_meta[pair_key]
                    if meta["same_script"]:
                        continue
                    print_per_pair_detail(
                        pair_key, pairs[pair_key], meta, layer_names
                    )

            print(f"\n{'=' * 100}")
            print("SUMMARY")
            print(f"{'=' * 100}")

            b = find_phase_boundaries(cls, layer_names)
            se = b["source_exit"]
            te = b["target_entry"]

            specificities = [max(ps, pt) for ps, pt in zip(p_s, p_t)]
            peak_idx = min(
                range(len(specificities)),
                key=lambda i: specificities[i],
            )

            print(f"\n  Aggregate switching pattern:")
            if se:
                se_idx = b["source_exit_idx"]
                print(
                    f"    Source-specific phase:  "
                    f"{layer_names[0]} .. {layer_names[se_idx - 1]}"
                )
                if te:
                    te_idx = b["target_entry_idx"]
                    print(
                        f"    Neutral phase:          "
                        f"{se} .. {layer_names[te_idx - 1]}"
                    )
                    print(
                        f"    Target-specific phase:  "
                        f"{te} .. {layer_names[-1]}"
                    )
                else:
                    print(
                        f"    Neutral phase:          "
                        f"{se} .. {layer_names[-1]}"
                    )
                    print(f"    Target-specific phase:  (none)")
            else:
                if te:
                    te_idx = b["target_entry_idx"]
                    print(f"    Source-specific phase:  (none)")
                    print(
                        f"    Neutral phase:          "
                        f"{layer_names[0]} .. {layer_names[te_idx - 1]}"
                    )
                    print(
                        f"    Target-specific phase:  "
                        f"{te} .. {layer_names[-1]}"
                    )
                else:
                    print(f"    No clear phase boundaries detected.")

            print(
                f"\n    Neutrality peak:        {layer_names[peak_idx]}"
                f" (neutrality={1 - specificities[peak_idx]:.0%})"
            )

            patterns: dict[str, int] = defaultdict(int)
            for pair_key in pairs:
                meta = pair_meta[pair_key]
                if meta["same_script"]:
                    patterns["ambiguous (same script)"] += 1
                    continue
                stats = pairs[pair_key]
                _, _, _, pcls, _ = compute_fractions(stats, layer_names)
                pb = find_phase_boundaries(pcls, layer_names)
                if pb["source_exit"] and pb["target_entry"]:
                    patterns["source -> neutral -> target"] += 1
                elif pb["target_entry"] and not pb["source_exit"]:
                    patterns["-> target (no source phase)"] += 1
                elif pb["source_exit"] and not pb["target_entry"]:
                    patterns["source -> neutral (no target phase)"] += 1
                else:
                    patterns["other"] += 1

            print(f"\n  Pattern distribution across {n_pairs} pairs:")
            for pattern, count in sorted(
                patterns.items(), key=lambda x: -x[1]
            ):
                print(f"    {pattern:<40} {count} pairs")

            se_indices = []
            te_indices = []
            for pair_key in pairs:
                meta = pair_meta[pair_key]
                if meta["same_script"]:
                    continue
                stats = pairs[pair_key]
                _, _, _, pcls, _ = compute_fractions(stats, layer_names)
                pb = find_phase_boundaries(pcls, layer_names)
                if pb["source_exit_idx"] is not None:
                    se_indices.append(pb["source_exit_idx"])
                if pb["target_entry_idx"] is not None:
                    te_indices.append(pb["target_entry_idx"])

            if se_indices:
                avg_se = sum(se_indices) / len(se_indices)
                min_se = min(se_indices)
                max_se = max(se_indices)
                print(
                    f"\n  Source->Neutral transition across"
                    f" {len(se_indices)} pairs:"
                    f"  avg=layer {avg_se:.1f}, range={min_se}-{max_se}"
                )
            if te_indices:
                avg_te = sum(te_indices) / len(te_indices)
                min_te = min(te_indices)
                max_te = max(te_indices)
                print(
                    f"  Neutral->Target transition across"
                    f" {len(te_indices)} pairs:"
                    f"  avg=layer {avg_te:.1f}, range={min_te}-{max_te}"
                )

            print()
        finally:
            sys.stdout = old_stdout

    print(f"Saved: {output_path}", file=sys.stderr)
    return True


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def _sanitize_model_name(name: str) -> str:
    """Sanitize model name for use in directory/file names."""
    return name.replace("/", "__")


def find_model_dirs(lens_dir: Path) -> list[Path]:
    """Find model subdirectories within *lens_dir*.

    If *lens_dir* contains subdirectories with JSON files, each such
    subdirectory is treated as a model directory.  Otherwise, if
    *lens_dir* itself contains JSON files, it is returned as a single
    model directory.
    """
    children = sorted(d for d in lens_dir.iterdir() if d.is_dir())
    model_dirs = [child for child in children if any(child.rglob("*.json"))]
    if model_dirs:
        return model_dirs
    if any(lens_dir.rglob("*.json")):
        return [lens_dir]
    return []


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze lens outputs for language neutrality vs specificity."
        ),
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Lens output directory (default: lens_output_all). "
        "Can be a parent dir with model subdirs or a single model dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output text files (default: same as --dir).",
    )
    parser.add_argument(
        "--per-pair",
        action="store_true",
        help="Show detailed per-pair tables",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N files (0 to disable)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        ),
        help="Number of worker processes (default: SLURM_CPUS_PER_TASK"
        " or CPU count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Files per batch sent to each worker (default: 200)",
    )
    parser.add_argument(
        "--lang-dist",
        type=str,
        default=None,
        help="Path to token language distribution JSON. When provided,"
        " uses top-5 predictions weighted by per-token language"
        " probabilities instead of script-based classification.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., google/gemma-4-12B). Auto-resolves lens"
        " dir and lang dist path, and loads tokenizer for vocab size.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top predictions per layer to read from lens JSON"
        " (default: 10, reads 'top10' field).",
    )
    args = parser.parse_args()
    print(args)

    num_tokens = None
    model_name_override = None

    if args.model:
        sanitized = _sanitize_model_name(args.model)
        model_name_override = sanitized
        if not args.dir:
            lens_dir = LENS_DIR / sanitized
        else:
            lens_dir = Path(args.dir)
        if not args.lang_dist:
            lang_dist_file = (
                Path(__file__).parent
                / f"token_lang_dist_fineweb_{sanitized}.json"
            )
            if lang_dist_file.exists():
                args.lang_dist = str(lang_dist_file)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        num_tokens = len(tokenizer)
        print(
            f"Tokenizer vocab size: {num_tokens}",
            file=sys.stderr,
        )
        model_dirs = [lens_dir]
    else:
        lens_dir = Path(args.dir) if args.dir else LENS_DIR
        model_dirs = find_model_dirs(lens_dir)

    if not lens_dir.exists():
        print(f"Error: {lens_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else lens_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dirs:
        print(f"No lens JSON files found in {lens_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Found {len(model_dirs)} model "
        f"director{'y' if len(model_dirs) == 1 else 'ies'}",
        file=sys.stderr,
    )
    for d in model_dirs:
        print(f"  - {d.name}", file=sys.stderr)

    lang_dist_path = Path(args.lang_dist) if args.lang_dist else None

    for model_dir in model_dirs:
        model_name = model_name_override or model_dir.name
        suffix = "_langdist" if lang_dist_path else ""
        output_path = output_dir / f"{model_name}_analysis{suffix}.txt"
        run_model_analysis(
            model_dir,
            model_name,
            output_path,
            per_pair=args.per_pair,
            progress_every=args.progress_every,
            workers=args.workers,
            batch_size=args.batch_size,
            lang_dist_path=lang_dist_path,
            num_tokens=num_tokens,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
