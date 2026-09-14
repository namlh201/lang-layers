"""Detect language switching points between layers in logit lens results.

For each generated token, the logit lens produces a top-1 prediction at
every layer.  By classifying each prediction's script (Latin = source
language, target script = target language), we can pinpoint **exactly which
layer** the model switches from "thinking in English" to "outputting the
target language".

Key metrics per language pair:
  * **switch_layer**: the layer where target-script probability first
    exceeds Latin-script probability (the crossover point).
  * **switch_layer_first**: the first layer where any target-script token
    appears in top-1 (earliest emergence).
  * **switch_layer_stable**: the layer after which target script stays
    dominant for the rest of the network (consolidation).
  * **Latin plateau**: the range of layers where Latin dominates >90%
    (the "language-neutral" phase).

When ``--lang-dist`` is provided, the analysis uses per-token language
probability distributions (from a token_lang_dist JSON) weighted by the
logit-lens top-5 probabilities, instead of script detection on top-1.

Usage::

    python analyze_switching.py
    python analyze_switching.py --lang-dist token_lang_dist_fineweb.json
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import langcodes as _langcodes
except ImportError:
    _langcodes = None

LENS_DIR = Path(__file__).parent / "lens_output_all"

MACROLANGUAGE_MEMBERS = {
    "ara": ["arb", "arz", "ary", "arq", "ars", "acm", "acq", "aeb",
            "apc", "apd", "ajp"],
    "zho": ["cmn", "yue", "wuu", "hak", "nan", "hsn", "mnp", "cdo"],
    "nor": ["nob", "nno"],
    "swa": ["swh", "swc"],
    "lav": ["lvs"],
    "est": ["ekk"],
}

SCRIPT_RANGES = {
    "Arabic": (0x0600, 0x06FF),
    "Cyrillic": (0x0400, 0x04FF),
    "Bengali": (0x0980, 0x09FF),
    "CJK": (0x4E00, 0x9FFF),
    "Hangul": (0xAC00, 0xD7AF),
    "Latin": (0x0000, 0x024F),
    "Hiragana": (0x3040, 0x309F),
    "Katakana": (0x30A0, 0x30FF),
    "Ethiopic": (0x1200, 0x137F),
    "Devanagari": (0x0900, 0x097F),
    "Tamil": (0x0B80, 0x0BFF),
    "Greek": (0x0370, 0x03FF),
    "Hebrew": (0x0590, 0x05FF),
    "Thai": (0x0E00, 0x0E7F),
    "Armenian": (0x0530, 0x058F),
    "Georgian": (0x10A0, 0x10FF),
    "Gujarati": (0x0A80, 0x0AFF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}

TARGET_SCRIPT = {
    "ar_EG": "Arabic", "ar_SA": "Arabic", "fa_IR": "Arabic",
    "bg_BG": "Cyrillic", "ru_RU": "Cyrillic", "sr_RS": "Cyrillic",
    "uk_UA": "Cyrillic",
    "bn_IN": "Bengali",
    "cmn_Hans": "CJK", "ja_JP": "CJK", "jpn_Jpan": "CJK",
    "ko_KR": "Hangul", "kor_Hang": "Hangul",
    "el_GR": "Greek",
    "he_IL": "Hebrew",
    "hi_IN": "Devanagari", "mr_IN": "Devanagari",
    "gu_IN": "Gujarati",
    "pa_IN": "Gurmukhi",
    "kn_IN": "Kannada",
    "ml_IN": "Malayalam",
    "ta_IN": "Tamil",
    "th_TH": "Thai",
    "ca_ES": "Latin", "cs_CZ": "Latin", "da_DK": "Latin",
    "de_DE": "Latin", "es_MX": "Latin", "et_EE": "Latin",
    "fi_FI": "Latin", "fil_PH": "Latin", "fr_CA": "Latin",
    "fr_FR": "Latin", "hr_HR": "Latin", "hu_HU": "Latin",
    "id_ID": "Latin", "is_IS": "Latin", "it_IT": "Latin",
    "lt_LT": "Latin", "lv_LV": "Latin", "nl_NL": "Latin",
    "no_NO": "Latin", "pl_PL": "Latin", "pt_BR": "Latin",
    "pt_PT": "Latin", "ro_RO": "Latin", "sk_SK": "Latin",
    "sl_SI": "Latin", "sv_SE": "Latin", "sw_KE": "Latin",
}


def detect_script(token: str) -> str:
    for char in token.strip():
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


def normalize_lang_code(lang: str) -> str:
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
    if not code:
        return 0.0
    if code in lang_dist:
        return lang_dist[code]
    if code in MACROLANGUAGE_MEMBERS:
        return sum(
            lang_dist.get(m, 0.0) for m in MACROLANGUAGE_MEMBERS[code]
        )
    return 0.0


_TOKEN_LANG_DIST: dict[str, dict[str, float]] | None = None
_NUM_TOKENS: int | None = None
_NUM_LANGS: int | None = None
_TOP_K: int = 10


def load_token_lang_dist(path: Path) -> dict[str, dict[str, float]]:
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
    print(f"Loaded {n_tokens} tokens with language data", file=sys.stderr)
    return mapping


def load_all_files(lens_dir: Path | None = None):
    search_dir = lens_dir or LENS_DIR
    files = sorted(
        f
        for f in glob.glob(str(search_dir / "**" / "*.json"), recursive=True)
        if "summary.json" not in f
    )
    results = []
    for fpath in files:
        with open(fpath) as f:
            results.append(json.load(f))
    return results


def _compute_lang_probs(
    layer_entry: dict,
    src_code: str,
    tgt_code: str,
) -> tuple[float, float, float]:
    """Compute (p_source, p_target, p_other) for a layer entry using top-k."""
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

    return p_src, p_tgt, p_oth


def analyze_switching(
    lang_dist_path: Path | None = None,
    lens_dir: Path | None = None,
    num_tokens: int | None = None,
    top_k: int = 10,
):
    global _TOKEN_LANG_DIST, _NUM_TOKENS, _TOP_K

    _TOP_K = top_k
    if num_tokens is not None:
        _NUM_TOKENS = num_tokens

    if lang_dist_path is not None and _TOKEN_LANG_DIST is None:
        _TOKEN_LANG_DIST = load_token_lang_dist(lang_dist_path)

    use_lang_dist = _TOKEN_LANG_DIST is not None

    all_data = load_all_files(lens_dir)
    print(f"Loaded {len(all_data)} JSON files")
    if use_lang_dist:
        print(f"Mode: language-probability (top{_TOP_K})")
    else:
        print("Mode: script-based (top1)")
    print()

    # Separate by model (17 layers = Llama-1B, 36 layers = gemma-4-12B)
    small_model = [
        d for d in all_data
        if d["tokens"] and len(d["tokens"][0]["layers"]) <= 20
    ]
    large_model = [
        d for d in all_data
        if d["tokens"] and len(d["tokens"][0]["layers"]) > 20
    ]
    print(f"Small model (<=20 layers): {len(small_model)} files")
    print(f"Large model (>20 layers): {len(large_model)} files")

    # Focus on the large model
    dataset = large_model
    if not dataset:
        dataset = small_model

    layer_names = [
        ly["layer"] for ly in dataset[0]["tokens"][0]["layers"]
    ]
    n_layers = len(layer_names)

    # Per-file per-token: track script or language probs at each layer
    pair_data: dict[str, list] = defaultdict(list)

    # Global per-layer distribution
    if use_lang_dist:
        layer_lang_probs: dict[str, dict] = {
            lname: {"src_sum": 0.0, "tgt_sum": 0.0, "oth_sum": 0.0, "n": 0}
            for lname in layer_names
        }
    else:
        non_latin_layer_scripts: dict[str, defaultdict] = {
            lname: defaultdict(int) for lname in layer_names
        }
    total_tokens = 0

    for data in dataset:
        tgt = data["target_lang"]
        src = data.get("source_lang", "")

        if use_lang_dist:
            src_code = normalize_lang_code(src)
            tgt_code = normalize_lang_code(tgt)
            if src_code == tgt_code:
                continue
        else:
            tgt_script = TARGET_SCRIPT.get(tgt, "Other")
            if tgt_script == "Latin":
                continue

        for tok_data in data["tokens"]:
            total_tokens += 1

            if use_lang_dist:
                # Track (p_src, p_tgt, p_oth) at each layer
                probs_per_layer: list[tuple[float, float, float]] = []
                for layer_entry in tok_data["layers"]:
                    p_src, p_tgt, p_oth = _compute_lang_probs(
                        layer_entry, src_code, tgt_code
                    )
                    probs_per_layer.append((p_src, p_tgt, p_oth))
                    lp = layer_lang_probs[layer_entry["layer"]]
                    lp["src_sum"] += p_src
                    lp["tgt_sum"] += p_tgt
                    lp["oth_sum"] += p_oth
                    lp["n"] += 1

                # switch_first: first layer where p_target > 0.3
                switch_first = None
                for i, (ps, pt, po) in enumerate(probs_per_layer):
                    if pt > 0.3:
                        switch_first = i
                        break

                # switch_crossover: first layer where p_target > p_source
                switch_crossover = None
                for i, (ps, pt, po) in enumerate(probs_per_layer):
                    if pt > ps:
                        switch_crossover = i
                        break

                # switch_stable: after which p_target >= p_source
                switch_stable = None
                for i in range(len(probs_per_layer) - 1, -1, -1):
                    if probs_per_layer[i][0] > probs_per_layer[i][1]:
                        switch_stable = i + 1
                        break
                if switch_stable is None:
                    switch_stable = 0
                if switch_stable >= len(probs_per_layer):
                    switch_stable = len(probs_per_layer) - 1

                pair_data[tgt].append(
                    (switch_first, switch_crossover, switch_stable,
                     probs_per_layer)
                )
            else:
                # Script-based mode (existing code)
                scripts_per_layer: list[str] = []
                for layer_entry in tok_data["layers"]:
                    top1 = layer_entry["top1"]
                    script = detect_script(top1)
                    scripts_per_layer.append(script)
                    non_latin_layer_scripts[layer_entry["layer"]][script] += 1

                switch_first = None
                for i, s in enumerate(scripts_per_layer):
                    if s == tgt_script:
                        switch_first = i
                        break

                switch_crossover = None
                for i in range(len(scripts_per_layer)):
                    if scripts_per_layer[i] == tgt_script:
                        if i + 1 < len(scripts_per_layer) and (
                            scripts_per_layer[i + 1] == tgt_script
                            or i >= len(scripts_per_layer) - 1
                        ):
                            switch_crossover = i
                            break

                switch_stable = None
                for i in range(len(scripts_per_layer) - 1, -1, -1):
                    if scripts_per_layer[i] != tgt_script:
                        switch_stable = i + 1
                        break
                if switch_stable is None:
                    switch_stable = 0
                if switch_stable >= len(scripts_per_layer):
                    switch_stable = len(scripts_per_layer) - 1

                pair_data[tgt].append(
                    (switch_first, switch_crossover, switch_stable,
                     scripts_per_layer)
                )

    # ---- Print global layer-by-layer distribution ----
    print(f"\n{'=' * 100}")
    if use_lang_dist:
        print(
            f"GLOBAL LAYER-BY-LAYER LANGUAGE DISTRIBUTION "
            f"({total_tokens} tokens)"
        )
    else:
        print(
            f"GLOBAL LAYER-BY-LAYER SCRIPT DISTRIBUTION "
            f"(non-Latin targets, {total_tokens} tokens)"
        )
    print("=" * 100)

    if use_lang_dist:
        header = (
            f"{'Layer':<12} {'Source%':>8} {'Target%':>8} {'Other%':>8}"
            f"  {'Dom':>6} {'Phase':<12}"
        )
        print(f"\n{header}")
        print("-" * len(header))

        for lname in layer_names:
            lp = layer_lang_probs[lname]
            n = lp["n"]
            if n == 0:
                continue
            ps = lp["src_sum"] / n
            pt = lp["tgt_sum"] / n
            po = lp["oth_sum"] / n
            dom = max(ps, pt)
            if pt > 0.7:
                phase = "TARGET"
            elif ps > 0.7:
                phase = "SOURCE"
            elif dom < 0.3:
                phase = "NEUTRAL"
            else:
                phase = "EMERGING"
            bar_s = "S" * int(ps * 30)
            bar_t = "T" * int(pt * 30)
            bar_o = "." * int(po * 30)
            print(
                f"{lname:<12} {ps:>7.1%} {pt:>8.1%} {po:>8.1%}"
                f"  {dom:>5.0%} {phase:<12} |{bar_s}{bar_t}{bar_o}"
            )
    else:
        script_order = [
            "Arabic", "Cyrillic", "CJK", "Hangul",
            "Greek", "Hebrew", "Devanagari", "Bengali",
            "Kannada", "Malayalam", "Gujarati", "Gurmukhi",
            "Tamil", "Thai", "Latin", "Other",
        ]
        header = f"{'Layer':<12}"
        for s in script_order:
            header += f"{s:>9}"
        header += f"  {'Dom%':>6} {'Phase':<12}"
        print(f"\n{header}")
        print("-" * len(header))

        for lname in layer_names:
            s = non_latin_layer_scripts[lname]
            total = sum(s.values())
            if total == 0:
                continue
            row = f"{lname:<12}"
            target_count = 0
            latin_count = 0
            for sn in script_order:
                c = s.get(sn, 0)
                pct = c / total
                row += f"{pct:>8.0%}"
                if sn not in ("Latin", "Other"):
                    target_count += c
                if sn == "Latin":
                    latin_count += c
            dom_pct = target_count / total
            if dom_pct > 0.7:
                phase = "TARGET"
            elif dom_pct > 0.4:
                phase = "EMERGING"
            elif latin_count / total > 0.7:
                phase = "LATIN"
            else:
                phase = "NOISE"
            row += f"  {dom_pct:>5.0%} {phase:<12}"
            print(row)

    # ---- Per-language switching analysis ----
    print(f"\n\n{'=' * 100}")
    print("LANGUAGE SWITCHING POINTS PER TARGET LANGUAGE")
    print("=" * 100)

    label_col = "Target" if not use_lang_dist else "Target"
    extra_col = "Script" if not use_lang_dist else "Code"
    plateau_label = "Latin plateau" if not use_lang_dist else "Src plateau"

    print(
        f"\n{label_col:<10} {extra_col:<10} {'#Tok':>5} "
        f"{'First':>6} {'Cross':>6} {'Stable':>7}  "
        f"{plateau_label:<20} {'Switch span':<20}"
    )
    print("-" * 100)

    all_firsts: list[int] = []
    all_crosses: list[int] = []
    all_stables: list[int] = []

    for tgt in sorted(pair_data.keys()):
        entries = pair_data[tgt]

        if use_lang_dist:
            tgt_code = normalize_lang_code(tgt)
            extra = tgt_code
        else:
            tgt_script = TARGET_SCRIPT.get(tgt, "Other")
            extra = tgt_script

        firsts = [e[0] for e in entries if e[0] is not None]
        crosses = [e[1] for e in entries if e[1] is not None]
        stables = [e[2] for e in entries if e[2] is not None]

        avg_first = sum(firsts) / len(firsts) if firsts else None
        avg_cross = sum(crosses) / len(crosses) if crosses else None
        avg_stable = sum(stables) / len(stables) if stables else None

        all_firsts.extend(firsts)
        all_crosses.extend(crosses)
        all_stables.extend(stables)

        # Plateau: layers where source/Latin dominates > 90%
        plateau_layers: list[int] = []
        for li, lname in enumerate(layer_names):
            if use_lang_dist:
                lp = layer_lang_probs[lname]
                n = lp["n"]
                if n > 0 and lp["src_sum"] / n > 0.9:
                    plateau_layers.append(li)
            else:
                s = non_latin_layer_scripts[lname]
                latin = s.get("Latin", 0)
                total = sum(s.values())
                if total > 0 and latin / total > 0.9:
                    plateau_layers.append(li)

        if plateau_layers:
            plateau = f"L{plateau_layers[0]}-L{plateau_layers[-1]}"
        else:
            plateau = "none"

        if avg_first is not None and avg_stable is not None:
            span = f"L{int(avg_first)} -> L{int(avg_stable)}"
        else:
            span = "?"

        def fmt(v):
            return f"L{int(v)}" if v is not None else "—"

        print(
            f"{tgt:<10} {extra:<10} {len(entries):>5} "
            f"{fmt(avg_first):>6} {fmt(avg_cross):>6} "
            f"{fmt(avg_stable):>7}  "
            f"{plateau:<20} {span:<20}"
        )

    # ---- Aggregate switching statistics ----
    print(f"\n\n{'=' * 100}")
    print("AGGREGATE SWITCHING STATISTICS")
    print("=" * 100)

    def stats(vals: list[int], name: str) -> None:
        if not vals:
            print(f"  {name}: no data")
            return
        vals_sorted = sorted(vals)
        mean = sum(vals) / len(vals)
        median = vals_sorted[len(vals_sorted) // 2]
        p25 = vals_sorted[len(vals_sorted) // 4]
        p75 = vals_sorted[3 * len(vals_sorted) // 4]
        print(
            f"  {name}: mean=L{mean:.1f}  median=L{median}  "
            f"p25=L{p25}  p75=L{p75}  min=L{min(vals)}  max=L{max(vals)}"
        )

    tokens_label = "non-Latin targets" if not use_lang_dist else "distinct-lang pairs"
    print(f"\nAcross all {len(all_firsts)} tokens ({tokens_label}):")
    stats(all_firsts, "First emergence ")
    stats(all_crosses, "Crossover       ")
    stats(all_stables, "Stable dominance")

    # ---- Switching distribution histogram ----
    print(f"\n\n{'=' * 100}")
    print("SWITCHING LAYER DISTRIBUTION (crossover point)")
    print("=" * 100)

    print(f"\n{'Layer':<12} {'Count':>6} {'%':>6}  Bar")
    print("-" * 60)
    max_count = max(
        all_crosses.count(i) for i in range(n_layers)
    ) if all_crosses else 1
    for i in range(n_layers):
        c = all_crosses.count(i)
        pct = c / len(all_crosses) * 100 if all_crosses else 0
        bar = "#" * int(c / max_count * 40) if max_count > 0 else ""
        if c > 0:
            print(f"L{i:<11} {c:>6} {pct:>5.1f}%  {bar}")

    # ---- Phase boundaries ----
    print(f"\n\n{'=' * 100}")
    print("PHASE BOUNDARIES (aggregated)")
    print("=" * 100)

    if use_lang_dist:
        src_fracs = []
        tgt_fracs = []
        for lname in layer_names:
            lp = layer_lang_probs[lname]
            n = lp["n"]
            if n > 0:
                src_fracs.append(lp["src_sum"] / n)
                tgt_fracs.append(lp["tgt_sum"] / n)
            else:
                src_fracs.append(0.0)
                tgt_fracs.append(0.0)

        src_plateau_end = None
        for i in range(n_layers - 1, -1, -1):
            if src_fracs[i] > 0.9:
                src_plateau_end = i
                break

        src_drop_layer = None
        for i in range(n_layers):
            if src_fracs[i] < 0.5:
                src_drop_layer = i
                break

        target_crossover_layer = None
        for i in range(n_layers):
            if tgt_fracs[i] > 0.5:
                target_crossover_layer = i
                break

        p_label = "Source"
        print(f"\n  {p_label} plateau end ({p_label} < 90%):  L{src_plateau_end}")
        print(f"  {p_label} drop below 50%:              L{src_drop_layer}")
        print(f"  Target crosses 50%:                 L{target_crossover_layer}")

        if src_plateau_end is not None and target_crossover_layer is not None:
            print(f"\n  PHASE 1 ({p_label}-dominant embed):  "
                  f"L0 - L{src_plateau_end - 1 if src_plateau_end > 0 else 0}")
            print(f"  PHASE 2 ({p_label} plateau / neutral): "
                  f"L{src_plateau_end} - L{target_crossover_layer - 1}")
            print(f"  PHASE 3 (Target recovery):         "
                  f"L{target_crossover_layer} - L{n_layers - 1}")
    else:
        # Existing script-based phase boundaries
        latin_fracs = []
        for lname in layer_names:
            s = non_latin_layer_scripts[lname]
            total = sum(s.values())
            latin = s.get("Latin", 0) / total if total > 0 else 0
            latin_fracs.append(latin)

        target_fracs = []
        for lname in layer_names:
            s = non_latin_layer_scripts[lname]
            total = sum(s.values())
            tgt_c = sum(
                v for k, v in s.items() if k not in ("Latin", "Other")
            )
            target_fracs.append(tgt_c / total if total > 0 else 0)

        target_crossover_layer = None
        for i in range(n_layers):
            if target_fracs[i] > 0.5:
                target_crossover_layer = i
                break

        latin_drop_layer = None
        for i in range(n_layers):
            if latin_fracs[i] < 0.5:
                latin_drop_layer = i
                break

        latin_plateau_end = None
        for i in range(n_layers - 1, -1, -1):
            if latin_fracs[i] > 0.9:
                latin_plateau_end = i
                break

        print(f"\n  Latin plateau end (Latin < 90%):  L{latin_plateau_end}")
        print(f"  Latin drop below 50%:              L{latin_drop_layer}")
        print(f"  Target crosses 50%:                 L{target_crossover_layer}")

        if latin_plateau_end is not None and target_crossover_layer is not None:
            print(f"\n  PHASE 1 (Target-dominant embed):  "
                  f"L0 - L{latin_plateau_end - 1 if latin_plateau_end > 0 else 0}")
            print(f"  PHASE 2 (Latin plateau / neutral): "
                  f"L{latin_plateau_end} - L{target_crossover_layer - 1}")
            print(f"  PHASE 3 (Target recovery):         "
                  f"L{target_crossover_layer} - L{n_layers - 1}")

    # ---- Per-token trajectory sample ----
    print(f"\n\n{'=' * 100}")
    print("SAMPLE TOKEN TRAJECTORIES (first 5 tokens of first 3 languages)")
    print("=" * 100)

    shown = 0
    for tgt in sorted(pair_data.keys()):
        if shown >= 3:
            break
        entries = pair_data[tgt]

        if use_lang_dist:
            tgt_code = normalize_lang_code(tgt)
            extra = tgt_code
        else:
            tgt_script = TARGET_SCRIPT.get(tgt, "Other")
            extra = tgt_script

        print(f"\n--- {tgt} ({extra}) ---")
        for ti, entry in enumerate(entries[:5]):
            switch_first, switch_cross, switch_stable, per_layer = entry

            if use_lang_dist:
                # Compact trajectory: S=source dominant, T=target, O=other
                traj = ""
                for ps, pt, po in per_layer:
                    m = max(ps, pt, po)
                    if m == pt and pt > 0.01:
                        traj += "T"
                    elif m == ps and ps > 0.01:
                        traj += "S"
                    else:
                        traj += "O"
            else:
                traj = ""
                for s in per_layer:
                    if s == tgt_script:
                        traj += "T"
                    elif s == "Latin":
                        traj += "L"
                    else:
                        traj += "O"

            cross_idx = switch_cross if switch_cross is not None else -1
            first_idx = switch_first if switch_first is not None else -1
            stable_idx = switch_stable if switch_stable is not None else -1

            marker = list(" " * len(traj))
            if first_idx >= 0:
                marker[first_idx] = "^"
            if cross_idx >= 0 and cross_idx != first_idx:
                marker[cross_idx] = "!"
            if stable_idx >= 0 and stable_idx != cross_idx:
                marker[stable_idx] = "*"
            marker_line = "".join(marker)

            print(f"  tok {ti}: {traj}")
            print(f"          {marker_line}  (^=first T, !=crossover, *=stable)")

        shown += 1

    # ---- Summary table ----
    print(f"\n\n{'=' * 100}")
    print("SUMMARY: LANGUAGE SWITCHING ACROSS ALL TARGETS")
    print("=" * 100)

    if use_lang_dist:
        plateau_end = src_plateau_end
        plateau_name = "Source-language"
    else:
        plateau_end = latin_plateau_end
        plateau_name = "Latin"

    print(f"""
  Total tokens analyzed: {total_tokens}
  Target languages: {len(pair_data)}

  Key findings:

  1. EMBEDDING PHASE (layers 0-{plateau_end - 1 if plateau_end else '?'}):
     The embedding and early layers reflect the few-shot context, which
     contains target-language examples.  Target {'language' if use_lang_dist else 'script'} dominates here.

  2. {plateau_name.upper()} PLATEAU (layers {plateau_end}-
     {target_crossover_layer - 1 if target_crossover_layer else '?'}):
     Middle layers are LANGUAGE-NEUTRAL — the model processes semantics
     in a source-language-centric representation space regardless of the
     target language.  {plateau_name} dominates >90% of predictions.

  3. TARGET RECOVERY (layers {target_crossover_layer}-{n_layers - 1}):
     The model switches back to the target language.  The
     crossover layer is where target first exceeds 50%.

  Switching point statistics (per-token crossover):
    Mean:   L{sum(all_crosses) / len(all_crosses):.1f}
    Median: L{sorted(all_crosses)[len(all_crosses) // 2]}
""")


def _sanitize_model_name(name: str) -> str:
    """Sanitize model name for use in directory/file names."""
    return name.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(
        description="Detect language switching points in logit lens results.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Lens output directory (default: lens_output_all)",
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

    if args.model:
        sanitized = _sanitize_model_name(args.model)
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
    else:
        lens_dir = Path(args.dir) if args.dir else LENS_DIR

    lang_dist_path = Path(args.lang_dist) if args.lang_dist else None

    if not lens_dir.exists():
        print(f"Error: {lens_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    analyze_switching(
        lang_dist_path=lang_dist_path,
        lens_dir=lens_dir,
        num_tokens=num_tokens,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
