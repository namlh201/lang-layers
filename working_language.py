#!/usr/bin/env python3
"""Working language dynamics analysis.

Implements the mathematical formulation from notes.md:

  A. Expected Language Probability Vector (entropy-weighted):

     P*(L_a | l, i) = sum_j w(t_j) * P(L_a | t_j) * p(t_j | l, i)
                      / sum_j w(t_j) * p(t_j | l, i)

     where w(t_j) = 1 - H(L|t_j) / log2(|L|)

  B. Three language dynamics metrics:
     - Dominant Working Language:  argmax_a P*(L_a | l, i)
     - Cross-lingual Entropy:     H = -sum_a P* log2(P*)
     - Language Shift Point:      KL(P*(L|l,i) || P*(L|l-1,i))

Reads:
  - p(t_j | l, i)  from lens_output_all/<model>/  (top-k predictions per layer)
  - P(L | t)       from token_lang_dist_fineweb_dataset_<model>.json

Usage::

    python working_language.py --model CohereLabs/tiny-aya-base
    python working_language.py --model google/gemma-4-12B --workers 32 --batch-size 500
    python working_language.py --model Qwen/Qwen3.5-9B-Base --output results.txt
    python working_language.py --model utter-project/EuroLLM-9B-2512 --per-pair-detail
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy as scipy_entropy

# Root directory containing per-model lens output subdirectories
LENS_DIR = Path(__file__).parent / "lens_output_all"

# Optional dependency: langcodes library for converting ISO 639-1 → ISO 639-3
try:
    import langcodes as _langcodes
except ImportError:
    _langcodes = None

# ISO 639-3 macrolanguages → their individual language members.
# Used as a fallback when a lang dist vector uses individual codes
# (e.g., "arb") but the lens data references a macrolanguage (e.g., "ara").
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
#  Helpers
# --------------------------------------------------------------------------- #

def normalize_lang_code(lang: str) -> str:
    """Convert FLORES+ / lang_Script codes (e.g., ``eng_Latn``) to ISO 639-3 (``eng``)."""
    # Split on underscore and take the first part (language code portion)
    first = lang.split("_")[0]
    # If already 3 letters (ISO 639-3), return as-is
    if len(first) == 3:
        return first
    # Try langcodes library for 2-letter → 3-letter conversion (e.g., "en" → "eng")
    if _langcodes is not None:
        try:
            return _langcodes.Language.get(lang).to_alpha3()
        except Exception:
            pass
    # Fallback: return whatever we have (may be a 2-letter code)
    return first


def _sanitize_model_name(name: str) -> str:
    """Replace ``/`` with ``__`` for filesystem-safe model directory names."""
    return name.replace("/", "__")


def _lookup_lang_prob_np(log_lang_vec: np.ndarray, code: str) -> float:
    """Look up a single language's probability from a log-prob vector.

    Uses macrolanguage fallback: if ``code`` is a macrolanguage (e.g., ``ara``)
    that doesn't appear directly in the vector, sums (via logsumexp) the
    probabilities of all its member languages (e.g., ``arb`` + ``arz`` + ...).
    """
    if not code:
        return 0.0
    # Direct lookup: map ISO 639-3 code to its index in the log-lang vector
    idx = _LANG_INDEX.get(code)
    if idx is not None:
        return float(np.exp(log_lang_vec[idx]))
    # Macrolanguage fallback: sum probabilities of all member languages
    # In log space, sum = logsumexp of individual log-probs
    if code in MACROLANGUAGE_MEMBERS:
        log_probs = []
        for m in MACROLANGUAGE_MEMBERS[code]:
            idx = _LANG_INDEX.get(m)
            if idx is not None:
                log_probs.append(log_lang_vec[idx])
        if log_probs:
            return float(np.exp(logsumexp(log_probs)))
        return 0.0
    return 0.0


# --------------------------------------------------------------------------- #
#  Globals for multiprocessing
#
#  On Linux, multiprocessing uses fork(), so child processes inherit these
#  globals directly from the parent without re-serialization.  They are set
#  once in the main process (via load_token_lang_dist) before the pool is
#  created, and all workers can read them concurrently (read-only).
# --------------------------------------------------------------------------- #

# token_str → dense numpy vector of log-language-probabilities (length = |L|)
# Stored in log space: log P(L_a | t), with -inf for zero-probability languages
_TOKEN_LANG_DIST_LOG_NP: dict[str, np.ndarray] | None = None
# Sorted list of ISO 639-3 language codes (deterministic ordering)
_LANG_CODES: list[str] = []
# lang_code → column index in the language vector (reverse lookup for _LANG_CODES)
_LANG_INDEX: dict[str, int] = {}
# Total number of languages detected (from metadata), used for max entropy
_NUM_LANGS: int = 0
# Total vocabulary size (from metadata), used for uniform fallback when lens probs sum to 0
_NUM_TOKENS: int = 0
# Number of top-k predictions per layer to read (reads "top10" field by default)
_TOP_K: int = 10


# --------------------------------------------------------------------------- #
#  Token language distribution loading
# --------------------------------------------------------------------------- #

def load_token_lang_dist(path: Path) -> None:
    """Load token language distribution JSON and build numpy vectors.

    Reads ``token_lang_dist_fineweb_dataset_<model>.json``, which maps
    each vocabulary token to its language distribution P(L|t).

    Sets globals: _TOKEN_LANG_DIST_LOG_NP, _LANG_CODES, _LANG_INDEX,
    _NUM_LANGS, _NUM_TOKENS.
    """
    global _TOKEN_LANG_DIST_LOG_NP, _LANG_CODES, _LANG_INDEX
    global _NUM_LANGS, _NUM_TOKENS

    print(f"Loading token language distribution from {path}...", file=sys.stderr)
    with open(path) as f:
        data = json.load(f)

    # Extract metadata: num_languages_detected and vocab_size
    meta = data.get("metadata", {})
    _NUM_LANGS = meta.get("num_languages_detected", 0)   # |L| in the formulas
    _NUM_TOKENS = meta.get("vocab_size", 0)               # |V| for uniform fallback

    print(f"  num_languages_detected: {_NUM_LANGS}", file=sys.stderr)
    print(f"  vocab_size: {_NUM_TOKENS}", file=sys.stderr)

    # Phase 1: Parse JSON into a Python dict mapping token_str → {lang_code: prob}
    mapping: dict[str, dict[str, float]] = {}
    all_langs: set[str] = set()    # Track all unique lang codes for index construction
    n_tokens = 0

    for tid, tdata in data.get("tokens", {}).items():
        token_str = tdata.get("token", "")
        if not token_str:
            continue
        # Convert SentencePiece marker (▁) to space to match lens output format
        token_str = token_str.replace("\u2581", " ")
        # Skip tokens with no language distribution data
        raw_langs = tdata.get("langs", {})
        if not raw_langs:
            continue
        # Normalize lang codes (e.g., "eng_Latn" → "eng") and merge duplicates
        langs: dict[str, float] = {}
        for lang_code, prob in raw_langs.items():
            norm = normalize_lang_code(lang_code)
            langs[norm] = langs.get(norm, 0.0) + prob
        if langs:
            mapping[token_str] = langs
            n_tokens += 1
            all_langs.update(langs.keys())

    # Phase 2: Build deterministic language ordering and index mapping
    _LANG_CODES = sorted(all_langs)                     # Deterministic column order for vectors
    _LANG_INDEX = {lc: i for i, lc in enumerate(_LANG_CODES)}  # lang_code → column index
    n_langs = len(_LANG_CODES)
    # If metadata didn't have num_languages_detected, derive from data
    if not _NUM_LANGS:
        _NUM_LANGS = n_langs

    # Phase 3: Convert each token's lang dist dict to a dense numpy vector
    # of LOG-probabilities for numerically-stable logsumexp operations
    _TOKEN_LANG_DIST_LOG_NP = {}
    for tok, langs in mapping.items():
        # Initialise to -inf (= log 0) for all languages
        log_vec = np.full(n_langs, -np.inf, dtype=np.float64)
        for lc, p in langs.items():
            log_vec[_LANG_INDEX[lc]] = np.log(p)
        _TOKEN_LANG_DIST_LOG_NP[tok] = log_vec

    print(
        f"  Loaded {n_tokens} tokens with language data ({n_langs} languages)",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
#  Math: entropy weight & KL divergence
# --------------------------------------------------------------------------- #

def _compute_entropy_weight(log_lang_vec: np.ndarray) -> float:
    """Compute w(t) = 1 - H(L|t) / log2(|L|) using scipy.stats.entropy.

    Takes a **log-probability** vector (log P(L|t)) to avoid underflow.
    - Returns 1.0 for tokens with a single language (zero entropy = fully informative)
    - Returns 0.0 for tokens with uniform distribution across all languages (max entropy = pure noise)
    """
    # Edge case: empty vector → no information to discount, treat as fully informative
    if log_lang_vec is None or log_lang_vec.size == 0:
        return 1.0
    # Edge case: only one language → entropy is 0, weight is 1
    if _NUM_LANGS is None or _NUM_LANGS <= 1:
        return 1.0
    # Skip if all entries are -inf (no language data)
    if not np.any(np.isfinite(log_lang_vec)):
        return 1.0
    # Convert log-probs to normalised probs for entropy computation
    # Subtract max for numerical stability before exp
    p = np.exp(log_lang_vec - logsumexp(log_lang_vec))
    # scipy.stats.entropy normalises pk internally and returns H in the given base
    entropy_bits = float(scipy_entropy(p, base=2))
    # Maximum possible entropy = log2(|L|), achieved when distribution is uniform
    max_entropy = np.log2(_NUM_LANGS)
    if max_entropy <= 0:
        return 1.0
    # Information weight = 1 - normalised entropy (0 = pure noise, 1 = fully informative)
    return float(1.0 - entropy_bits / max_entropy)


def _compute_kl_divergence(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """Compute KL(P || Q) = sum P(a) * log2(P(a) / Q(a)) using scipy.stats.entropy.

    Takes **log-probability** vectors to avoid underflow.
    This is the **Language Shift Point** metric from notes.md §2.
    """
    # Convert log-probs to normalised linear probs
    # logsumexp normalisation ensures numerical stability
    p = np.exp(log_p - logsumexp(log_p))
    q = np.exp(log_q - logsumexp(log_q))
    # scipy.stats.entropy(pk, qk, base=2) computes KL(pk || qk) in bits
    # Returns inf when qk has zeros where pk has nonzeros (correct KL behaviour)
    return float(scipy_entropy(p, q, base=2))


# --------------------------------------------------------------------------- #
#  Multiprocessing worker
# --------------------------------------------------------------------------- #

def _process_file_batch(
    file_paths: list[str],
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, np.ndarray]],
    dict[tuple[str, str], dict[str, list]],
    dict[tuple[str, str], dict],
    int,
]:
    """Process a batch of lens JSON files in a worker process.

    For each token at each layer, computes the entropy-weighted language
    probability vector P*(L|l,i) (notes.md §A + §B) and accumulates
    per-pair, per-layer statistics.  Also computes KL divergence between
    consecutive layers per token (notes.md §2, Language Shift Point).

    All probability multiplications are performed in **log space** using
    ``scipy.special.logsumexp`` to prevent underflow when individual
    probabilities are extremely small.

    Pipeline per token per layer:
      1. Extract top-k predictions from lens JSON  → p(t_j | l, i)
      2. Convert lens probs to log space           → log p(t_j | l, i)
      3. Compute entropy info weights in log space → log w(t_j) + log p(t_j|l,i)
      4. Compute P*(L|l,i) via logsumexp            → log P*(L_a | l, i)
      5. Accumulate into per-pair language vectors (linear space)
      6. Store per-token log vector for KL computation

    Returns:
        layer_names, pair_lang_vecs, pair_kl, pair_meta, n_processed
    """
    layer_names: list[str] = []
    # pair_key (src_lang, tgt_lang) → {layer_name → accumulated lang vec (linear)}
    pair_lang_vecs: dict[tuple, dict[str, np.ndarray]] = {}
    # pair_key → {layer_name → [kl_sum, token_count]} for averaging KL
    pair_kl: dict[tuple, dict[str, list]] = {}
    # pair_key → {n_tokens, src_code, tgt_code}
    pair_meta: dict[tuple, dict] = {}
    n_processed = 0
    n_langs = len(_LANG_CODES)    # Dimensionality of language vectors

    for fpath_str in file_paths:
        # Load lens JSON; skip if unreadable or empty
        try:
            data = json.loads(Path(fpath_str).read_text())
        except Exception:
            continue
        if "tokens" not in data or not data["tokens"]:
            continue

        n_processed += 1
        # Extract source/target language pair from lens JSON metadata
        src_lang = data.get("source_lang", "")
        tgt_lang = data.get("target_lang", "")
        pair_key = (src_lang, tgt_lang)

        # Initialise per-pair data structures on first encounter
        if pair_key not in pair_meta:
            pair_meta[pair_key] = {
                "n_tokens": 0,
                "src_code": normalize_lang_code(src_lang),
                "tgt_code": normalize_lang_code(tgt_lang),
            }
            pair_lang_vecs[pair_key] = {}
            pair_kl[pair_key] = {}

        # Extract layer names from the first token's layer list (all tokens share the same layers)
        if not layer_names:
            layer_names = [
                l["layer"] for l in data["tokens"][0]["layers"]
            ]

        # Get references to this pair's accumulators for convenience
        pmeta = pair_meta[pair_key]
        plvecs = pair_lang_vecs[pair_key]
        pkl = pair_kl[pair_key]

        # Process each token in the lens JSON
        for tok_data in data["tokens"]:
            pmeta["n_tokens"] += 1

            # Per-token storage: LOG language vector at each layer (for KL)
            tok_layer_log_vecs: list[np.ndarray] = []
            # Layer names in order (needed to attribute KL to the correct layer)
            tok_layer_names: list[str] = []

            # Iterate over layers (embed, layer 0, layer 1, ..., layer N)
            for layer_entry in tok_data["layers"]:
                lname = layer_entry["layer"]
                tok_layer_names.append(lname)

                # --- Step 1: Extract top-k predictions from lens JSON ---
                top_preds = layer_entry.get(f"top{_TOP_K}", [])
                if not top_preds:
                    top_preds = [{
                        "token": layer_entry.get("top1", ""),
                        "prob": layer_entry.get("prob", 1.0),
                    }]

                # --- Step 2: Convert lens probabilities to log space ---
                # log p(t_j | l, i) = log(prob) - logsumexp(log(probs))
                # This is the normalised log-softmax over the top-k
                probs = np.array(
                    [t["prob"] for t in top_preds], dtype=np.float64,
                )
                total_lp = probs.sum()
                if total_lp > 0:
                    # Normalised log lens weights: log(prob / sum(probs))
                    with np.errstate(divide="ignore"):
                        log_probs = np.log(probs)
                    log_lens_weights = log_probs - logsumexp(log_probs)
                elif _NUM_TOKENS:
                    # Fallback: all probs are 0 → uniform log-prob over entire vocab
                    log_lens_weights = np.full(
                        len(top_preds), -np.log(_NUM_TOKENS),
                    )
                else:
                    # Last-resort fallback: uniform log-prob over top-k
                    log_lens_weights = np.full(
                        len(top_preds), -np.log(len(top_preds)),
                    )

                # --- Step 3: Collect valid tokens & compute log info weights ---
                # For each token with lang dist data, compute:
                #   log_combined_j = log w(t_j) + log p(t_j | l, i)
                # Tokens without lang dist data are excluded.
                valid_log_ld: list[np.ndarray] = []       # log P(L|t_j) vectors
                valid_log_combined: list[float] = []       # log w(t_j) + log p(t_j|l,i)

                for j, t in enumerate(top_preds):
                    log_ld = _TOKEN_LANG_DIST_LOG_NP.get(t["token"])
                    if log_ld is not None:
                        w_j = _compute_entropy_weight(log_ld)
                        if w_j > 0:
                            # Normal case: include entropy weight
                            valid_log_ld.append(log_ld)
                            valid_log_combined.append(
                                np.log(w_j) + log_lens_weights[j]
                            )

                # --- Step 4: Compute P*(L|l,i) via logsumexp ---
                # P*(L_a|l,i) = sum_j w(t_j) * P(L_a|t_j) * p(t_j|l,i)
                #               / sum_j w(t_j) * p(t_j|l,i)
                #
                # In log space:
                #   log_num_a = logsumexp_j(log_combined_j + log P(L_a|t_j))
                #   log_den   = logsumexp_j(log_combined_j)
                #   log P*(L_a|l,i) = log_num_a - log_den
                log_lang_vec = np.full(n_langs, -np.inf, dtype=np.float64)

                if valid_log_combined:
                    # Stack into matrices for vectorised logsumexp
                    log_ld_matrix = np.array(valid_log_ld)          # (k_valid, n_langs)
                    log_combined = np.array(valid_log_combined)      # (k_valid,)

                    # Numerator: for each language a, logsumexp over j of
                    #   (log_combined_j + log_ld_matrix[j, a])
                    log_num = logsumexp(
                        log_ld_matrix + log_combined[:, None], axis=0,
                    )  # (n_langs,)

                    # Denominator: logsumexp of log_combined (scalar)
                    log_den = logsumexp(log_combined)

                    # log P*(L|l,i) = log_num - log_den
                    log_lang_vec = log_num - log_den
                else:
                    # Fallback: all tokens have zero info weight (all ambiguous)
                    # Re-collect with unweighted lens probs (drop entropy weight)
                    valid_log_ld_fb: list[np.ndarray] = []
                    valid_log_combined_fb: list[float] = []
                    for j, t in enumerate(top_preds):
                        log_ld = _TOKEN_LANG_DIST_LOG_NP.get(t["token"])
                        if log_ld is not None:
                            valid_log_ld_fb.append(log_ld)
                            valid_log_combined_fb.append(log_lens_weights[j])

                    if valid_log_combined_fb:
                        log_ld_matrix = np.array(valid_log_ld_fb)
                        log_combined = np.array(valid_log_combined_fb)
                        log_num = logsumexp(
                            log_ld_matrix + log_combined[:, None], axis=0,
                        )
                        log_den = logsumexp(log_combined)
                        log_lang_vec = log_num - log_den

                # --- Step 5: Accumulate into per-pair language vectors ---
                # Convert log P* to linear for accumulation (safe: P* is a normalised dist)
                lang_vec = np.exp(log_lang_vec)

                if lang_vec.any():
                    if lname not in plvecs:
                        plvecs[lname] = np.zeros(n_langs, dtype=np.float64)
                    plvecs[lname] += lang_vec

                # Store per-token LOG vector for KL computation
                tok_layer_log_vecs.append(log_lang_vec)

            # --- Step 6: KL divergence between consecutive layers (per token) ---
            # Delta_l = KL(P*(L|l,i) || P*(L|l-1,i)) — notes.md §2, Language Shift Point
            # Computed per-token from log P* vectors, then averaged across tokens later
            if len(tok_layer_log_vecs) > 1:
                for i in range(1, len(tok_layer_log_vecs)):
                    prev_log_vec = tok_layer_log_vecs[i - 1]   # log P*(L|l-1,i)
                    curr_log_vec = tok_layer_log_vecs[i]       # log P*(L|l,i)
                    curr_lname = tok_layer_names[i]

                    # Skip if either vector is all -inf (no lang dist data)
                    if not np.any(np.isfinite(prev_log_vec)) or not np.any(np.isfinite(curr_log_vec)):
                        continue

                    # KL(P*(L|l,i) || P*(L|l-1,i)) via scipy.stats.entropy
                    kl = _compute_kl_divergence(curr_log_vec, prev_log_vec)

                    # Accumulate KL sum and count for averaging in the main process
                    if curr_lname not in pkl:
                        pkl[curr_lname] = [0.0, 0]
                    pkl[curr_lname][0] += kl
                    pkl[curr_lname][1] += 1

    return layer_names, pair_lang_vecs, pair_kl, pair_meta, n_processed


# --------------------------------------------------------------------------- #
#  Metrics computation
# --------------------------------------------------------------------------- #

def compute_metrics(
    layer_names: list[str],
    lang_vecs: dict[str, np.ndarray],
    kl_data: dict[str, list],
) -> tuple[
    list[tuple[str, float]],
    list[float],
    list[float | None],
]:
    """Compute the three language dynamics metrics per layer (notes.md §2).

    Operates on the accumulated (summed) language vectors from the worker,
    normalising them to get the average P*(L|l) across all tokens.

    Returns:
        dominant_langs: [(lang_code, prob)] per layer — Metric 1
        entropies: [float] per layer in bits — Metric 2
        kl_divs: [float | None] per layer (None for first layer) — Metric 3
    """
    dominant_langs: list[tuple[str, float]] = []
    entropies: list[float] = []
    kl_divs: list[float | None] = []

    for i, lname in enumerate(layer_names):
        # Get the accumulated language vector for this layer
        lv = lang_vecs.get(lname)
        if lv is not None and lv.sum() > 0:
            # Normalise: divide by total to get average P*(L|l) across tokens
            avg_lv = lv / lv.sum()

            # Metric 1: Dominant Working Language = argmax_a P*(L_a | l)
            dom_idx = int(np.argmax(avg_lv))
            dom_lang = _LANG_CODES[dom_idx] if _LANG_CODES else "N/A"
            dom_prob = float(avg_lv[dom_idx])
            dominant_langs.append((dom_lang, dom_prob))

            # Metric 2: Cross-lingual Entropy via scipy.stats.entropy
            # H_working = -sum_a P* log2(P*) (in bits with base=2)
            entropy = float(scipy_entropy(avg_lv, base=2))
            entropies.append(entropy)
        else:
            # No language data for this layer — report N/A
            dominant_langs.append(("N/A", 0.0))
            entropies.append(0.0)

        # Metric 3: Language Shift Point = average KL divergence at this layer
        # KL computed per-token between layer l and l-1, then averaged:
        #   Delta_l = KL(P*(L|l,i) || P*(L|l-1,i))
        # kl_data = [kl_sum, token_count]; average = kl_sum / count
        kl = kl_data.get(lname)
        if kl and kl[1] > 0:
            kl_divs.append(kl[0] / kl[1])    # Average KL across tokens
        else:
            kl_divs.append(None)             # First layer (no previous) or no data

    return dominant_langs, entropies, kl_divs


# --------------------------------------------------------------------------- #
#  Output
# --------------------------------------------------------------------------- #

def print_metrics_table(
    layer_names: list[str],
    dominant_langs: list[tuple[str, float]],
    entropies: list[float],
    kl_divs: list[float | None],
    num_langs: int,
    title: str = "LANGUAGE DYNAMICS METRICS",
):
    # Maximum possible entropy = log2(|L|), achieved when all languages are equally likely
    max_entropy = (
        float(np.log2(num_langs)) if num_langs and num_langs > 1 else 1.0
    )

    # Print table header
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")

    header = (
        f"{'Layer':<14} {'Dominant Lang':<16} {'Dom Prob':>8}"
        f" {'Entropy':>8} {'Max Ent':>8} {'Norm Ent':>8}"
        f" {'KL Divergence':>14}  Bar"
    )
    print(f"\n{header}")
    print("-" * 110)

    # Print one row per layer with all three metrics
    for i, lname in enumerate(layer_names):
        dom_lang, dom_prob = dominant_langs[i]    # Metric 1: dominant language
        entropy = entropies[i]                     # Metric 2: cross-lingual entropy
        # Normalised entropy = H / H_max (0 = decisive, 1 = fully ambiguous)
        norm_ent = entropy / max_entropy if max_entropy > 0 else 0.0
        kl = kl_divs[i]                            # Metric 3: KL divergence shift
        kl_str = f"{kl:.4f}" if kl is not None else "-"
        # Visual bar: length proportional to normalised entropy
        bar = "#" * int(norm_ent * 30)

        print(
            f"{lname:<14} {dom_lang:<16} {dom_prob:>7.1%}"
            f" {entropy:>8.3f} {max_entropy:>8.3f} {norm_ent:>8.1%}"
            f" {kl_str:>14}  |{bar}"
        )

    # --- Summary section: min/max entropy, max/avg KL, shift layers ---
    print(f"\n  Summary:")
    # Convert to numpy arrays for efficient min/max/mean computation
    ent_arr = np.array(
        [e for e in entropies if e > 0], dtype=np.float64,
    )
    kl_arr = np.array(
        [k for k in kl_divs if k is not None], dtype=np.float64,
    )

    # Entropy summary: find layers with min and max cross-lingual entropy
    if ent_arr.size > 0:
        # Map back to original layer indices (we filtered out zero-entropy layers)
        valid = [(i, e) for i, e in enumerate(entropies) if e > 0]
        min_idx = int(np.argmin(ent_arr))    # Index into filtered array
        max_idx = int(np.argmax(ent_arr))
        min_ent = float(ent_arr[min_idx])
        max_ent = float(ent_arr[max_idx])
        min_lname = layer_names[valid[min_idx][0]]    # Map back to layer name
        max_lname = layer_names[valid[max_idx][0]]
        print(
            f"    Min entropy: {min_lname:<14}"
            f" ({min_ent:.3f} bits, norm={min_ent / max_entropy:.1%})"
        )
        print(
            f"    Max entropy: {max_lname:<14}"
            f" ({max_ent:.3f} bits, norm={max_ent / max_entropy:.1%})"
        )

    # KL divergence summary: find layer with max language shift
    if kl_arr.size > 0:
        # Map back to original layer indices
        valid_kls = [
            (i, k) for i, k in enumerate(kl_divs) if k is not None
        ]
        max_kl_idx = int(np.argmax(kl_arr))
        max_kl = float(kl_arr[max_kl_idx])
        avg_kl = float(np.mean(kl_arr))
        print(
            f"    Max KL divergence: {layer_names[valid_kls[max_kl_idx][0]]:<14}"
            f" ({max_kl:.4f})"
        )
        print(f"    Average KL divergence: {avg_kl:.4f}")

        # Flag layers where language shifts significantly (KL > threshold = 0.5 bits)
        threshold = 0.5
        shift_layers = [
            layer_names[i] for i, k in valid_kls if k > threshold
        ]
        if shift_layers:
            print(
                f"    Language shift layers (KL > {threshold}):"
                f" {', '.join(shift_layers)}"
            )


def print_per_pair_summary(
    pair_lang_vecs: dict[tuple, dict[str, np.ndarray]],
    pair_kl: dict[tuple, dict[str, list]],
    pair_meta: dict[tuple, dict],
    layer_names: list[str],
    num_langs: int,
):
    # Print per-pair summary table: one row per (src, tgt) language pair
    print(f"\n{'=' * 150}")
    print("PER-PAIR SUMMARY")
    print(f"{'=' * 150}")

    # Columns: pair name, token count, dominant language at first/middle/last layer,
    #          layer with min entropy, layer with max KL divergence
    header = (
        f"{'Pair':<50} {'Tokens':>7}"
        f" {'Dom(first)':>14} {'Dom(mid)':>14} {'Dom(last)':>14}"
        f" {'Min Ent':>14} {'Max KL':>14}"
    )
    print(header)
    print("-" * 150)

    # Middle layer index for sampling dominant language at midpoint
    mid_idx = len(layer_names) // 2

    for pair_key in sorted(pair_lang_vecs.keys()):
        src_lang, tgt_lang = pair_key
        meta = pair_meta[pair_key]
        pair_str = f"{src_lang} -> {tgt_lang}"

        # Get this pair's accumulated language vectors and KL data
        lang_vecs = pair_lang_vecs[pair_key]
        kl_data = pair_kl.get(pair_key, {})

        # Compute all three metrics for this pair
        dom_langs, entropies, kl_divs = compute_metrics(
            layer_names, lang_vecs, kl_data,
        )

        # Helper: format dominant language as "lang_code(prob%)"
        def _fmt_dom(idx):
            lang, prob = dom_langs[idx]
            if lang == "N/A":
                return "N/A"
            return f"{lang}({prob:.0%})"

        # Sample dominant language at three key positions: first, middle, last layer
        first_dom = _fmt_dom(0)
        mid_dom = _fmt_dom(mid_idx)
        last_dom = _fmt_dom(-1)

        # Find layer with minimum cross-lingual entropy (most decisive language representation)
        valid_ents = [(i, e) for i, e in enumerate(entropies) if e > 0]
        if valid_ents:
            min_ent_idx = min(valid_ents, key=lambda x: x[1])[0]
            min_ent_str = (
                f"{layer_names[min_ent_idx]}({entropies[min_ent_idx]:.2f})"
            )
        else:
            min_ent_str = "-"

        # Find layer with maximum KL divergence (biggest language shift)
        valid_kls = [(i, k) for i, k in enumerate(kl_divs) if k is not None]
        if valid_kls:
            max_kl_idx = max(valid_kls, key=lambda x: x[1])[0]
            max_kl_str = (
                f"{layer_names[max_kl_idx]}({kl_divs[max_kl_idx]:.4f})"
            )
        else:
            max_kl_str = "-"

        print(
            f"{pair_str:<50} {meta['n_tokens']:>7}"
            f" {first_dom:>14} {mid_dom:>14} {last_dom:>14}"
            f" {min_ent_str:>14} {max_kl_str:>14}"
        )


# --------------------------------------------------------------------------- #
#  Main analysis
# --------------------------------------------------------------------------- #

def run_model_analysis(
    lens_dir: Path,
    model_name: str,
    output_path: Path | None = None,
    workers: int | None = None,
    batch_size: int = 200,
    progress_every: int = 5000,
    per_pair_detail: bool = False,
) -> bool:
    """Run analysis for a single model directory.

    1. Collects all lens JSON files in the directory
    2. Batches them and dispatches to multiprocessing workers
    3. Merges partial results (language vectors + KL data) from workers
    4. Computes and prints the three language dynamics metrics
    """

    # Collect all lens JSON files (excluding summary.json) recursively
    files = sorted(
        f for f in lens_dir.rglob("*.json")
        if f.name != "summary.json"
    )
    if not files:
        print(f"No lens JSON files found in {lens_dir}", file=sys.stderr)
        return False

    print(f"Found {len(files)} files", file=sys.stderr)

    # Split files into batches for dispatch to worker processes
    file_batches: list[list[str]] = []
    batch: list[str] = []
    for f in files:
        batch.append(str(f))
        if len(batch) >= batch_size:
            file_batches.append(batch)
            batch = []
    if batch:    # Don't forget the last partial batch
        file_batches.append(batch)

    # Accumulators for merged results from all workers
    layer_names: list[str] = []
    # Per-pair: {pair_key → {layer_name → accumulated lang vec}}
    pair_lang_vecs: dict[tuple, dict[str, np.ndarray]] = {}
    # Per-pair KL: {pair_key → {layer_name → [kl_sum, count]}}
    pair_kl: dict[tuple, dict[str, list]] = {}
    # Per-pair metadata: {pair_key → {n_tokens, src_code, tgt_code}}
    pair_meta: dict[tuple, dict] = {}
    # Aggregate (all pairs): {layer_name → accumulated lang vec}
    agg_lang_vecs: dict[str, np.ndarray] = {}
    # Aggregate KL: {layer_name → [kl_sum, count]}
    agg_kl: dict[str, list] = {}

    n_workers = workers or os.cpu_count() or 1
    processed = 0
    total_files = len(files)

    print(
        f"Processing with {n_workers} workers "
        f"({len(file_batches)} batches of ~{batch_size})...",
        file=sys.stderr,
    )

    # Dispatch batches to worker processes via imap_unordered (results arrive in any order)
    with mp.Pool(n_workers) as pool:
        for (
            w_layers, w_plv, w_pkl, w_pmeta, w_n,
        ) in pool.imap_unordered(_process_file_batch, file_batches):
            # Capture layer names from the first batch that returns them
            if not layer_names and w_layers:
                layer_names = w_layers

            # --- Merge per-pair language vectors ---
            # numpy array addition: accumulate P*(L|l) across all tokens for each pair
            for pk, lv_dict in w_plv.items():
                if pk not in pair_lang_vecs:
                    pair_lang_vecs[pk] = {}
                for lname, lv in lv_dict.items():
                    if lname not in pair_lang_vecs[pk]:
                        pair_lang_vecs[pk][lname] = np.zeros_like(lv)   # Init zero vector
                    pair_lang_vecs[pk][lname] += lv                      # Accumulate
                    # Also accumulate into the aggregate (all-pairs) language vectors
                    if lname not in agg_lang_vecs:
                        agg_lang_vecs[lname] = np.zeros_like(lv)
                    agg_lang_vecs[lname] += lv

            # --- Merge per-pair KL divergence data ---
            # kl_data = [kl_sum, count]; we accumulate both for later averaging
            for pk, kl_dict in w_pkl.items():
                if pk not in pair_kl:
                    pair_kl[pk] = {}
                for lname, kl_data in kl_dict.items():
                    if lname not in pair_kl[pk]:
                        pair_kl[pk][lname] = [0.0, 0]
                    pair_kl[pk][lname][0] += kl_data[0]    # Accumulate KL sum
                    pair_kl[pk][lname][1] += kl_data[1]    # Accumulate token count
                    # Also accumulate into the aggregate KL data
                    if lname not in agg_kl:
                        agg_kl[lname] = [0.0, 0]
                    agg_kl[lname][0] += kl_data[0]
                    agg_kl[lname][1] += kl_data[1]

            # --- Merge per-pair metadata (token counts) ---
            for pk, wmeta in w_pmeta.items():
                if pk not in pair_meta:
                    pair_meta[pk] = {
                        "n_tokens": 0,
                        "src_code": wmeta["src_code"],
                        "tgt_code": wmeta["tgt_code"],
                    }
                pair_meta[pk]["n_tokens"] += wmeta["n_tokens"]

            # Progress reporting
            processed += w_n
            if progress_every and (
                processed // progress_every
                != (processed - w_n) // progress_every
            ):
                print(
                    f"  ...{processed}/{total_files} files",
                    file=sys.stderr,
                )

    n_pairs = len(pair_meta)
    n_tokens = sum(m["n_tokens"] for m in pair_meta.values())

    print(
        f"Processed {total_files} files, {n_pairs} pairs, {n_tokens} tokens",
        file=sys.stderr,
    )
    if layer_names:
        print(
            f"Layers: {len(layer_names)} "
            f"({layer_names[0]} ... {layer_names[-1]})",
            file=sys.stderr,
        )

    # Redirect stdout to output file if specified, otherwise print to terminal
    out = open(output_path, "w") if output_path else None
    old_stdout = sys.stdout
    if out:
        sys.stdout = out

    try:
        # --- Print header info ---
        print(f"Model: {model_name}")
        print(f"Directory: {lens_dir}")
        print(f"Files: {total_files}")
        print(f"Pairs: {n_pairs}")
        print(f"Tokens: {n_tokens}")
        if layer_names:
            print(
                f"Layers: {len(layer_names)} "
                f"({layer_names[0]} ... {layer_names[-1]})"
            )
        print(f"Languages: {_NUM_LANGS}")
        print(f"Top-k: {_TOP_K}")

        # --- Print aggregate metrics table (all pairs combined) ---
        if agg_lang_vecs:
            dom_langs, entropies, kl_divs = compute_metrics(
                layer_names, agg_lang_vecs, agg_kl,
            )
            print_metrics_table(
                layer_names, dom_langs, entropies, kl_divs, _NUM_LANGS,
                title="AGGREGATE LANGUAGE DYNAMICS (all pairs, entropy-weighted)",
            )

        # --- Print per-pair summary table (one row per pair) ---
        if pair_lang_vecs:
            print_per_pair_summary(
                pair_lang_vecs, pair_kl, pair_meta,
                layer_names, _NUM_LANGS,
            )

        # --- Optionally print detailed per-pair metrics tables ---
        if per_pair_detail:
            for pair_key in sorted(pair_lang_vecs.keys()):
                src_lang, tgt_lang = pair_key
                meta = pair_meta[pair_key]
                lang_vecs = pair_lang_vecs[pair_key]
                kl_data = pair_kl.get(pair_key, {})
                dom_langs, entropies, kl_divs = compute_metrics(
                    layer_names, lang_vecs, kl_data,
                )
                print_metrics_table(
                    layer_names, dom_langs, entropies, kl_divs, _NUM_LANGS,
                    title=(
                        f"DETAIL: {src_lang} -> {tgt_lang}"
                        f" ({meta['n_tokens']} tokens)"
                    ),
                )

        print()
    finally:
        # Restore stdout and close output file
        sys.stdout = old_stdout
        if out:
            out.close()

    if output_path:
        print(f"Saved: {output_path}", file=sys.stderr)
    return True


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Working language dynamics analysis (notes.md metrics). "
            "Computes entropy-weighted language probability vectors and "
            "three metrics: dominant language, cross-lingual entropy, "
            "and KL divergence language shift points."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., CohereLabs/tiny-aya-base). "
        "Auto-resolves lens dir and lang dist path.",
    )
    parser.add_argument(
        "--lens-dir",
        type=str,
        default=None,
        help="Override lens output directory.",
    )
    parser.add_argument(
        "--lang-dist",
        type=str,
        default=None,
        help="Override token lang dist JSON path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        ),
        help="Number of worker processes "
        "(default: SLURM_CPUS_PER_TASK or CPU count).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Files per batch sent to each worker (default: 200).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top predictions per layer to read from lens JSON"
        " (default: 10, reads 'top10' field).",
    )
    parser.add_argument(
        "--per-pair-detail",
        action="store_true",
        help="Show detailed per-pair metrics tables.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N files (0 to disable).",
    )
    args = parser.parse_args()

    global _TOP_K    # Set the global top-k for use in worker processes

    # --- Resolve paths from model name ---
    sanitized = _sanitize_model_name(args.model)    # e.g., "google/gemma-4-12B" → "google__gemma-4-12B"
    _TOP_K = args.top_k

    # Lens directory: lens_output_all/<sanitized_model>/
    lens_dir = (
        Path(args.lens_dir) if args.lens_dir
        else LENS_DIR / sanitized
    )
    if not lens_dir.exists():
        print(f"Error: lens dir {lens_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Language distribution: token_lang_dist_fineweb_dataset_<sanitized_model>.json
    lang_dist_path = (
        Path(args.lang_dist) if args.lang_dist
        else Path(__file__).parent
        / f"token_lang_dist_fineweb_dataset_{sanitized}.json"
    )
    if not lang_dist_path.exists():
        print(
            f"Error: lang dist {lang_dist_path} not found",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load token language distribution into globals (inherited by forked workers)
    load_token_lang_dist(lang_dist_path)

    output_path = Path(args.output) if args.output else None

    # Run the analysis
    run_model_analysis(
        lens_dir=lens_dir,
        model_name=args.model,
        output_path=output_path,
        workers=args.workers,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        per_pair_detail=args.per_pair_detail,
    )


if __name__ == "__main__":
    main()
