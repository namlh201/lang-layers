#!/usr/bin/env python3
"""Visualise language-probability heatmaps for source and target languages.

For each token position *i* and layer *j*, computes the probability that the
model's prediction at that layer/position is in a given language *L*, using
top-K marginalisation over the lens predictions weighted by per-token
language probabilities from the token language distribution:

    P_L(i, j) = sum_k P(L | t_k) * p_lens(t_k | h)

computed in log-space via logsumexp for numerical stability:

    log P_L(i, j) = logsumexp_k [ log P(L | t_k) + log p_lens(t_k) ]

Tokens not found in the language distribution receive a uniform prior 1/M
(maximum-entropy default).

Reads ``source_lang`` and ``target_lang`` from the lens JSON metadata and
produces **two** heatmaps — one for the source language, one for the target
language.

Usage::

    python visualize_lang_prob.py \\
        lens_output_all/Qwen__Qwen3.5-9B-Base/flores_eng_Latn_ace_Arab/lens_flores_eng_Latn_ace_Arab_sample0.json \\
        --lang-dist bak/token_lang_dist_fineweb_dataset_Qwen__Qwen3.5-9B-Base.json \\
        --output lang_heatmap

    # Override languages manually
    python visualize_lang_prob.py lens.json --lang-dist dist.json --output out --source-lang eng_Latn --target-lang arb_Arab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import logsumexp

ALL_FONTS = font_manager.get_font_names()
ALL_NOTO_SANS_FONTS = [font for font in ALL_FONTS if "Noto Sans" in font or "Emoji" in font]

sns.set_style(rc={"font.family": ALL_NOTO_SANS_FONTS})


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #

def load_lens(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_lang_dist(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_lang_prob_map(
    lang_dist: dict, tokenizer, lang_code: str,
) -> tuple[dict[str, float], int, dict[str, str]]:
    """Build decoded_token_str -> P(lang_code | token) map from the lang dist JSON.

    Iterates all token_ids in the lang dist, decodes each to its string
    representation via tokenizer.decode([tid]), and maps that string to
    P(lang_code | token). When multiple token_ids decode to the same string,
    uses the weighted average (weighted by total_count) to merge them.

    Also returns M (number of languages) for the uniform fallback, and a
    decoded_str -> raw_bpe_str lookup table (via convert_ids_to_tokens).
    """
    meta = lang_dist.get("metadata", {})
    num_langs = meta.get("num_languages_detected", 0)

    tokens = lang_dist.get("tokens", {})

    # Phase 1: Collect (decoded_str, lang_prob, total_count) for each token_id
    str_to_langs: dict[str, list[tuple[float, float]]] = {}
    decoded_to_raw: dict[str, str] = {}
    for tid_str, entry in tokens.items():
        tid = int(tid_str)
        langs = entry.get("langs", {})
        lang_prob = langs.get(lang_code, 0.0)
        total_count = entry.get("total_count", 0)
        decoded = tokenizer.decode([tid])
        str_to_langs.setdefault(decoded, []).append((lang_prob, total_count))
        raw = tokenizer.convert_ids_to_tokens(tid)
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        decoded_to_raw.setdefault(decoded, raw)

    # Phase 2: Merge collisions via total_count-weighted average
    str_to_lang: dict[str, float] = {}
    for decoded, lang_count_list in str_to_langs.items():
        total_weight = sum(tc for _, tc in lang_count_list)
        if total_weight > 0:
            weighted_lang = sum(lp * tc for lp, tc in lang_count_list) / total_weight
        else:
            weighted_lang = sum(lp for lp, _ in lang_count_list) / len(lang_count_list)
        str_to_lang[decoded] = weighted_lang

    if not num_langs:
        all_langs: set[str] = set()
        for entry in tokens.values():
            all_langs.update(entry.get("langs", {}).keys())
        num_langs = len(all_langs)

    return str_to_lang, num_langs, decoded_to_raw


def lookup_lang_logprob(
    token_str: str,
    str_to_lang: dict[str, float],
    uniform_log: float,
    _cache: dict[str, float],
) -> float:
    """Look up log P(lang | token_str) from the decode-string lookup table.

    Returns log P(lang | t) if found, otherwise the uniform log-prior (1/M).
    Caches results per token string for efficiency.
    """
    if token_str in _cache:
        return _cache[token_str]

    lang_prob = str_to_lang.get(token_str)
    if lang_prob is not None:
        log_p = np.log(lang_prob) if lang_prob > 0 else -np.inf
    else:
        log_p = uniform_log

    _cache[token_str] = log_p
    return log_p


# --------------------------------------------------------------------------- #
#  Core computation
# --------------------------------------------------------------------------- #

def compute_lang_prob_matrix(
    lens_data: dict,
    str_to_lang: dict[str, float],
    num_langs: int,
    top_k: int,
    top_1_only: bool = False,
    tokenizer=None,
    decoded_to_raw: dict[str, str] | None = None,
    is_bpe: bool = True,
) -> tuple[list[str], list[str], np.ndarray, list[list[str]]]:
    """Compute the language-probability matrix.

    Returns:
        layer_names:  e.g. ["embed", "layer 0", ...]
        token_labels: e.g. ["Ace", "hn", ...]
        lang_probs:   (n_tokens, n_layers) array of P(lang) in [0, 1]
        top1_tokens:  (n_tokens, n_layers) list-of-lists of top-1 token strings
    """
    tokens = lens_data["tokens"]
    layer_names = [l["layer"] for l in tokens[0]["layers"]]
    n_tokens = len(tokens)
    n_layers = len(layer_names)

    lang_probs = np.zeros((n_tokens, n_layers), dtype=np.float64)
    top1_tokens: list[list[str]] = [[""] * n_layers for _ in range(n_tokens)]
    token_labels: list[str] = []
    uniform_log = -np.log(num_langs) if num_langs > 0 else -np.inf
    _cache: dict[str, float] = {}

    for i, tok_data in enumerate(tokens):
        token_str = tok_data.get("token", "")
        if tokenizer is not None and "token_id" in tok_data and not is_bpe:
            raw = tokenizer.convert_ids_to_tokens(tok_data["token_id"])
            if raw:
                token_str = raw
        elif is_bpe:
            token_str = token_str.replace(" ", "Ġ")
        token_labels.append(_truncate(token_str))

        for j, layer_entry in enumerate(tok_data["layers"]):
            top_preds = layer_entry.get(f"top{top_k}", [])

            if not top_preds:
                # Fallback to top1/prob fields
                top1 = layer_entry.get("top1", "")
                prob = layer_entry.get("prob", 0.0)
                top1_tokens[i][j] = _short(_to_raw(top1, decoded_to_raw, is_bpe))
                if top1 and prob > 0:
                    log_lang = lookup_lang_logprob(
                        top1, str_to_lang, uniform_log, _cache,
                    )
                    lang_probs[i, j] = np.exp(log_lang + np.log(prob))
                else:
                    lang_probs[i, j] = 0.0
                continue

            top1_tokens[i][j] = _short(_to_raw(top_preds[0]["token"], decoded_to_raw, is_bpe))

            if top_1_only:
                pred = top_preds[0]
                log_lang = lookup_lang_logprob(
                    pred["token"], str_to_lang, uniform_log, _cache,
                )
                log_prob = np.log(pred["prob"]) if pred["prob"] > 0 else -np.inf
                lang_probs[i, j] = np.exp(log_lang + log_prob)
                continue

            # Build log-space terms for logsumexp
            log_terms = np.empty(len(top_preds), dtype=np.float64)
            for k, pred in enumerate(top_preds):
                log_lang = lookup_lang_logprob(
                    pred["token"], str_to_lang, uniform_log, _cache,
                )
                log_prob = np.log(pred["prob"]) if pred["prob"] > 0 else -np.inf
                log_terms[k] = log_lang + log_prob

            lang_probs[i, j] = np.exp(logsumexp(log_terms))

    np.clip(lang_probs, 0.0, 1.0, out=lang_probs)

    return layer_names, token_labels, lang_probs, top1_tokens


def _compress_runs(s: str, min_run: int = 3) -> str:
    """Compress runs of identical characters: 'aaaa' -> '4 * a'."""
    if len(s) < min_run:
        return s
    result = []
    i = 0
    while i < len(s):
        ch = s[i]
        run_len = 1
        while i + run_len < len(s) and s[i + run_len] == ch:
            run_len += 1
        if run_len >= min_run:
            result.append(f"{run_len} * {ch}")
        else:
            result.append(s[i:i + run_len])
        i += run_len
    return "".join(result)


def _escape_raw(s: str) -> str:
    """Escape control characters for display: newline -> \\n, tab -> \\t, etc."""
    result = []
    for ch in s:
        if ch == "\n":
            result.append("\\n")
        elif ch == "\r":
            result.append("\\r")
        elif ch == "\t":
            result.append("\\t")
        elif ch == "\v":
            result.append("\\v")
        elif ch == "\f":
            result.append("\\f")
        elif ch == "\0":
            result.append("\\0")
        elif ord(ch) < 32 or ord(ch) == 127:
            result.append(f"\\x{ord(ch):02x}")
        else:
            result.append(ch)
    return "".join(result)


def _truncate(s: str, maxlen: int = 12) -> str:
    if len(s) > maxlen:
        return s[:maxlen - 2] + ".."
    return s


def _is_bpe_tokenizer(tokenizer) -> bool:
    """Detect if tokenizer is BPE-based (Ġ marker) vs SentencePiece (▁ marker)."""
    try:
        test_ids = tokenizer.encode(" hello", add_special_tokens=False)
        if test_ids:
            raw = tokenizer.convert_ids_to_tokens(test_ids[0])
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            if raw.startswith("Ġ"):
                return True
            if raw.startswith("▁"):
                return False
    except Exception:
        pass
    return True


def _to_raw(token_str: str, decoded_to_raw: dict[str, str] | None = None, is_bpe: bool = True) -> str:
    """Convert a decoded token string to its raw BPE representation.

    For BPE tokenizers, convert_ids_to_tokens returns byte-level encoded
    strings with Unicode rendering issues, so we use the decoded string
    and manually replace spaces with 'Ġ'.

    For SentencePiece tokenizers, use the decoded_to_raw lookup table
    (via convert_ids_to_tokens) which returns '▁'-prefixed strings.
    """
    if is_bpe:
        return token_str.replace(" ", "Ġ")
    if decoded_to_raw is not None:
        raw = decoded_to_raw.get(token_str)
        if raw is not None:
            return raw
    return token_str.replace(" ", "Ġ")


def _short(s: str, maxlen: int = 12) -> str:
    s = _compress_runs(s)
    s = _escape_raw(s)
    if len(s) > maxlen:
        return s[:maxlen - 2] + ".."
    return s


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #

def plot_heatmap(
    layer_names: list[str],
    token_labels: list[str],
    lang_probs: np.ndarray,
    top1_tokens: list[list[str]],
    lang_label: str,
    output_path: str | None,
    dpi: int,
    cmap: str,
    figsize: tuple[float, float] | None,
):
    """Plot a single language-probability heatmap.

    Args:
        lang_label: human-readable label, e.g. "Source (eng_Latn)".
    """
    # Transpose: layers on y-axis, token positions on x-axis
    probs_t = lang_probs.T  # (n_layers, n_tokens)
    n_tokens = probs_t.shape[1]
    n_layers = probs_t.shape[0]

    if figsize is None:
        col_w = max(1.5, min(1.5, 60.0 / n_tokens))
        row_h = 0.5
        figsize = (n_tokens * col_w, n_layers * row_h)

    fig, ax = plt.subplots(figsize=figsize)

    # Build annotation grid: top-1 token + percentage
    annot = np.empty((n_layers, n_tokens), dtype=object)
    for j in range(n_layers):
        for i in range(n_tokens):
            annot[j, i] = f"{top1_tokens[i][j]}\n{probs_t[j, i]:.2%}"

    sns.heatmap(
        probs_t,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=1,
        xticklabels=token_labels,
        yticklabels=layer_names,
        cbar_kws={"label": f"P({lang_label})"},
        linewidths=0.3,
        linecolor="#dddddd",
        annot_kws={
            "fontsize": 7,
            "ha": "center",
            "va": "center",
        },
    )

    ax.set_ylabel("Layer")
    ax.set_title(f"{lang_label} Probability Across Layers")

    # Top x-axis: current token labels
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=45, labelsize=11)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    # Bottom x-axis: shifted right by one position
    shifted_labels = [""] + token_labels[:-1]
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(np.arange(n_tokens) + 0.5)
    secax.set_xticklabels(shifted_labels, rotation=45, ha="right", fontsize=11)
    secax.set_xlabel("Token Position")

    ax.invert_yaxis()

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}", file=sys.stderr)
    else:
        plt.show()
    plt.close(fig)


def plot_layer_avg(
    layer_names: list[str],
    lang_probs: np.ndarray,
    lang_label: str,
    output_path: str | None,
    dpi: int,
    figsize: tuple[float, float] | None,
):
    n_layers = lang_probs.shape[1]
    layer_avg = lang_probs.mean(axis=0)

    if figsize is None:
        figsize = (max(8, n_layers * 0.4), 5)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_layers)
    ax.plot(x, layer_avg, marker="o", markersize=3, linewidth=1.5, color="#2c7fb8")
    ax.fill_between(x, layer_avg, alpha=0.15, color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=9)
    y_max = min(layer_avg.max() + 0.1, 1.0)
    ax.set_ylim(0, y_max)
    ax.set_ylabel(f"Mean P({lang_label})")
    ax.set_xlabel("Layer")
    ax.set_title(f"Mean {lang_label} Probability per Layer")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}", file=sys.stderr)
    else:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Heatmaps of P(source lang) and P(target lang) across layers and positions.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to lens output JSON file.",
    )
    parser.add_argument(
        "--lang-dist",
        type=str,
        required=True,
        help="Path to token_lang_dist_fineweb_dataset_<model>.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Which top-k field to read from the JSON (default: 10 -> 'top10').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output prefix. Four files will be created: <prefix>_source.png, <prefix>_top1_source.png, "
             "<prefix>_target.png, <prefix>_top1_target.png (default: show interactively).",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default=None,
        help="Override source language code (default: read from lens JSON).",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default=None,
        help="Override target language code (default: read from lens JSON).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="YlGnBu",
        help="Seaborn/matplotlib colormap name (default: YlGnBu).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved image (default: 150).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        help="Figure size (width height) in inches (default: auto).",
    )

    args = parser.parse_args()

    # Load data
    lens_data = load_lens(Path(args.input))
    if "tokens" not in lens_data or not lens_data["tokens"]:
        print("Error: no tokens in lens JSON", file=sys.stderr)
        sys.exit(1)

    # Determine source and target languages
    source_lang = args.source_lang or lens_data.get("source_lang", "")
    target_lang = args.target_lang or lens_data.get("target_lang", "")

    if not source_lang and not target_lang:
        print("Error: no source_lang/target_lang in lens JSON and none provided via CLI",
              file=sys.stderr)
        sys.exit(1)

    lang_dist = load_lang_dist(Path(args.lang_dist))

    # Load tokenizer from metadata
    tokenizer_name = lang_dist.get("metadata", {}).get("tokenizer", "")
    if not tokenizer_name:
        print("Error: no tokenizer in lang dist metadata", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {tokenizer_name}", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    is_bpe = _is_bpe_tokenizer(tokenizer)
    print(f"  Tokenizer type: {'BPE' if is_bpe else 'SentencePiece'}", file=sys.stderr)

    figsize = tuple(args.figsize) if args.figsize else None

    for lang_code, role in [(source_lang, "source"), (target_lang, "target")]:
        if not lang_code:
            print(f"Skipping {role} language (not specified)", file=sys.stderr)
            continue

        print(f"\n--- {role} language: {lang_code} ---", file=sys.stderr)

        str_to_lang, num_langs, decoded_to_raw = build_lang_prob_map(lang_dist, tokenizer, lang_code)
        print(f"  {len(str_to_lang)} unique decoded token strings in lang dist, {num_langs} languages",
              file=sys.stderr)

        has_nonzero = any(v > 0 for v in str_to_lang.values())
        if not has_nonzero:
            print(f"  WARNING: language '{lang_code}' not found in any token's lang distribution. "
                  f"Heatmap will show near-zero values (uniform prior only).", file=sys.stderr)

        for top_1_only, suffix in [(False, ""), (True, "top1_")]:
            mode_label = "top-1" if top_1_only else "top-k"
            print(f"  Computing [{mode_label}] matrix...", file=sys.stderr)

            layer_names, token_labels, lang_probs, top1_tokens = compute_lang_prob_matrix(
                lens_data, str_to_lang, num_langs, args.top_k,
                top_1_only=top_1_only,
                tokenizer=tokenizer,
                decoded_to_raw=decoded_to_raw,
                is_bpe=is_bpe,
            )

            if args.output:
                output_path = f"{args.output}_{suffix}{role}.png"
                line_path = f"{args.output}_{suffix}{role}_line.png"
            else:
                output_path = None
                line_path = None

            lang_label = f"{role.capitalize()} ({lang_code})"

            plot_heatmap(
                layer_names, token_labels, lang_probs, top1_tokens,
                lang_label=lang_label,
                output_path=output_path,
                dpi=args.dpi,
                cmap=args.cmap,
                figsize=figsize,
            )

            plot_layer_avg(
                layer_names, lang_probs, lang_label,
                output_path=line_path,
                dpi=args.dpi,
                figsize=figsize,
            )


if __name__ == "__main__":
    main()
