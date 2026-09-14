#!/usr/bin/env python3
"""Visualise English-probability heatmap across layers and token positions.

For each token position *i* and layer *j*, computes the probability that the
model's prediction at that layer/position is English, using top-K
marginalisation over the lens predictions weighted by per-token English
probabilities from the token language distribution:

    P_eng(i, j) = sum_k P(eng | t_k) * p_lens(t_k | h)

computed in log-space via logsumexp for numerical stability:

    log P_eng(i, j) = logsumexp_k [ log P(eng | t_k) + log p_lens(t_k) ]

Tokens not found in the language distribution receive a uniform prior 1/M
(maximum-entropy default).

Usage::

    python visualize_english_prob.py \\
        lens_output_all/Qwen__Qwen3.5-9B-Base/flores_ace_Arab_ace_Arab/lens_flores_ace_Arab_ace_Arab_sample0.json \\
        --lang-dist bak/token_lang_dist_fineweb_dataset_Qwen__Qwen3.5-9B-Base.json

    python visualize_english_prob.py lens.json --lang-dist dist.json --output eng_heatmap.png
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


def build_eng_prob_map(
    lang_dist: dict, tokenizer,
) -> tuple[dict[str, float], int, dict[str, str]]:
    """Build decoded_token_str -> P(eng | token) map from the lang dist JSON.

    Iterates all token_ids in the lang dist, decodes each to its string
    representation via tokenizer.decode([tid]), and maps that string to
    P(eng | token). When multiple token_ids decode to the same string,
    uses the weighted average (weighted by total_count) to merge them.

    Also returns M (number of languages) for the uniform fallback, and a
    decoded_str -> raw_bpe_str lookup table (via convert_ids_to_tokens).
    """
    meta = lang_dist.get("metadata", {})
    num_langs = meta.get("num_languages_detected", 0)

    tokens = lang_dist.get("tokens", {})

    # Phase 1: Collect (decoded_str, eng_prob, total_count) for each token_id
    str_to_engs: dict[str, list[tuple[float, float]]] = {}
    decoded_to_raw: dict[str, str] = {}
    for tid_str, entry in tokens.items():
        tid = int(tid_str)
        langs = entry.get("langs", {})
        eng = langs.get("eng_Latn", 0.0)
        total_count = entry.get("total_count", 0)
        decoded = tokenizer.decode([tid])
        str_to_engs.setdefault(decoded, []).append((eng, total_count))
        raw = tokenizer.convert_ids_to_tokens(tid)
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        decoded_to_raw.setdefault(decoded, raw)

    # Phase 2: Merge collisions via total_count-weighted average
    str_to_eng: dict[str, float] = {}
    for decoded, eng_count_list in str_to_engs.items():
        total_weight = sum(tc for _, tc in eng_count_list)
        if total_weight > 0:
            weighted_eng = sum(eng * tc for eng, tc in eng_count_list) / total_weight
        else:
            weighted_eng = sum(eng for eng, _ in eng_count_list) / len(eng_count_list)
        str_to_eng[decoded] = weighted_eng

    if not num_langs:
        all_langs: set[str] = set()
        for entry in tokens.values():
            all_langs.update(entry.get("langs", {}).keys())
        num_langs = len(all_langs)

    return str_to_eng, num_langs, decoded_to_raw


def lookup_eng_logprob(
    token_str: str,
    str_to_eng: dict[str, float],
    uniform_log: float,
    _cache: dict[str, float],
) -> float:
    """Look up log P(eng | token_str) from the decode-string lookup table.

    Returns log P(eng | t) if found, otherwise the uniform log-prior (1/M).
    Caches results per token string for efficiency.
    """
    if token_str in _cache:
        return _cache[token_str]

    eng = str_to_eng.get(token_str)
    if eng is not None:
        log_p = np.log(eng) if eng > 0 else -np.inf
    else:
        log_p = uniform_log

    _cache[token_str] = log_p
    return log_p


# --------------------------------------------------------------------------- #
#  Core computation
# --------------------------------------------------------------------------- #

def compute_eng_prob_matrix(
    lens_data: dict,
    str_to_eng: dict[str, float],
    num_langs: int,
    top_k: int,
    top_1_only: bool = False,
    tokenizer=None,
    decoded_to_raw: dict[str, str] | None = None,
    is_bpe: bool = True,
) -> tuple[list[str], list[str], np.ndarray, list[list[str]]]:
    """Compute the English-probability matrix.

    Returns:
        layer_names:  e.g. ["embed", "layer 0", ...]
        token_labels: e.g. ["Ace", "hn", ...]
        eng_probs:    (n_tokens, n_layers) array of P(eng) in [0, 1]
        top1_tokens:  (n_tokens, n_layers) list-of-lists of top-1 token strings
    """
    tokens = lens_data["tokens"]
    layer_names = [l["layer"] for l in tokens[0]["layers"]]
    n_tokens = len(tokens)
    n_layers = len(layer_names)

    eng_probs = np.zeros((n_tokens, n_layers), dtype=np.float64)
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
                    log_eng = lookup_eng_logprob(
                        top1, str_to_eng, uniform_log, _cache,
                    )
                    eng_probs[i, j] = np.exp(log_eng + np.log(prob))
                else:
                    eng_probs[i, j] = 0.0
                continue

            top1_tokens[i][j] = _short(_to_raw(top_preds[0]["token"], decoded_to_raw, is_bpe))

            if top_1_only:
                pred = top_preds[0]
                log_eng = lookup_eng_logprob(
                    pred["token"], str_to_eng, uniform_log, _cache,
                )
                log_prob = np.log(pred["prob"]) if pred["prob"] > 0 else -np.inf
                eng_probs[i, j] = np.exp(log_eng + log_prob)
                continue

            # Build log-space terms for logsumexp
            log_terms = np.empty(len(top_preds), dtype=np.float64)
            for k, pred in enumerate(top_preds):
                log_eng = lookup_eng_logprob(
                    pred["token"], str_to_eng, uniform_log, _cache,
                )
                log_prob = np.log(pred["prob"]) if pred["prob"] > 0 else -np.inf
                log_terms[k] = log_eng + log_prob

            eng_probs[i, j] = np.exp(logsumexp(log_terms))

    np.clip(eng_probs, 0.0, 1.0, out=eng_probs)

    return layer_names, token_labels, eng_probs, top1_tokens


def _compress_runs(s: str, min_run: int = 4) -> str:
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
    eng_probs: np.ndarray,
    top1_tokens: list[list[str]],
    output_path: str | None,
    dpi: int,
    cmap: str,
    figsize: tuple[float, float] | None,
):
    # Transpose: layers on y-axis, token positions on x-axis
    probs_t = eng_probs.T  # (n_layers, n_tokens)
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
        cbar_kws={"label": "P(English)"},
        linewidths=0.3,
        linecolor="#dddddd",
        annot_kws={
            "fontsize": 7,
            "ha": "center",
            "va": "center",
        },
    )

    ax.set_ylabel("Layer")
    ax.set_title("English Probability Across Layers")

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
    eng_probs: np.ndarray,
    output_path: str | None,
    dpi: int,
    figsize: tuple[float, float] | None,
):
    n_layers = eng_probs.shape[1]
    layer_avg = eng_probs.mean(axis=0)

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
    ax.set_ylabel("Mean P(English)")
    ax.set_xlabel("Layer")
    ax.set_title("Mean English Probability per Layer")
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
        description="Heatmap of P(English) across layers and token positions.",
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
        help="Output prefix. Two files will be created: <prefix>_eng.png and <prefix>_top1_eng.png "
             "(default: show interactively).",
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

    lang_dist = load_lang_dist(Path(args.lang_dist))

    # Load tokenizer from metadata
    tokenizer_name = lang_dist.get("metadata", {}).get("tokenizer", "")
    if not tokenizer_name:
        print("Error: no tokenizer in lang dist metadata", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {tokenizer_name}", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Build decoded_str -> P(eng) map
    str_to_eng, num_langs, decoded_to_raw = build_eng_prob_map(lang_dist, tokenizer)
    print(f"  {len(str_to_eng)} unique decoded token strings in lang dist, {num_langs} languages",
          file=sys.stderr)

    is_bpe = _is_bpe_tokenizer(tokenizer)
    print(f"  Tokenizer type: {'BPE' if is_bpe else 'SentencePiece'}", file=sys.stderr)

    figsize = tuple(args.figsize) if args.figsize else None

    for top_1_only, suffix in [(False, ""), (True, "top1_")]:
        mode_label = "top-1" if top_1_only else "top-k"
        print(f"\n--- Computing [{mode_label}] English probability matrix ---", file=sys.stderr)

        layer_names, token_labels, eng_probs, top1_tokens = compute_eng_prob_matrix(
            lens_data, str_to_eng, num_langs, args.top_k,
            top_1_only=top_1_only,
            tokenizer=tokenizer,
            decoded_to_raw=decoded_to_raw,
            is_bpe=is_bpe,
        )

        if args.output:
            output_path = f"{args.output}_{suffix}eng.png"
            line_path = f"{args.output}_{suffix}eng_line.png"
        else:
            output_path = None
            line_path = None

        plot_heatmap(
            layer_names, token_labels, eng_probs, top1_tokens,
            output_path=output_path,
            dpi=args.dpi,
            cmap=args.cmap,
            figsize=figsize,
        )

        plot_layer_avg(
            layer_names, eng_probs,
            output_path=line_path,
            dpi=args.dpi,
            figsize=figsize,
        )


if __name__ == "__main__":
    main()
