#!/usr/bin/env python3
"""Visualise combined source/middle/target language-probability heatmap.

For each token position *i* and layer *j*, computes three probabilities:
  - P(source_lang):  probability the prediction is in the source language
  - P(middle_lang):   probability the prediction is in the middle language
  - P(target_lang):  probability the prediction is in the target language

Each cell in the heatmap is divided into three vertical sub-cells, each
colored by its respective probability using a different colormap:

    +----------+----------+----------+
    |  Source  | Middle   |  Target  |
    +----------+----------+----------+

Annotations show the top-1 predicted token string (centered) and three
percentages (one per sub-cell).

Usage::

    python visualize_combined.py \\
        lens_output_all/Qwen__Qwen3.5-9B-Base/flores_eng_Latn_ace_Arab/lens_flores_eng_Latn_ace_Arab_sample0.json \\
        --lang-dist bak/token_lang_dist_fineweb_dataset_Qwen__Qwen3.5-9B-Base.json \\
        --output combined_heatmap.png

    # Override languages
    python visualize_combined.py lens.json --lang-dist dist.json --output out.png --source-lang eng_Latn --target-lang arb_Arab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from scipy.special import logsumexp

ALL_FONTS = font_manager.get_font_names()
ALL_NOTO_SANS_FONTS = [font for font in ALL_FONTS if "Noto Sans" in font or "Emoji" in font]

sns.set_style(rc={"font.family": ALL_NOTO_SANS_FONTS})

ENG_CODE = "eng_Latn"  # kept for backwards reference; use --middle-lang


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #

def load_lens(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_lang_dist(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Build lang prob maps (all three in one pass)
# --------------------------------------------------------------------------- #

def build_multi_lang_prob_map(
    lang_dist: dict, tokenizer, lang_codes: list[str],
) -> tuple[dict[str, dict[str, float]], int, dict[str, str]]:
    """Build decoded_token_str -> P(lang | token) for multiple languages at once.

    Iterates all token_ids in the lang dist once, building a prob map for each
    requested language code. Also builds a decoded_str -> raw_bpe_str lookup
    table.

    Returns:
        lang_maps:     {lang_code: {decoded_str: P(lang | token)}}
        num_langs:     M for the uniform fallback
        decoded_to_raw: decoded_str -> raw BPE string
    """
    meta = lang_dist.get("metadata", {})
    num_langs = meta.get("num_languages_detected", 0)

    tokens = lang_dist.get("tokens", {})

    # Phase 1: Collect per-language (decoded_str, lang_prob, total_count)
    lang_accum: dict[str, dict[str, list[tuple[float, float]]]] = {
        lc: {} for lc in lang_codes
    }
    decoded_to_raw: dict[str, str] = {}

    for tid_str, entry in tokens.items():
        tid = int(tid_str)
        langs = entry.get("langs", {})
        total_count = entry.get("total_count", 0)
        decoded = tokenizer.decode([tid])

        for lc in lang_codes:
            lang_prob = langs.get(lc, 0.0)
            lang_accum[lc].setdefault(decoded, []).append((lang_prob, total_count))

        raw = tokenizer.convert_ids_to_tokens(tid)
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        decoded_to_raw.setdefault(decoded, raw)

    # Phase 2: Merge collisions via total_count-weighted average
    lang_maps: dict[str, dict[str, float]] = {}
    for lc in lang_codes:
        str_to_lang: dict[str, float] = {}
        for decoded, lang_count_list in lang_accum[lc].items():
            total_weight = sum(tc for _, tc in lang_count_list)
            if total_weight > 0:
                weighted = sum(lp * tc for lp, tc in lang_count_list) / total_weight
            else:
                weighted = sum(lp for lp, _ in lang_count_list) / len(lang_count_list)
            str_to_lang[decoded] = weighted
        lang_maps[lc] = str_to_lang

    if not num_langs:
        all_langs: set[str] = set()
        for entry in tokens.values():
            all_langs.update(entry.get("langs", {}).keys())
        num_langs = len(all_langs)

    return lang_maps, num_langs, decoded_to_raw


def _lookup_logprob(
    token_str: str,
    str_to_lang: dict[str, float],
    uniform_log: float,
    _cache: dict[str, float],
) -> float:
    """Look up log P(lang | token_str). Returns uniform log-prior if not found."""
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
#  Core computation (all three languages in one pass)
# --------------------------------------------------------------------------- #

def compute_multi_lang_prob_matrix(
    lens_data: dict,
    lang_maps: dict[str, dict[str, float]],
    lang_order: list[str],
    num_langs: int,
    top_k: int,
    top_1_only: bool = False,
    tokenizer=None,
    decoded_to_raw: dict[str, str] | None = None,
    is_bpe: bool = True,
) -> tuple[list[str], list[str], dict[str, np.ndarray], list[list[str]]]:
    """Compute probability matrices for multiple languages in one pass.

    Args:
        lang_order: ordered list of lang codes (e.g. [src, eng, tgt]).
                    The returned dict will have keys in this order.

    Returns:
        layer_names, token_labels, {lang_code: probs_array}, top1_tokens
    """
    tokens = lens_data["tokens"]
    layer_names = [l["layer"] for l in tokens[0]["layers"]]
    n_tokens = len(tokens)
    n_layers = len(layer_names)

    lang_probs: dict[str, np.ndarray] = {
        lc: np.zeros((n_tokens, n_layers), dtype=np.float64) for lc in lang_order
    }
    top1_tokens: list[list[str]] = [[""] * n_layers for _ in range(n_tokens)]
    token_labels: list[str] = []
    uniform_log = -np.log(num_langs) if num_langs > 0 else -np.inf
    caches: dict[str, dict[str, float]] = {lc: {} for lc in lang_order}

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
                top1 = layer_entry.get("top1", "")
                prob = layer_entry.get("prob", 0.0)
                top1_tokens[i][j] = _short(_to_raw(top1, decoded_to_raw, is_bpe))
                if top1 and prob > 0:
                    for lc in lang_order:
                        log_lang = _lookup_logprob(
                            top1, lang_maps[lc], uniform_log, caches[lc],
                        )
                        lang_probs[lc][i, j] = np.exp(log_lang + np.log(prob))
                continue

            top1_tokens[i][j] = _short(_to_raw(top_preds[0]["token"], decoded_to_raw, is_bpe))

            if top_1_only:
                pred = top_preds[0]
                log_prob = np.log(pred["prob"]) if pred["prob"] > 0 else -np.inf
                for lc in lang_order:
                    log_lang = _lookup_logprob(
                        pred["token"], lang_maps[lc], uniform_log, caches[lc],
                    )
                    lang_probs[lc][i, j] = np.exp(log_lang + log_prob)
                continue

            # Build log-space terms for logsumexp — iterate preds once, compute for all langs
            n_preds = len(top_preds)
            log_probs = np.empty(n_preds, dtype=np.float64)
            for k, pred in enumerate(top_preds):
                log_probs[k] = np.log(pred["prob"]) if pred["prob"] > 0 else -np.inf

            for lc in lang_order:
                log_terms = np.empty(n_preds, dtype=np.float64)
                for k, pred in enumerate(top_preds):
                    log_lang = _lookup_logprob(
                        pred["token"], lang_maps[lc], uniform_log, caches[lc],
                    )
                    log_terms[k] = log_lang + log_probs[k]
                lang_probs[lc][i, j] = np.exp(logsumexp(log_terms))

    for lc in lang_order:
        np.clip(lang_probs[lc], 0.0, 1.0, out=lang_probs[lc])

    return layer_names, token_labels, lang_probs, top1_tokens


# --------------------------------------------------------------------------- #
#  Text helpers
# --------------------------------------------------------------------------- #

def _compress_runs(s: str, min_run: int = 3) -> str:
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


def _text_color(rgba: tuple) -> str:
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #

def plot_combined_heatmap(
    layer_names: list[str],
    token_labels: list[str],
    lang_probs: dict[str, np.ndarray],
    top1_tokens: list[list[str]],
    lang_order: list[str],
    lang_labels: list[str],
    cmaps: list[str],
    output_path: str | None,
    dpi: int,
    figsize: tuple[float, float] | None,
    block_h: int = 14,
    sub_w: int = 5,
    fontsize: int = 6,
):
    """Plot a single combined heatmap with three sub-cells per cell.

    Args:
        lang_order:   ordered list of lang codes (e.g. [src, eng, tgt])
        lang_labels:  human-readable labels matching lang_order
        cmaps:        colormap names matching lang_order
        block_h:      pixel height per cell
        sub_w:        pixel width per sub-cell (total cell width = sub_w * 3)
    """
    n_tokens = lang_probs[lang_order[0]].shape[0]
    n_layers = lang_probs[lang_order[0]].shape[1]

    # Transpose to (n_layers, n_tokens)
    probs_t = {lc: lang_probs[lc].T for lc in lang_order}

    # Get colormap objects
    cmap_objs = [plt.get_cmap(c) for c in cmaps]

    # Build combined RGBA image via vectorised operations
    # Each probs_t[lc] has shape (n_layers, n_tokens)
    sub_images = []
    for idx, lc in enumerate(lang_order):
        rgba = cmap_objs[idx](probs_t[lc])  # (n_layers, n_tokens, 4)
        # Expand: repeat block_h times vertically, sub_w times horizontally
        rgba = np.repeat(rgba, block_h, axis=0)    # (n_layers * block_h, n_tokens, 4)
        rgba = np.repeat(rgba, sub_w, axis=1)      # (n_layers * block_h, n_tokens * sub_w, 4)
        # Reshape to isolate per-token blocks: (H, n_tokens, sub_w, 4)
        rgba = rgba.reshape(n_layers * block_h, n_tokens, sub_w, 4)
        sub_images.append(rgba)

    # Stack along a new axis at position 2: (H, n_tokens, n_sub, sub_w, 4)
    stacked = np.stack(sub_images, axis=2)

    # Reshape to final image: (H, n_tokens * n_sub * sub_w, 4)
    img = stacked.reshape(n_layers * block_h, n_tokens * len(lang_order) * sub_w, 4)

    # Auto-size
    if figsize is None:
        col_w = max(2.0, min(2.0, 80.0 / n_tokens))
        row_h = 0.5
        figsize = (n_tokens * col_w, n_layers * row_h)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, aspect="auto", interpolation="nearest", origin="lower")

    # Ticks
    cell_w = sub_w * len(lang_order)
    tick_positions = np.arange(n_tokens) * cell_w + cell_w / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_yticks(np.arange(n_layers) * block_h + block_h / 2)
    ax.set_yticklabels(layer_names, fontsize=9)

    # Bottom x-axis: shifted right by one position
    shifted_labels = [""] + token_labels[:-1]
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(tick_positions)
    secax.set_xticklabels(shifted_labels, rotation=45, ha="right", fontsize=11)
    secax.set_xlabel("Token Position")

    # Grid: major lines between cells
    for i in range(n_tokens + 1):
        ax.axvline(x=i * cell_w, color="#555555", linewidth=0.6)
    for j in range(n_layers + 1):
        ax.axhline(y=j * block_h, color="#555555", linewidth=0.6)

    # Grid: minor lines between sub-cells
    for i in range(n_tokens):
        for s in range(1, len(lang_order)):
            ax.axvline(x=i * cell_w + s * sub_w, color="#bbbbbb", linewidth=0.3, linestyle="--")

    # Annotations
    for i in range(n_tokens):
        for j in range(n_layers):
            cx = i * cell_w + cell_w / 2
            cy = (j + 0.5) * block_h

            # Token string at top of cell
            ax.text(
                cx, cy - block_h * 0.22,
                top1_tokens[i][j],
                ha="center", va="center",
                fontsize=fontsize, color="black",
            )

            # Percentages in each sub-cell
            for s, lc in enumerate(lang_order):
                rgba_val = cmap_objs[s](probs_t[lc][j, i])
                ax.text(
                    i * cell_w + (s + 0.5) * sub_w,
                    cy + block_h * 0.22,
                    f"{probs_t[lc][j, i]:.2%}",
                    ha="center", va="center",
                    fontsize=fontsize - 1,
                    color=_text_color(rgba_val),
                )

    ax.set_ylabel("Layer")
    ax.set_title(" / ".join(lang_labels) + " — Probability Across Layers")

    # Legend
    legend_elements = [
        Patch(
            facecolor=cmap_objs[s](0.7),
            edgecolor="gray",
            label=lang_labels[s],
        )
        for s in range(len(lang_order))
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}", file=sys.stderr)
    else:
        plt.show()
    plt.close(fig)


def plot_layer_avg_multi(
    layer_names: list[str],
    lang_probs: dict[str, np.ndarray],
    lang_order: list[str],
    lang_labels: list[str],
    cmaps: list[str],
    output_path: str | None,
    dpi: int,
    figsize: tuple[float, float] | None,
):
    n_layers = lang_probs[lang_order[0]].shape[1]

    if figsize is None:
        figsize = (max(8, n_layers * 0.4), 5)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_layers)
    colors = [plt.get_cmap(c)(0.7) for c in cmaps]

    for idx, lc in enumerate(lang_order):
        layer_avg = lang_probs[lc].mean(axis=0)
        ax.plot(x, layer_avg, marker="o", markersize=3, linewidth=1.5,
                color=colors[idx], label=lang_labels[idx])
        ax.fill_between(x, layer_avg, alpha=0.1, color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=9)
    all_max = max(lang_probs[lc].mean(axis=0).max() for lc in lang_order)
    y_max = min(all_max + 0.1, 1.0)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Mean Probability")
    ax.set_xlabel("Layer")
    ax.set_title(" / ".join(lang_labels) + " — Mean Probability per Layer")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

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
        description="Combined heatmap of P(source), P(middle), P(target) across layers.",
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
        "--middle-lang",
        type=str,
        default="eng_Latn",
        help="Middle language code (default: eng_Latn).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output prefix. Two files will be created: <prefix>_combined.png and "
             "<prefix>_top1_combined.png (default: show interactively).",
    )
    parser.add_argument(
        "--cmap-src",
        type=str,
        default="Reds",
        help="Colormap for source language (default: Reds).",
    )
    parser.add_argument(
        "--cmap-eng",
        type=str,
        default="Blues",
        help="Colormap for middle language (default: Blues).",
    )
    parser.add_argument(
        "--cmap-tgt",
        type=str,
        default="Greens",
        help="Colormap for target language (default: Greens).",
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
    parser.add_argument(
        "--block-h",
        type=int,
        default=14,
        help="Pixel height per cell in the image (default: 14).",
    )
    parser.add_argument(
        "--sub-w",
        type=int,
        default=5,
        help="Pixel width per sub-cell (default: 5). Total cell width = sub_w * 3.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=6,
        help="Fontsize for annotations (default: 6).",
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

    if not source_lang:
        print("Error: no source_lang in lens JSON and none provided via CLI", file=sys.stderr)
        sys.exit(1)
    if not target_lang:
        print("Error: no target_lang in lens JSON and none provided via CLI", file=sys.stderr)
        sys.exit(1)

    lang_order = [source_lang, args.middle_lang, target_lang]
    lang_labels = [
        f"Source ({source_lang})",
        f"Middle ({args.middle_lang})",
        f"Target ({target_lang})",
    ]
    cmaps = [args.cmap_src, args.cmap_eng, args.cmap_tgt]

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

    # Build lang prob maps for all unique languages in one pass
    unique_langs = list(dict.fromkeys(lang_order))  # preserve order, remove dups
    print(f"Building prob maps for: {unique_langs}", file=sys.stderr)
    lang_maps, num_langs, decoded_to_raw = build_multi_lang_prob_map(
        lang_dist, tokenizer, unique_langs,
    )
    print(f"  {num_langs} languages detected in lang dist", file=sys.stderr)

    for lc in unique_langs:
        has_nonzero = any(v > 0 for v in lang_maps[lc].values())
        if not has_nonzero:
            print(f"  WARNING: language '{lc}' not found in any token's lang distribution. "
                  f"Heatmap will show near-zero values (uniform prior only).", file=sys.stderr)

    figsize = tuple(args.figsize) if args.figsize else None

    for top_1_only, suffix in [(False, ""), (True, "top1_")]:
        mode_label = "top-1" if top_1_only else "top-k"
        print(f"\nComputing [{mode_label}] probability matrices...", file=sys.stderr)

        layer_names, token_labels, all_probs, top1_tokens = compute_multi_lang_prob_matrix(
            lens_data, lang_maps, lang_order, num_langs, args.top_k,
            top_1_only=top_1_only,
            tokenizer=tokenizer,
            decoded_to_raw=decoded_to_raw,
            is_bpe=is_bpe,
        )

        if args.output:
            output_path = f"{args.output}_{suffix}combined.png"
            line_path = f"{args.output}_{suffix}combined_line.png"
        else:
            output_path = None
            line_path = None

        plot_combined_heatmap(
            layer_names, token_labels, all_probs, top1_tokens,
            lang_order=lang_order,
            lang_labels=lang_labels,
            cmaps=cmaps,
            output_path=output_path,
            dpi=args.dpi,
            figsize=figsize,
            block_h=args.block_h,
            sub_w=args.sub_w,
            fontsize=args.fontsize,
        )

        plot_layer_avg_multi(
            layer_names, all_probs,
            lang_order=lang_order,
            lang_labels=lang_labels,
            cmaps=cmaps,
            output_path=line_path,
            dpi=args.dpi,
            figsize=figsize,
        )


if __name__ == "__main__":
    main()
