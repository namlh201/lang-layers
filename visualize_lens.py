#!/usr/bin/env python3
"""Visualise top-k predictions across layers and token positions from a single lens JSON.

Produces a heatmap: rows = token positions, columns = layers, cells = token strings
annotated with probability.  Each cell shows the top-1 predicted token and its prob.

Usage::

    python visualize_lens.py lens_flores_ace_Arab_ace_Arab_sample0.json
    python visualize_lens.py lens_output.json --top-k 5 --output heatmap.png
    python visualize_lens.py lens_output.json --dpi 200 --cmap YlOrRd
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

ALL_FONTS = font_manager.get_font_names()
ALL_NOTO_SANS_FONTS = [font for font in ALL_FONTS if "Noto Sans" in font or "Emoji" in font]

# print(ALL_NOTO_SANS_FONTS)

sns.set_style(rc={"font.family": ALL_NOTO_SANS_FONTS})


def load_lens(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_decoded_to_raw(tokenizer, vocab_size: int | None = None) -> dict[str, str]:
    """Build decoded_str -> raw_bpe_str lookup table for the full vocabulary."""
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    decoded_to_raw: dict[str, str] = {}
    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        raw = tokenizer.convert_ids_to_tokens(tid)
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        decoded_to_raw.setdefault(decoded, raw)
    return decoded_to_raw


def build_matrices(data: dict, top_k: int, tokenizer=None, decoded_to_raw: dict[str, str] | None = None, is_bpe: bool = True) -> tuple[list[str], list[str], np.ndarray, list[list[str]]]:
    """Extract layer names, token labels, prob matrix, and token-string grid.

    Returns:
        layer_names:  e.g. ["embed", "layer 0", ...]
        token_labels:  e.g. ["Ace", "hn", "é", ...]
        probs:         (n_tokens, n_layers) array of top-1 probs
        tokens_grid:   (n_tokens, n_layers) list-of-lists of top-1 token strings
    """
    tokens = data["tokens"]
    layer_names = [l["layer"] for l in tokens[0]["layers"]]
    n_tokens = len(tokens)
    n_layers = len(layer_names)

    probs = np.zeros((n_tokens, n_layers), dtype=np.float64)
    tokens_grid: list[list[str]] = [[""] * n_layers for _ in range(n_tokens)]

    for i, tok_data in enumerate(tokens):
        for j, layer_entry in enumerate(tok_data["layers"]):
            top_preds = layer_entry.get(f"top{top_k}", [])
            if top_preds:
                best = top_preds[0]
                probs[i, j] = best["prob"]
                tokens_grid[i][j] = _short(_to_raw(best["token"], decoded_to_raw, is_bpe))
            else:
                top1 = layer_entry.get("top1", "")
                prob = layer_entry.get("prob", 0.0)
                probs[i, j] = prob
                tokens_grid[i][j] = _short(_to_raw(top1, decoded_to_raw, is_bpe))

    token_labels = []
    for tok in tokens:
        token_str = tok.get("token", "")
        if tokenizer is not None and "token_id" in tok and not is_bpe:
            raw = tokenizer.convert_ids_to_tokens(tok["token_id"])
            if raw:
                token_str = raw
        elif is_bpe:
            token_str = token_str.replace(" ", "Ġ")
        token_labels.append(_truncate(token_str))

    return layer_names, token_labels, probs, tokens_grid


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


def plot_heatmap(
    layer_names: list[str],
    token_labels: list[str],
    probs: np.ndarray,
    tokens_grid: list[list[str]],
    top_k: int,
    output_path: str | None,
    cmap: str,
    dpi: int,
    figsize: tuple[float, float] | None,
):
    # Transpose so layers are rows (y-axis) and token positions are columns (x-axis)
    probs_t = probs.T          # (n_layers, n_tokens)
    n_tokens = probs_t.shape[1]
    n_layers = probs_t.shape[0]

    # Auto-size if not specified
    if figsize is None:
        col_w = max(1.5, min(1.5, 60.0 / n_tokens))
        row_h = 0.5
        figsize = (n_tokens * col_w, n_layers * row_h)

    fig, ax = plt.subplots(figsize=figsize)

    # Build annotation: token string + prob percentage (transposed)
    annot = np.empty((n_layers, n_tokens), dtype=object)
    for j in range(n_layers):
        for i in range(n_tokens):
            p = probs[i, j]
            tok = tokens_grid[i][j]
            if p > 0:
                annot[j, i] = f"{tok}\n{p:.2%}"
            else:
                annot[j, i] = tok

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
        cbar_kws={"label": "Top-1 Probability"},
        linewidths=0.3,
        linecolor="#dddddd",
        annot_kws={
            "fontsize": 11,
            "ha": "center",
            "va": "center",
            # "family": ['Noto Sans CJK JP', 'Noto Sans', 'Noto Sans Display', 'Noto Sans Arabic', 'Noto Sans Bengali', 'Noto Sans Devanagari', 'Noto Sans Hebrew', 'Noto Sans Malayalam', 'Noto Sans Thai', 'Noto Sans Tamil', 'Noto Sans Symbols']
            # "family": ALL_NOTO_SANS_FONTS
        },
    )

    ax.set_ylabel("Layer")
    ax.set_title(f"Top-{top_k} Lens Predictions (top-1 shown)")

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


def main():
    parser = argparse.ArgumentParser(
        description="Visualise top-k lens predictions as a layer x position heatmap.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to lens output JSON file.",
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
        help="Output image file (default: show interactively).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="YlOrRd",
        help="Seaborn/matplotlib colormap name (default: YlOrRd).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path for raw BPE token labels on x-axis (default: use decoded token strings).",
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

    data = load_lens(Path(args.input))
    if "tokens" not in data or not data["tokens"]:
        print("Error: no tokens in lens JSON", file=sys.stderr)
        sys.exit(1)

    tokenizer = None
    decoded_to_raw = None
    is_bpe = True
    if args.tokenizer:
        print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        is_bpe = _is_bpe_tokenizer(tokenizer)
        print(f"  Tokenizer type: {'BPE' if is_bpe else 'SentencePiece'}", file=sys.stderr)
        if not is_bpe:
            print(f"Building decoded->raw lookup table...", file=sys.stderr)
            decoded_to_raw = build_decoded_to_raw(tokenizer)

    layer_names, token_labels, probs, tokens_grid = build_matrices(
        data, args.top_k, tokenizer=tokenizer, decoded_to_raw=decoded_to_raw, is_bpe=is_bpe,
    )

    figsize = tuple(args.figsize) if args.figsize else None

    plot_heatmap(
        layer_names, token_labels, probs, tokens_grid,
        top_k=args.top_k,
        output_path=args.output,
        cmap=args.cmap,
        dpi=args.dpi,
        figsize=figsize,
    )


if __name__ == "__main__":
    main()
