#!/usr/bin/env python3
"""Visualise language distribution entropy across layers and token positions.

For each token position *i* and layer *j*, computes the Shannon entropy of
the marginal language distribution obtained by marginalising top-K lens
predictions over per-token language distributions:

    P(lang | h) = sum_k P(lang | t_k) * p_lens(t_k | h)

The per-token language distribution matrix (K x M) is thresholded at a cutoff
value (default 0.05): entries below the cutoff are zeroed out before
marginalisation.

Entropy is normalised to [0, 1] by dividing by log(M):

    H_norm(i, j) = H(i, j) / log(M)

A value of 0 means the distribution is perfectly concentrated on one language;
1 means maximum uncertainty (uniform across all M languages).

Usage::

    python visualize_entropy.py \\
        lens_output_all/Qwen__Qwen3.5-9B-Base/flores_eng_Latn_ace_Arab/lens_flores_eng_Latn_ace_Arab_sample0.json \\
        --lang-dist bak/token_lang_dist_fineweb_dataset_Qwen__Qwen3.5-9B-Base.json \\
        --output entropy_heatmap
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
from scipy.stats import entropy as scipy_entropy

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


# --------------------------------------------------------------------------- #
#  Build per-token language distribution vectors
# --------------------------------------------------------------------------- #

def build_lang_dist_matrix(
    lang_dist: dict, tokenizer,
) -> tuple[dict[str, np.ndarray], list[str], int, dict[str, str]]:
    """Build decoded_token_str -> language probability vector mapping.

    Iterates all token_ids in the lang dist once, building a vector of
    shape (M,) for each decoded token string, where M is the number of
    languages. When multiple token_ids decode to the same string, uses
    the total_count-weighted average to merge them.

    Returns:
        str_to_lang_vec: {decoded_str: np.ndarray(M,)}
        all_langs:       sorted list of language codes
        num_langs:       M (number of languages)
        decoded_to_raw:  decoded_str -> raw BPE string
    """
    meta = lang_dist.get("metadata", {})
    num_langs = meta.get("num_languages_detected", 0)

    # Build sorted list of all language codes
    all_langs_list = meta.get("languages_detected", [])
    if not all_langs_list:
        all_langs_set: set[str] = set()
        for entry in lang_dist.get("tokens", {}).values():
            all_langs_set.update(entry.get("langs", {}).keys())
        all_langs_list = sorted(all_langs_set)

    all_langs = sorted(all_langs_list)
    lang_to_idx = {lc: i for i, lc in enumerate(all_langs)}
    M = len(all_langs)

    if not num_langs:
        num_langs = M

    tokens = lang_dist.get("tokens", {})

    # Phase 1: Collect per-token (decoded_str, langs_dict, total_count)
    str_to_langs: dict[str, list[tuple[dict[str, float], int]]] = {}
    decoded_to_raw: dict[str, str] = {}

    for tid_str, entry in tokens.items():
        tid = int(tid_str)
        langs = entry.get("langs", {})
        total_count = entry.get("total_count", 0)
        decoded = tokenizer.decode([tid])
        str_to_langs.setdefault(decoded, []).append((langs, total_count))

        raw = tokenizer.convert_ids_to_tokens(tid)
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        decoded_to_raw.setdefault(decoded, raw)

    # Phase 2: Merge collisions via total_count-weighted average
    str_to_lang_vec: dict[str, np.ndarray] = {}
    for decoded, lang_count_list in str_to_langs.items():
        total_weight = sum(tc for _, tc in lang_count_list)
        vec = np.zeros(M, dtype=np.float64)
        if total_weight > 0:
            for langs, tc in lang_count_list:
                for lc, prob in langs.items():
                    vec[lang_to_idx[lc]] += prob * tc / total_weight
        else:
            for langs, _ in lang_count_list:
                for lc, prob in langs.items():
                    vec[lang_to_idx[lc]] += prob / len(lang_count_list)
        s = vec.sum()
        if s > 0:
            vec /= s
        str_to_lang_vec[decoded] = vec

    return str_to_lang_vec, all_langs, num_langs, decoded_to_raw


# --------------------------------------------------------------------------- #
#  Core computation
# --------------------------------------------------------------------------- #

def compute_entropy_matrix(
    lens_data: dict,
    str_to_lang_vec: dict[str, np.ndarray],
    all_langs: list[str],
    num_langs: int,
    top_k: int,
    cutoff: float = 0.05,
    top_1_only: bool = False,
    tokenizer=None,
    decoded_to_raw: dict[str, str] | None = None,
    is_bpe: bool = True,
) -> tuple[list[str], list[str], np.ndarray, list[list[str]]]:
    """Compute the normalised entropy matrix.

    For each (token_position, layer):
      1. Collect top-K lens predictions: p_lens (K,), each with a token string
      2. Look up each token's language distribution: lang_matrix (K, M)
      3. Threshold: lang_matrix[lang_matrix < cutoff] = 0
      4. Marginalise: marginal = p_lens @ lang_matrix  -> (M,)
      5. Renormalise marginal to sum to 1
      6. H = -sum(marginal * log(marginal)) for marginal > 0
      7. H_norm = H / log(M)

    Returns:
        layer_names, token_labels, entropy (n_tokens, n_layers), top1_tokens
    """
    tokens = lens_data["tokens"]
    layer_names = [l["layer"] for l in tokens[0]["layers"]]
    n_tokens = len(tokens)
    n_layers = len(layer_names)
    M = len(all_langs)
    log_M = 1.0 # np.log(M) if M > 1 else 1.0

    entropy = np.zeros((n_tokens, n_layers), dtype=np.float64)
    top1_tokens: list[list[str]] = [[""] * n_layers for _ in range(n_tokens)]
    token_labels: list[str] = []
    uniform_vec = np.ones(M, dtype=np.float64) / M

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
                print("top_preds", layer_entry.get(f"top{top_k}", []))
                top1 = layer_entry.get("top1", "")
                prob = layer_entry.get("prob", 0.0)
                top1_tokens[i][j] = _short(_to_raw(top1, decoded_to_raw, is_bpe))
                if top1 and prob > 0:
                    vec = str_to_lang_vec.get(top1, uniform_vec).copy()
                    vec[vec < cutoff] = 0
                    marginal = vec * prob
                    total = marginal.sum()
                    if total > 0:
                        # marginal /= total
                        H = scipy_entropy(marginal)
                        entropy[i, j] = H / log_M
                    else:
                        entropy[i, j] = 1.0
                continue

            top1_tokens[i][j] = _short(_to_raw(top_preds[0]["token"], decoded_to_raw, is_bpe))

            if top_1_only:
                pred = top_preds[0]
                vec = str_to_lang_vec.get(pred["token"], uniform_vec).copy()
                vec[vec < cutoff] = 0
                marginal = vec * pred["prob"]
                total = marginal.sum()
                if total > 0:
                    # print("top 1")
                    # print("total", total)
                    # print("marginal", marginal)
                    # marginal /= total
                    H = scipy_entropy(marginal)
                    entropy[i, j] = H / log_M
                else:
                    entropy[i, j] = 0.0
                continue

            # Build K x M matrix
            k = len(top_preds)
            p_lens = np.empty(k, dtype=np.float64)
            lang_matrix = np.empty((k, M), dtype=np.float64)
            for idx, pred in enumerate(top_preds):
                p_lens[idx] = pred["prob"]
                vec = str_to_lang_vec.get(pred["token"])
                if vec is not None:
                    lang_matrix[idx] = vec
                else:
                    lang_matrix[idx] = uniform_vec

            # Threshold
            lang_matrix[lang_matrix < cutoff] = 0

            # Marginalise: (M,) = (K,) @ (K, M)
            marginal = p_lens @ lang_matrix

            # Renormalise
            total = marginal.sum()
            if total > 0:
                # print("p_lens", p_lens)
                # print("lang_matrix", lang_matrix)
                # print("total", total)
                # print("marginal", marginal)
                # marginal /= total
                H = scipy_entropy(marginal)
                entropy[i, j] = H / log_M
            else:
                # print("p_lens", p_lens)
                # print("lang_matrix", lang_matrix)
                # print("total", total)
                # print("marginal", marginal)
                entropy[i, j] = 0.0

    np.clip(entropy, 0.0, 1.0, out=entropy)

    return layer_names, token_labels, entropy, top1_tokens


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
    entropy: np.ndarray,
    top1_tokens: list[list[str]],
    output_path: str | None,
    dpi: int,
    cmap: str,
    figsize: tuple[float, float] | None,
):
    probs_t = entropy.T  # (n_layers, n_tokens)
    n_tokens = probs_t.shape[1]
    n_layers = probs_t.shape[0]

    if figsize is None:
        col_w = max(1.5, min(1.5, 60.0 / n_tokens))
        row_h = 0.5
        figsize = (n_tokens * col_w, n_layers * row_h)

    fig, ax = plt.subplots(figsize=figsize)

    annot = np.empty((n_layers, n_tokens), dtype=object)
    for j in range(n_layers):
        for i in range(n_tokens):
            annot[j, i] = f"{top1_tokens[i][j]}\n{probs_t[j, i]:.2f}"

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
        cbar_kws={"label": "Normalised Entropy"},
        linewidths=0.3,
        linecolor="#dddddd",
        annot_kws={
            "fontsize": 7,
            "ha": "center",
            "va": "center",
        },
    )

    ax.set_ylabel("Layer")
    ax.set_title("Language Distribution Entropy Across Layers")

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=45, labelsize=11)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

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
    entropy: np.ndarray,
    output_path: str | None,
    dpi: int,
    figsize: tuple[float, float] | None,
):
    n_layers = entropy.shape[1]
    layer_avg = entropy.mean(axis=0)

    if figsize is None:
        figsize = (max(8, n_layers * 0.4), 5)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_layers)
    ax.plot(x, layer_avg, marker="o", markersize=3, linewidth=1.5, color="#d62728")
    ax.fill_between(x, layer_avg, alpha=0.15, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=9)
    y_max = min(layer_avg.max() + 0.1, 1.0)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Mean Normalised Entropy")
    ax.set_xlabel("Layer")
    ax.set_title("Mean Language Distribution Entropy per Layer")
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
        description="Heatmap of language distribution entropy across layers and positions.",
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
        "--cutoff",
        type=float,
        default=0.05,
        help="Threshold for the per-token language distribution matrix: "
             "entries below this value are zeroed (default: 0.05).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output prefix. Four files will be created: <prefix>_entropy.png, "
             "<prefix>_entropy_line.png, <prefix>_top1_entropy.png, "
             "<prefix>_top1_entropy_line.png (default: show interactively).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="YlOrRd",
        help="Seaborn/matplotlib colormap name (default: YlOrRd).",
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

    lens_data = load_lens(Path(args.input))
    if "tokens" not in lens_data or not lens_data["tokens"]:
        print("Error: no tokens in lens JSON", file=sys.stderr)
        sys.exit(1)

    lang_dist = load_lang_dist(Path(args.lang_dist))

    tokenizer_name = lang_dist.get("metadata", {}).get("tokenizer", "")
    if not tokenizer_name:
        print("Error: no tokenizer in lang dist metadata", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {tokenizer_name}", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    str_to_lang_vec, all_langs, num_langs, decoded_to_raw = build_lang_dist_matrix(
        lang_dist, tokenizer,
    )
    print(f"  {len(str_to_lang_vec)} unique decoded token strings, {num_langs} languages",
          file=sys.stderr)

    is_bpe = _is_bpe_tokenizer(tokenizer)
    print(f"  Tokenizer type: {'BPE' if is_bpe else 'SentencePiece'}", file=sys.stderr)

    figsize = tuple(args.figsize) if args.figsize else None

    for top_1_only, suffix in [(False, ""), (True, "top1_")]:
        mode_label = "top-1" if top_1_only else "top-k"
        print(f"\n--- Computing [{mode_label}] entropy matrix (cutoff={args.cutoff}) ---",
              file=sys.stderr)

        layer_names, token_labels, entropy, top1_tokens = compute_entropy_matrix(
            lens_data, str_to_lang_vec, all_langs, num_langs, args.top_k,
            cutoff=args.cutoff,
            top_1_only=top_1_only,
            tokenizer=tokenizer,
            decoded_to_raw=decoded_to_raw,
            is_bpe=is_bpe,
        )

        if args.output:
            heatmap_path = f"{args.output}_{suffix}entropy.png"
            line_path = f"{args.output}_{suffix}entropy_line.png"
        else:
            heatmap_path = None
            line_path = None

        plot_heatmap(
            layer_names, token_labels, entropy, top1_tokens,
            output_path=heatmap_path,
            dpi=args.dpi,
            cmap=args.cmap,
            figsize=figsize,
        )

        plot_layer_avg(
            layer_names, entropy,
            output_path=line_path,
            dpi=args.dpi,
            figsize=figsize,
        )


if __name__ == "__main__":
    main()
