#!/usr/bin/env python3
"""Trellis-based working language analysis across transformer layers.

For each layer, runs a forward (HMM-style) trellis across token positions,
using language-similarity transitions between adjacent positions' top-K
predicted tokens. The trellis captures temporal coherence: if position i
is strongly English-dominant, position i+1 is more likely to also be English.

For each layer *j*, the algorithm:
  1. Builds a K×M language matrix T_i for each position *i* from top-K predictions
  2. Computes K×K transition matrices A_i from language similarity (inner product)
  3. Runs a vectorised forward pass: alpha_{i+1} ∝ b_{i+1} ⊙ (A_i^T @ alpha_i)
  4. Projects to language space: lambda_i = alpha_i^T @ T_i
  5. Aggregates: working_lang = mean(lambdas)

Also runs Viterbi to find the most likely token sequence per layer.

Token strings in the lens JSON and the lang dist JSON are both produced via
``tokenizer.decode([tid])``, achieving a 100% match rate.

Usage::

    python visualize_trellis.py \\
        lens_output_all/Qwen__Qwen3.5-9B-Base/flores_eng_Latn_ace_Arab/lens_flores_eng_Latn_ace_Arab_sample0.json \\
        --lang-dist bak/token_lang_dist_fineweb_dataset_Qwen__Qwen3.5-9B-Base.json \\
        --output trellis
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
#  Trellis core algorithm
# --------------------------------------------------------------------------- #

def trellis_layer_analysis(
    p_lens: list[np.ndarray],
    lang_mats: list[np.ndarray],
    use_entropy_weight: bool = False,
    transition_mode: str = "row_norm",
    temperature: float = 1.0,
) -> dict:
    """Run trellis forward pass and Viterbi for a single layer.

    Parameters
    ----------
    p_lens : list of N np.ndarray, each (K,)
        Normalised lens probabilities at each position.
    lang_mats : list of N np.ndarray, each (K, M)
        Language distribution matrix at each position.
        Row k = P(L | token_k).
    use_entropy_weight : bool
        If True, emissions become b_i[k] = w_i[k] * p_i[k] where
        w_i[k] = 1 - H(T_i[k,:]) / log2(M).
    transition_mode : str
        "row_norm" or "softmax".
    temperature : float
        Temperature for softmax mode.

    Returns
    -------
    dict with keys:
        alphas : (N, K)      — forward state probabilities
        lambdas : (N, M)     — per-position language distributions
        transitions : list   — (N-1) transition matrices, each (K, K)
        working_lang : (M,)   — layer working language
        viterbi_path : (N,)  — most likely token path (indices)
        log_likelihood : float
        emissions : (N, K)   — normalised emission vectors b_i
        z_per_step : (N-1,)  — forward normalizers z_{i+1}
    """
    N = len(p_lens)
    K = p_lens[0].shape[0]
    M = lang_mats[0].shape[1]
    log_M = np.log2(M) if M > 1 else 1.0

    # --- Build emission vectors ---
    T_stack = np.stack(lang_mats)  # (N, K, M)
    p_stack = np.stack(p_lens)     # (N, K)

    if use_entropy_weight:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_T = np.where(T_stack > 0, np.log2(T_stack), 0.0)
        H = -np.sum(T_stack * log_T, axis=2)  # (N, K) — entropy in bits
        w = 1.0 - H / log_M  # (N, K)
        b_stack = w * p_stack
    else:
        b_stack = p_stack.copy()

    b_sums = b_stack.sum(axis=1, keepdims=True)
    b_sums = np.where(b_sums > 0, b_sums, 1.0)
    b_stack = b_stack / b_sums  # (N, K) — normalised

    # --- Compute transition matrices ---
    transitions = []
    for i in range(N - 1):
        S = lang_mats[i] @ lang_mats[i + 1].T  # (K, K)

        if transition_mode == "row_norm":
            row_sums = S.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            A = S / row_sums
        elif transition_mode == "softmax":
            S_scaled = S / temperature
            S_scaled = S_scaled - S_scaled.max(axis=1, keepdims=True)
            A = np.exp(S_scaled)
            A = A / A.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown transition_mode: {transition_mode}")
        transitions.append(A)

    # --- Forward pass ---
    alphas = np.zeros((N, K), dtype=np.float64)
    lambdas = np.zeros((N, M), dtype=np.float64)
    log_z = 0.0
    z_per_step = np.zeros(max(N - 1, 0), dtype=np.float64)

    alphas[0] = b_stack[0]
    lambdas[0] = alphas[0] @ lang_mats[0]

    for i in range(N - 1):
        A = transitions[i]
        pred = A.T @ alphas[i]              # (K,)
        alpha_unnorm = b_stack[i + 1] * pred  # (K,)
        z = alpha_unnorm.sum()
        alphas[i + 1] = alpha_unnorm / z if z > 0 else np.full(K, 1.0 / K)
        if z > 0:
            log_z += np.log(z)
            z_per_step[i] = z
        lambdas[i + 1] = alphas[i + 1] @ lang_mats[i + 1]

    working_lang = lambdas.mean(axis=0)

    # --- Viterbi ---
    deltas = np.zeros((N, K), dtype=np.float64)
    backpointers = np.zeros((N, K), dtype=int)
    deltas[0] = b_stack[0]

    for i in range(N - 1):
        A = transitions[i]
        scores = deltas[i][:, None] * A        # (K, K)
        backpointers[i + 1] = np.argmax(scores, axis=0)
        deltas[i + 1] = b_stack[i + 1] * scores.max(axis=0)

    path = np.zeros(N, dtype=int)
    path[-1] = np.argmax(deltas[-1])
    for i in range(N - 2, -1, -1):
        path[i] = backpointers[i + 1, path[i + 1]]

    return {
        "alphas": alphas,
        "lambdas": lambdas,
        "transitions": transitions,
        "working_lang": working_lang,
        "viterbi_path": path,
        "log_likelihood": log_z,
        "emissions": b_stack,
        "z_per_step": z_per_step,
    }


# --------------------------------------------------------------------------- #
#  Per-layer extraction from lens JSON
# --------------------------------------------------------------------------- #

def compute_trellis_all_layers(
    lens_data: dict,
    str_to_lang_vec: dict[str, np.ndarray],
    all_langs: list[str],
    num_langs: int,
    top_k: int,
    lang_codes: list[str],
    use_entropy_weight: bool = False,
    transition_mode: str = "row_norm",
    temperature: float = 1.0,
    top_1_only: bool = False,
    tokenizer=None,
    decoded_to_raw: dict[str, str] | None = None,
    is_bpe: bool = True,
) -> tuple[list[str], list[str], dict[str, np.ndarray], list[list[str]]]:
    """Run trellis for all layers, extract language probabilities.

    Returns:
        layer_names, token_labels, {lang_code: (n_tokens, n_layers)}, viterbi_tokens
    """
    tokens = lens_data["tokens"]
    layer_names = [l["layer"] for l in tokens[0]["layers"]]
    n_tokens = len(tokens)
    n_layers = len(layer_names)
    M = len(all_langs)
    lang_to_idx = {lc: i for i, lc in enumerate(all_langs)}
    uniform_vec = np.ones(M, dtype=np.float64) / M

    trellis_probs: dict[str, np.ndarray] = {
        lc: np.zeros((n_tokens, n_layers), dtype=np.float64) for lc in lang_codes
    }
    viterbi_tokens: list[list[str]] = [[""] * n_layers for _ in range(n_tokens)]
    token_labels: list[str] = []

    for i, tok_data in enumerate(tokens):
        token_str = tok_data.get("token", "")
        if tokenizer is not None and "token_id" in tok_data and not is_bpe:
            raw = tokenizer.convert_ids_to_tokens(tok_data["token_id"])
            if raw:
                token_str = raw
        elif is_bpe:
            token_str = token_str.replace(" ", "Ġ")
        token_labels.append(_truncate(token_str))

    n_found = 0
    n_missing = 0

    for j in range(n_layers):
        p_lens: list[np.ndarray] = []
        lang_mats: list[np.ndarray] = []
        pred_tokens_per_pos: list[list[str]] = []

        for i in range(n_tokens):
            layer_entry = tokens[i]["layers"][j]
            top_preds = layer_entry.get(f"top{top_k}", [])

            if not top_preds:
                top1 = layer_entry.get("top1", "")
                prob = layer_entry.get("prob", 0.0)
                if top1:
                    top_preds = [{"token": top1, "prob": prob if prob > 0 else 1.0}]
                else:
                    top_preds = [{"token": "", "prob": 1.0}]

            if top_1_only:
                top_preds = top_preds[:1]

            K = len(top_preds)

            p = np.array([pred.get("prob", 0.0) for pred in top_preds], dtype=np.float64)
            s = p.sum()
            p = p / s if s > 0 else np.ones(K) / K
            p_lens.append(p)

            T = np.zeros((K, M), dtype=np.float64)
            strs: list[str] = []
            for k, pred in enumerate(top_preds):
                vec = str_to_lang_vec.get(pred["token"])
                if vec is not None:
                    T[k] = vec
                    n_found += 1
                else:
                    T[k] = uniform_vec
                    n_missing += 1
                strs.append(pred["token"])
            lang_mats.append(T)
            pred_tokens_per_pos.append(strs)

        result = trellis_layer_analysis(
            p_lens, lang_mats,
            use_entropy_weight=use_entropy_weight,
            transition_mode=transition_mode,
            temperature=temperature,
        )

        lambdas = result["lambdas"]
        viterbi_path = result["viterbi_path"]

        for lc in lang_codes:
            idx = lang_to_idx.get(lc)
            if idx is not None:
                trellis_probs[lc][:, j] = lambdas[:, idx]

        for i in range(n_tokens):
            k = int(viterbi_path[i])
            if k < len(pred_tokens_per_pos[i]):
                raw_str = pred_tokens_per_pos[i][k]
                viterbi_tokens[i][j] = _short(_to_raw(raw_str, decoded_to_raw, is_bpe))

    if n_missing > 0:
        print(f"  Token lookup: {n_found} found, {n_missing} missing (using uniform fallback)",
              file=sys.stderr)

    for lc in lang_codes:
        np.clip(trellis_probs[lc], 0.0, 1.0, out=trellis_probs[lc])

    return layer_names, token_labels, trellis_probs, viterbi_tokens


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
    viterbi_tokens: list[list[str]],
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
    n_tokens = lang_probs[lang_order[0]].shape[0]
    n_layers = lang_probs[lang_order[0]].shape[1]

    probs_t = {lc: lang_probs[lc].T for lc in lang_order}
    cmap_objs = [plt.get_cmap(c) for c in cmaps]

    sub_images = []
    for idx, lc in enumerate(lang_order):
        rgba = cmap_objs[idx](probs_t[lc])
        rgba = np.repeat(rgba, block_h, axis=0)
        rgba = np.repeat(rgba, sub_w, axis=1)
        rgba = rgba.reshape(n_layers * block_h, n_tokens, sub_w, 4)
        sub_images.append(rgba)

    stacked = np.stack(sub_images, axis=2)
    img = stacked.reshape(n_layers * block_h, n_tokens * len(lang_order) * sub_w, 4)

    if figsize is None:
        col_w = max(2.0, min(2.0, 80.0 / n_tokens))
        row_h = 0.5
        figsize = (n_tokens * col_w, n_layers * row_h)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, aspect="auto", interpolation="nearest", origin="lower")

    cell_w = sub_w * len(lang_order)
    tick_positions = np.arange(n_tokens) * cell_w + cell_w / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_yticks(np.arange(n_layers) * block_h + block_h / 2)
    ax.set_yticklabels(layer_names, fontsize=9)

    shifted_labels = [""] + token_labels[:-1]
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(tick_positions)
    secax.set_xticklabels(shifted_labels, rotation=45, ha="right", fontsize=11)
    secax.set_xlabel("Token Position")

    for i in range(n_tokens + 1):
        ax.axvline(x=i * cell_w, color="#555555", linewidth=0.6)
    for j_layer in range(n_layers + 1):
        ax.axhline(y=j_layer * block_h, color="#555555", linewidth=0.6)
    for i in range(n_tokens):
        for s in range(1, len(lang_order)):
            ax.axvline(x=i * cell_w + s * sub_w, color="#bbbbbb", linewidth=0.3, linestyle="--")

    for i in range(n_tokens):
        for j_layer in range(n_layers):
            cx = i * cell_w + cell_w / 2
            cy = (j_layer + 0.5) * block_h

            ax.text(
                cx, cy - block_h * 0.22,
                viterbi_tokens[i][j_layer],
                ha="center", va="center",
                fontsize=fontsize, color="black",
            )

            for s, lc in enumerate(lang_order):
                rgba_val = cmap_objs[s](probs_t[lc][j_layer, i])
                ax.text(
                    i * cell_w + (s + 0.5) * sub_w,
                    cy + block_h * 0.22,
                    f"{probs_t[lc][j_layer, i]:.2%}",
                    ha="center", va="center",
                    fontsize=fontsize - 1,
                    color=_text_color(rgba_val),
                )

    ax.set_ylabel("Layer")
    ax.set_title("Trellis: " + " / ".join(lang_labels) + " — Probability Across Layers")

    legend_elements = [
        Patch(facecolor=cmap_objs[s](0.7), edgecolor="gray", label=lang_labels[s])
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
    ax.set_title("Trellis: " + " / ".join(lang_labels) + " — Mean Probability per Layer")
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
        description="Trellis-based working language heatmap across layers and positions.",
    )
    parser.add_argument("input", type=str, help="Path to lens output JSON file.")
    parser.add_argument("--lang-dist", type=str, required=True,
                        help="Path to token_lang_dist_fineweb_dataset_<model>.json.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Which top-k field to read from the JSON (default: 10).")
    parser.add_argument("--source-lang", type=str, default=None,
                        help="Override source language code (default: read from lens JSON).")
    parser.add_argument("--target-lang", type=str, default=None,
                        help="Override target language code (default: read from lens JSON).")
    parser.add_argument("--middle-lang", type=str, default="eng_Latn",
                        help="Middle language code (default: eng_Latn).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output prefix. Four files: <prefix>_trellis.png, "
                             "<prefix>_trellis_line.png, <prefix>_top1_trellis.png, "
                             "<prefix>_top1_trellis_line.png.")
    parser.add_argument("--cmap-src", type=str, default="Reds")
    parser.add_argument("--cmap-eng", type=str, default="Blues",
                        help="Colormap for middle language (default: Blues).")
    parser.add_argument("--cmap-tgt", type=str, default="Greens")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=float, nargs=2, default=None)
    parser.add_argument("--block-h", type=int, default=14)
    parser.add_argument("--sub-w", type=int, default=5)
    parser.add_argument("--fontsize", type=int, default=6)
    parser.add_argument("--entropy-weight", action="store_true",
                        help="Weight emissions by 1 - H(lang|token)/log2(M).")
    parser.add_argument("--transition-mode", type=str, default="row_norm",
                        choices=["row_norm", "softmax"],
                        help="Transition matrix normalisation (default: row_norm).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softmax transition mode (default: 1.0).")

    args = parser.parse_args()

    lens_data = load_lens(Path(args.input))
    if "tokens" not in lens_data or not lens_data["tokens"]:
        print("Error: no tokens in lens JSON", file=sys.stderr)
        sys.exit(1)

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

    lang_to_idx = {lc: i for i, lc in enumerate(all_langs)}
    for lc in lang_order:
        if lc not in lang_to_idx:
            print(f"  WARNING: language '{lc}' not in lang dist. "
                  f"Values will be 0.", file=sys.stderr)

    figsize = tuple(args.figsize) if args.figsize else None

    for top_1_only, suffix in [(False, ""), (True, "top1_")]:
        mode_label = "top-1" if top_1_only else "top-k"
        print(f"\n--- Computing [{mode_label}] trellis "
              f"(transition={args.transition_mode}, entropy_weight={args.entropy_weight}) ---",
              file=sys.stderr)

        layer_names, token_labels, all_probs, viterbi_tokens = compute_trellis_all_layers(
            lens_data, str_to_lang_vec, all_langs, num_langs, args.top_k,
            lang_codes=lang_order,
            use_entropy_weight=args.entropy_weight,
            transition_mode=args.transition_mode,
            temperature=args.temperature,
            top_1_only=top_1_only,
            tokenizer=tokenizer,
            decoded_to_raw=decoded_to_raw,
            is_bpe=is_bpe,
        )

        if args.output:
            heatmap_path = f"{args.output}_{suffix}trellis.png"
            line_path = f"{args.output}_{suffix}trellis_line.png"
        else:
            heatmap_path = None
            line_path = None

        plot_combined_heatmap(
            layer_names, token_labels, all_probs, viterbi_tokens,
            lang_order=lang_order,
            lang_labels=lang_labels,
            cmaps=cmaps,
            output_path=heatmap_path,
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
