#!/usr/bin/env python3
"""Language switch detection and layer specificity metrics from the trellis.

Implements ``switch_and_layer_metrics.md`` on top of the trellis forward pass
of ``visualize_trellis.py``:

Switch metrics (per boundary i, per layer j):
  * evidence      d_i = TV(lambda_{i+1}, lambda_i)
  * commitment    kappa_i = (max_m lambda_i[m] - 1/M) / (1 - 1/M)
  * persistence   rho_i  = mean posterior mass on the post-switch language
  * confidence    c_i    = d_i * min(kappa_i, kappa_{i+1}) * rho_i
  * smoothed posterior P(switch | all evidence) via a backward pass
  * likelihood-ratio statistic G_i = log(K * z_{i+1})

Layer metrics (per layer j):
  * commitment    C_j = mean kappa_i
  * tracking      T_j = mean lambda_i[g_i] against the actual token language,
                  with a permutation null (mixed reference) or bootstrap
                  against the uniform baseline (constant reference)
  * coherence     D_j = 2 * (ell_j + (N-1) ln K)   (ground-truth free)

Cross-layer:
  * consensus confidence (Z-weighted) and switch onset depth

Usage::

    python analyze_switch_layers.py \\
        lens_output_all/Qwen__Qwen3.5-9B-Base/flores_eng_Latn_ace_Arab/lens_flores_eng_Latn_ace_Arab_sample0.json \\
        --lang-dist bak/token_lang_dist_fineweb_dataset_Qwen__Qwen3.5-9B-Base.json \\
        --output switch
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import norm as sps_norm

from visualize_trellis import (
    _is_bpe_tokenizer,
    _truncate,
    build_lang_dist_matrix,
    load_lang_dist,
    load_lens,
    trellis_layer_analysis,
)

CLASS_COLORS = {
    "specific": "tab:green",
    "committed-no-track": "tab:red",
    "weakly-tracking": "tab:orange",
    "neutral": "tab:gray",
}


# --------------------------------------------------------------------------- #
#  Ground truth (Definition 2)
# --------------------------------------------------------------------------- #

def ground_truth_languages(
    lens_data: dict,
    str_to_lang_vec: dict[str, np.ndarray],
    all_langs: list[str],
) -> np.ndarray:
    """g_i = argmax language of the actual generated token's distribution.

    Returns an (N,) int array with -1 where the token is missing from the
    lang dist database or its distribution is uniform (ambiguous).
    """
    M = len(all_langs)
    g = np.full(len(lens_data["tokens"]), -1, dtype=int)
    for i, tok in enumerate(lens_data["tokens"]):
        vec = str_to_lang_vec.get(tok.get("token", ""))
        if vec is None:
            continue
        if vec.max() <= 1.0 / M + 1e-9:
            continue
        g[i] = int(np.argmax(vec))
    return g


# --------------------------------------------------------------------------- #
#  Per-layer input extraction
# --------------------------------------------------------------------------- #

def build_layer_inputs(
    lens_data: dict,
    str_to_lang_vec: dict[str, np.ndarray],
    all_langs: list[str],
    top_k: int,
    layer_idx: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract (p_lens, lang_mats) for one layer, mirroring the top-k loop
    of visualize_trellis.compute_trellis_all_layers."""
    tokens = lens_data["tokens"]
    M = len(all_langs)
    uniform = np.full(M, 1.0 / M)
    p_lens: list[np.ndarray] = []
    lang_mats: list[np.ndarray] = []

    for tok_data in tokens:
        entry = tok_data["layers"][layer_idx]
        top_preds = entry.get(f"top{top_k}", [])
        if not top_preds:
            top1 = entry.get("top1", "")
            prob = entry.get("prob", 0.0)
            if top1:
                top_preds = [{"token": top1, "prob": prob if prob > 0 else 1.0}]
            else:
                top_preds = [{"token": "", "prob": 1.0}]

        K = len(top_preds)
        p = np.array([pr.get("prob", 0.0) for pr in top_preds], dtype=np.float64)
        s = p.sum()
        p = p / s if s > 0 else np.full(K, 1.0 / K)

        T = np.zeros((K, M), dtype=np.float64)
        for k, pr in enumerate(top_preds):
            vec = str_to_lang_vec.get(pr["token"])
            T[k] = vec if vec is not None else uniform

        p_lens.append(p)
        lang_mats.append(T)

    return p_lens, lang_mats


def restrict_to_task(lambdas: np.ndarray, lang_idx: list[int]) -> np.ndarray:
    """Renormalised restriction of lambda to the task languages (Def. 4)."""
    sub = lambdas[:, lang_idx]
    s = sub.sum(axis=1, keepdims=True)
    out = sub / np.where(s > 0, s, 1.0)
    return np.where(s > 0, out, 1.0 / len(lang_idx))


# --------------------------------------------------------------------------- #
#  Switch metrics (Definitions 5-13)
# --------------------------------------------------------------------------- #

def tv_distances(lambdas: np.ndarray) -> np.ndarray:
    """d_i = TV(lambda_{i+1}, lambda_i), shape (N-1,)."""
    if lambdas.shape[0] < 2:
        return np.zeros(0)
    return 0.5 * np.abs(np.diff(lambdas, axis=0)).sum(axis=1)


def commitments(lambdas: np.ndarray) -> np.ndarray:
    """kappa_i, shape (N,)."""
    M = lambdas.shape[1]
    if M <= 1:
        return np.zeros(lambdas.shape[0])
    return (lambdas.max(axis=1) - 1.0 / M) / (1.0 - 1.0 / M)


def switch_confidence(
    lambdas: np.ndarray, window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (d, kappa, rho, c) for one layer's language posteriors.

    rho follows Def. 8 of switch_and_layer_metrics.md: mean posterior mass
    on the post-boundary dominant language over the ``window`` positions
    after the switch position. Ties give rho = 0; an empty window (final
    boundary) falls back to the switch position's own dominant mass.
    """
    N = lambdas.shape[0]
    d = tv_distances(lambdas)
    kap = commitments(lambdas)
    rho = np.zeros(max(N - 1, 0))

    for i in range(N - 1):
        vec_next = lambdas[i + 1]
        mx = vec_next.max()
        if (vec_next == mx).sum() > 1:
            continue
        mu = int(np.argmax(vec_next))
        last = min(window, N - 2 - i)
        if last <= 0:
            rho[i] = mx
        else:
            rho[i] = lambdas[i + 2 : i + 2 + last, mu].mean()

    if N < 2:
        return d, kap, rho, np.zeros(0)
    c = d * np.minimum(kap[:-1], kap[1:]) * rho
    return d, kap, rho, c


def smoothed_switch_posterior(
    result: dict, lang_mats: list[np.ndarray],
) -> np.ndarray:
    """P(switch at boundary i | all evidence), shape (N-1,).

    Standard HMM backward pass, then the pairwise smoothed posterior xi_i
    projected onto language disagreement via the similarity matrices
    S_i = T_i @ T_{i+1}.T (Def. 11-12).
    """
    alphas = result["alphas"]
    emissions = result["emissions"]
    transitions = result["transitions"]
    N, K = emissions.shape
    if N < 2:
        return np.zeros(0)

    betas = np.zeros((N, K), dtype=np.float64)
    betas[-1] = 1.0
    for i in range(N - 2, -1, -1):
        betas[i] = transitions[i] @ (emissions[i + 1] * betas[i + 1])

    probs = np.zeros(N - 1)
    for i in range(N - 1):
        joint = (
            alphas[i][:, None]
            * transitions[i]
            * (emissions[i + 1] * betas[i + 1])[None, :]
        )
        denom = joint.sum()
        if denom <= 0:
            continue
        xi = joint / denom
        S = lang_mats[i] @ lang_mats[i + 1].T
        probs[i] = 1.0 - float((xi * S).sum())
    return probs


def boundary_lr(z_per_step: np.ndarray, K: int) -> np.ndarray:
    """G_i = log(K * z_{i+1}), shape (N-1,)."""
    z = np.asarray(z_per_step, dtype=np.float64)
    out = np.zeros_like(z)
    mask = z > 0
    out[mask] = np.log(K * z[mask])
    return out


# --------------------------------------------------------------------------- #
#  Layer specificity metrics (Definitions 16-19)
# --------------------------------------------------------------------------- #

def tracking_scores(
    lambdas_all: np.ndarray,
    g: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Soft tracking T_j with null calibration.

    With >= 2 distinct reference languages, uses the permutation null
    (Null B). With a constant reference (typical single-target
    translation), permutation degenerates and a bootstrap over positions
    against the uniform baseline 1/M is used instead (Null A).

    Returns (T, null_mean, null_std, Z, p, mode).
    """
    J, N, M = lambdas_all.shape
    T = np.zeros(J)
    null_mean = np.full(J, 1.0 / M)
    null_std = np.zeros(J)
    Z = np.zeros(J)
    pvals = np.ones(J)

    idx = np.where(g >= 0)[0]
    if len(idx) < 2:
        return T, null_mean, null_std, Z, pvals, "none (no defined reference positions)"

    cols = g[idx]
    n = len(idx)
    arange_n = np.arange(n)

    if len(set(cols.tolist())) >= 2:
        mode = f"permutation (P={n_perm})"
        perms = np.array([rng.permutation(n) for _ in range(n_perm)])
        for j in range(J):
            R = lambdas_all[j][idx]
            vals = R[arange_n, cols]
            T[j] = vals.mean()
            Tp = R[perms, cols].mean(axis=1)
            null_mean[j] = Tp.mean()
            if n_perm > 1:
                null_std[j] = Tp.std(ddof=1)
            if null_std[j] > 0:
                Z[j] = (T[j] - null_mean[j]) / null_std[j]
            pvals[j] = (1.0 + np.sum(Tp >= T[j])) / (n_perm + 1)
    else:
        mode = f"bootstrap vs uniform (P={n_perm}, constant reference)"
        boots = rng.integers(0, n, size=(n_perm, n))
        for j in range(J):
            R = lambdas_all[j][idx]
            vals = R[arange_n, cols]
            T[j] = vals.mean()
            Tb = vals[boots].mean(axis=1)
            if n_perm > 1:
                null_std[j] = Tb.std(ddof=1)
            if null_std[j] > 0:
                Z[j] = (T[j] - 1.0 / M) / null_std[j]
                pvals[j] = sps_norm.sf(Z[j])

    return T, null_mean, null_std, Z, pvals, mode


def classify_layers(
    C: np.ndarray, Z: np.ndarray, z_crit: float, c_min: float,
) -> list[str]:
    classes = []
    for c, z in zip(C, Z):
        if z >= z_crit and c >= c_min:
            classes.append("specific")
        elif c >= c_min:
            classes.append("committed-no-track")
        elif z >= z_crit:
            classes.append("weakly-tracking")
        else:
            classes.append("neutral")
    return classes


# --------------------------------------------------------------------------- #
#  Cross-layer consensus (Definitions 14-15)
# --------------------------------------------------------------------------- #

def consensus(
    conf: np.ndarray, layer_z: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Z-weighted consensus confidence per boundary (Definition 14)."""
    J = conf.shape[1]
    w = np.maximum(layer_z, 0.0)
    if w.sum() <= 0:
        w = np.ones(J)
        mode = "uniform (no positive Z)"
    else:
        mode = "Z-weighted"
    cons = (conf * w[None, :]).sum(axis=1) / w.sum()
    return cons, mode


def onset_depths(
    conf: np.ndarray, theta_c: float, onset_rate: float = 0.8,
) -> list:
    """Switch onset depth per boundary (Definition 15)."""
    n_bound, J = conf.shape
    det = conf >= theta_c
    onset: list = []
    for i in range(n_bound):
        j_star = None
        for j in range(J):
            if det[i, j:].mean() >= onset_rate:
                j_star = j
                break
        onset.append(j_star)
    return onset


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #

def plot_switch_heatmaps(
    layer_names: list[str],
    token_labels: list[str],
    conf: np.ndarray,
    post: np.ndarray,
    theta_c: float,
    output_path: str | None,
    dpi: int,
) -> None:
    n_bound, J = conf.shape
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(10, n_bound * 0.35), 9), sharex=True,
    )

    im1 = ax1.imshow(conf.T, aspect="auto", interpolation="nearest",
                     origin="lower", cmap="viridis")
    for i in range(n_bound):
        for j in range(J):
            if conf[i, j] >= theta_c:
                ax1.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False,
                                        edgecolor="white", linewidth=0.8))
    ax1.set_ylabel("Layer")
    ax1.set_yticks(range(J))
    ax1.set_yticklabels(layer_names, fontsize=8)
    ax1.set_title(f"Switch confidence c (white box: >= theta_c={theta_c:.3g})")
    fig.colorbar(im1, ax=ax1, label="c")

    im2 = ax2.imshow(post.T, aspect="auto", interpolation="nearest",
                     origin="lower", cmap="magma", vmin=0, vmax=1)
    ax2.set_ylabel("Layer")
    ax2.set_yticks(range(J))
    ax2.set_yticklabels(layer_names, fontsize=8)
    ax2.set_xticks(range(n_bound))
    ax2.set_xticklabels(token_labels[1:], rotation=45, ha="right", fontsize=9)
    ax2.set_xlabel("Boundary (landing token)")
    ax2.set_title("Smoothed switch posterior P(switch | all evidence)")
    fig.colorbar(im2, ax=ax2, label="P(switch)")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}", file=sys.stderr)
    else:
        plt.show()
    plt.close(fig)


def plot_consensus(
    token_labels: list[str],
    cons: np.ndarray,
    theta_c: float,
    onset: list,
    layer_names: list[str],
    output_path: str | None,
    dpi: int,
) -> None:
    n = len(cons)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), 4.5))
    colors = ["tab:red" if c >= theta_c else "lightgray" for c in cons]
    ax.bar(range(n), cons, color=colors)
    ax.axhline(theta_c, linestyle="--", color="black", linewidth=1)
    ax.text(n - 1, theta_c, f" theta_c={theta_c:.3g}", va="bottom",
            ha="right", fontsize=8)
    for i in range(n):
        if cons[i] >= theta_c and onset[i] is not None:
            ax.annotate(
                f"L{layer_names[onset[i]]}", (i, cons[i]),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7, rotation=90,
            )
    ax.set_xticks(range(n))
    ax.set_xticklabels(token_labels[1:], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Boundary (landing token)")
    ax.set_ylabel("Consensus confidence")
    ax.set_title("Cross-layer consensus switch confidence (annotated with onset depth)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}", file=sys.stderr)
    else:
        plt.show()
    plt.close(fig)


def plot_switch_by_layer(
    layer_names: list[str],
    conf: np.ndarray,
    post: np.ndarray,
    output_path: str | None,
    dpi: int,
) -> None:
    """Switching aggregated over boundaries (tokens), one point per layer."""
    J = conf.shape[1]
    x = np.arange(J)
    mean_c = conf.mean(axis=0)
    max_c = conf.max(axis=0)
    mean_post = post.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, J * 0.4), 7.5),
                                   sharex=True)

    ax1.plot(x, mean_c, marker="o", markersize=3, color="tab:blue",
             label="mean c")
    ax1.plot(x, max_c, marker="o", markersize=3, color="tab:blue",
             alpha=0.35, label="max c")
    ax1.set_ylabel("Switch confidence c")
    ax1.set_title("Switching aggregated over boundaries, per layer")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(x, mean_post, marker="o", markersize=3, color="tab:orange")
    ax2.set_ylabel("Mean P(switch | all evidence)")
    ax2.set_xlabel("Layer")
    ax2.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}", file=sys.stderr)
    else:
        plt.show()
    plt.close(fig)


def plot_layer_profile(
    layer_names: list[str],
    C: np.ndarray,
    T: np.ndarray,
    null_mean: np.ndarray,
    Z: np.ndarray,
    D: np.ndarray,
    classes: list[str],
    z_crit: float,
    c_min: float,
    output_path: str | None,
    dpi: int,
) -> None:
    J = len(C)
    x = np.arange(J)
    fig, axes = plt.subplots(4, 1, figsize=(max(8, J * 0.4), 11), sharex=True)

    axes[0].plot(x, C, marker="o", markersize=3, color="tab:blue")
    axes[0].axhline(c_min, linestyle="--", color="gray", linewidth=1)
    axes[0].text(J - 1, c_min, f" C_min={c_min:g}", va="bottom",
                 ha="right", fontsize=8)
    axes[0].set_ylabel("Commitment C_j")
    axes[0].set_title("Layer language-specificity profile")

    axes[1].plot(x, T, marker="o", markersize=3, color="tab:cyan",
                 label="T_j")
    axes[1].plot(x, null_mean, linestyle="--", color="gray", linewidth=1,
                 label="null mean")
    axes[1].set_ylabel("Tracking T_j")
    axes[1].legend(fontsize=8, loc="best")

    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].axhline(z_crit, linestyle="--", color="tab:red", linewidth=1)
    axes[2].text(J - 1, z_crit, f" z={z_crit:g}", va="bottom", ha="right",
                 fontsize=8)
    axes[2].plot(x, Z, marker="o", markersize=3, color="lightgray",
                 zorder=2)
    for cls, colr in CLASS_COLORS.items():
        m = [i for i in range(J) if classes[i] == cls]
        if m:
            axes[2].scatter(m, Z[m], color=colr, label=cls, zorder=3, s=20)
    axes[2].set_ylabel("Tracking Z_j")
    axes[2].legend(fontsize=7, loc="best")

    axes[3].axhline(0, color="black", linewidth=0.5)
    axes[3].plot(x, D, marker="o", markersize=3, color="tab:purple")
    axes[3].set_ylabel("Coherence D_j")
    axes[3].set_xlabel("Layer")

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)

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
        description="Language switch detection and layer specificity from the trellis.",
    )
    parser.add_argument("input", type=str, help="Path to lens output JSON file.")
    parser.add_argument("--lang-dist", type=str, required=True,
                        help="Path to token_lang_dist_fineweb_dataset_<model>.json.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--source-lang", type=str, default=None)
    parser.add_argument("--target-lang", type=str, default=None)
    parser.add_argument("--middle-lang", type=str, default="eng_Latn")
    parser.add_argument("--window", type=int, default=2,
                        help="Persistence window w (default: 2).")
    parser.add_argument("--task-restricted", action="store_true",
                        help="Compute switch metrics on the renormalised "
                             "restriction to {source, middle, target}.")
    parser.add_argument("--theta-c", type=float, default=None,
                        help="Detection threshold on c (default: percentile).")
    parser.add_argument("--theta-pct", type=float, default=95.0,
                        help="Percentile for the default theta_c (default: 95).")
    parser.add_argument("--z-crit", type=float, default=2.0)
    parser.add_argument("--c-min", type=float, default=0.3)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Output prefix. Four files: <prefix>_switch.png, "
                             "<prefix>_switch_line.png, "
                             "<prefix>_switch_by_layer.png, "
                             "<prefix>_layers.png.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--entropy-weight", action="store_true")
    parser.add_argument("--transition-mode", type=str, default="row_norm",
                        choices=["row_norm", "softmax"])
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()

    lens_data = load_lens(Path(args.input))
    if "tokens" not in lens_data or not lens_data["tokens"]:
        print("Error: no tokens in lens JSON", file=sys.stderr)
        sys.exit(1)

    tokens = lens_data["tokens"]
    N = len(tokens)
    if N < 2:
        print("Error: need at least 2 tokens for switch analysis", file=sys.stderr)
        sys.exit(1)

    source_lang = args.source_lang or lens_data.get("source_lang", "")
    target_lang = args.target_lang or lens_data.get("target_lang", "")
    if not source_lang or not target_lang:
        print("Error: source/target language missing", file=sys.stderr)
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
    lang_to_idx = {lc: i for i, lc in enumerate(all_langs)}
    M = len(all_langs)

    task_langs = [source_lang, args.middle_lang, target_lang]
    task_idx = []
    for lc in task_langs:
        if lc in lang_to_idx:
            task_idx.append(lang_to_idx[lc])
        else:
            print(f"  WARNING: language '{lc}' not in lang dist.", file=sys.stderr)
    task_idx = list(dict.fromkeys(task_idx))
    if args.task_restricted and len(task_idx) < 2:
        print("  WARNING: fewer than 2 distinct task languages available; "
              "task restriction degenerates, falling back to full-M.", file=sys.stderr)
    use_restriction = args.task_restricted and len(task_idx) >= 2

    token_labels = []
    for tok in tokens:
        s = tok.get("token", "")
        if is_bpe:
            s = s.replace(" ", "Ġ")
        token_labels.append(_truncate(s))

    layer_names = [str(l["layer"]) for l in tokens[0]["layers"]]
    J = len(layer_names)
    rng = np.random.default_rng(args.seed)

    g = ground_truth_languages(lens_data, str_to_lang_vec, all_langs)
    g_idx = np.where(g >= 0)[0]
    cnt = Counter(all_langs[gi] for gi in g if gi >= 0)
    print(f"\nGround truth: {len(g_idx)}/{N} positions defined; "
          + ", ".join(f"{k}={v}" for k, v in cnt.most_common()), file=sys.stderr)

    mode_label = "task-restricted" if use_restriction else "full-M"
    print(f"\n--- Computing switch/layer metrics [{mode_label}] "
          f"(window={args.window}, transition={args.transition_mode}, "
          f"entropy_weight={args.entropy_weight}) ---", file=sys.stderr)

    lambdas_all = np.zeros((J, N, M), dtype=np.float64)
    conf = np.zeros((N - 1, J), dtype=np.float64)
    post_all = np.zeros((N - 1, J), dtype=np.float64)
    G_all = np.zeros((N - 1, J), dtype=np.float64)
    D_all = np.zeros(J, dtype=np.float64)

    for j in range(J):
        p_lens, lang_mats = build_layer_inputs(
            lens_data, str_to_lang_vec, all_langs, args.top_k, j,
        )
        result = trellis_layer_analysis(
            p_lens, lang_mats,
            use_entropy_weight=args.entropy_weight,
            transition_mode=args.transition_mode,
            temperature=args.temperature,
        )
        lambdas_all[j] = result["lambdas"]

        work = (restrict_to_task(result["lambdas"], task_idx)
                if use_restriction else result["lambdas"])
        _, _, _, c = switch_confidence(work, args.window)
        conf[:, j] = c
        post_all[:, j] = smoothed_switch_posterior(result, lang_mats)

        K = p_lens[0].shape[0]
        G_all[:, j] = boundary_lr(result["z_per_step"], K)
        D_all[j] = 2.0 * (result["log_likelihood"] + (N - 1) * np.log(K))

    C_all = commitments(lambdas_all.reshape(-1, M)).reshape(J, N).mean(axis=1)

    T, null_mean, null_std, Z, pvals, null_mode = tracking_scores(
        lambdas_all, g, args.n_perm, rng,
    )
    classes = classify_layers(C_all, Z, args.z_crit, args.c_min)

    cons, wmode = consensus(conf, Z)

    if args.theta_c is not None:
        theta_c = args.theta_c
        theta_src = "given"
    else:
        theta_c = float(np.percentile(cons, args.theta_pct))
        theta_src = f"p{args.theta_pct:g} of consensus"

    onset = onset_depths(conf, theta_c)

    print(f"\n=== Switch detection (theta_c={theta_c:.4g}, source: {theta_src}) ===",
          file=sys.stderr)
    print(f"Layer weights: {wmode}; consensus: max={cons.max():.4f}, "
          f"mean={cons.mean():.4f}", file=sys.stderr)
    det_rows = [i for i in range(N - 1) if cons[i] >= theta_c]
    if not det_rows:
        print("No boundaries reached the consensus threshold.", file=sys.stderr)
    for i in det_rows:
        nd = int((conf[i] >= theta_c).sum())
        j_star = onset[i]
        on = layer_names[j_star] if j_star is not None else "-"
        print(f"  boundary {i:3d} (token '{token_labels[i + 1]}'): "
              f"consensus={cons[i]:.3f}, layers={nd}/{J}, "
              f"onset=L{on}, meanG={G_all[i].mean():.3f}", file=sys.stderr)

    mean_c_by_layer = conf.mean(axis=0)
    order = np.argsort(mean_c_by_layer)[::-1]
    print(f"\nSwitching by layer (aggregated over boundaries):", file=sys.stderr)
    print("  highest mean c: "
          + ", ".join(f"{layer_names[j]} ({mean_c_by_layer[j]:.4f})"
                      for j in order[:3]), file=sys.stderr)
    print("  lowest  mean c: "
          + ", ".join(f"{layer_names[j]} ({mean_c_by_layer[j]:.4f})"
                      for j in order[-3:]), file=sys.stderr)

    print(f"\nStrongest per-layer evidence (top 5):", file=sys.stderr)
    flat = np.argsort(conf, axis=None)[::-1][:5]
    for f in flat:
        i, j = np.unravel_index(f, conf.shape)
        g_a = all_langs[g[i]] if g[i] >= 0 else "-"
        g_b = all_langs[g[i + 1]] if g[i + 1] >= 0 else "-"
        print(f"  boundary {i:3d} layer {layer_names[j]:>6}: c={conf[i, j]:.4f} "
              f"(ref: {g_a} -> {g_b})", file=sys.stderr)

    print(f"\n=== Layer specificity (null: {null_mode}) ===", file=sys.stderr)
    print(f"  {'layer':>6} {'C':>7} {'T':>7} {'null':>7} {'Z':>7} {'p':>9} "
          f"{'D':>8}  class", file=sys.stderr)
    for j in range(J):
        print(f"  {layer_names[j]:>6} {C_all[j]:7.3f} {T[j]:7.3f} "
              f"{null_mean[j]:7.3f} {Z[j]:7.2f} {pvals[j]:9.4f} "
              f"{D_all[j]:8.2f}  {classes[j]}", file=sys.stderr)

    if args.output:
        plot_switch_heatmaps(
            layer_names, token_labels, conf, post_all, theta_c,
            f"{args.output}_switch.png", args.dpi,
        )
        plot_consensus(
            token_labels, cons, theta_c, onset, layer_names,
            f"{args.output}_switch_line.png", args.dpi,
        )
        plot_switch_by_layer(
            layer_names, conf, post_all,
            f"{args.output}_switch_by_layer.png", args.dpi,
        )
        plot_layer_profile(
            layer_names, C_all, T, null_mean, Z, D_all, classes,
            args.z_crit, args.c_min,
            f"{args.output}_layers.png", args.dpi,
        )
    else:
        plot_switch_heatmaps(layer_names, token_labels, conf, post_all,
                             theta_c, None, args.dpi)
        plot_consensus(token_labels, cons, theta_c, onset, layer_names,
                       None, args.dpi)
        plot_switch_by_layer(layer_names, conf, post_all, None, args.dpi)
        plot_layer_profile(layer_names, C_all, T, null_mean, Z, D_all,
                           classes, args.z_crit, args.c_min, None, args.dpi)


if __name__ == "__main__":
    main()
