"""Analyze lens.txt: classify each layer as language-specific or language-neutral.

Reads the per-token logit lens grid from lens.txt and, for each layer row,
classifies every predicted token into a language category. A layer is
"language-specific" if its predictions are concentrated in real-language
scripts (Chinese, French/Latin, etc.); "language-neutral" if dominated by
special tokens, markup, punctuation, or generic cross-lingual connectors.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

SPECIAL_RE = re.compile(r"^<.*>$|^\[multimod")
HTML_RE = re.compile(r"^</?[a-z0-9]+$")
PUNCT_RE = re.compile(r'^[\s+":*\-_/().,!?;|#&=+\[\]{}<>]+$|^\\n$|^\\t$|^\\r$')


def classify_token(tok: str) -> str:
    """Classify a single predicted token into a language category."""
    t = tok.strip()
    if not t:
        return "empty"
    if SPECIAL_RE.match(t) or "[multimod" in t:
        return "special"
    if HTML_RE.match(t):
        return "markup"
    if PUNCT_RE.match(t):
        return "punct"
    if re.match(r"^[\u4e00-\u9fff\u3400-\u4dbf]", t):
        return "zh"
    if re.match(r"^[\u3040-\u309f\u30a0-\u30ff]", t):
        return "ja"
    if re.match(r"^[\uac00-\ud7af]", t):
        return "ko"
    if re.match(r"^[\u0900-\u097f]", t):
        return "hi"
    if re.match(r"^[\u0e00-\u0e7f]", t):
        return "th"
    if re.match(r"^[\u0600-\u06ff]", t):
        return "ar"
    if re.match(r"^[\u0400-\u04ff]", t):
        return "ru"
    if re.match(r"^[\u0370-\u03ff]", t):
        return "el"
    if re.match(r"^[\u0980-\u09ff]", t):
        return "bn"
    if re.match(r"^[\u0e80-\u0eff]", t):
        return "lo"
    if re.match(r"^[\u1000-\u109f]", t):
        return "my"
    if re.match(r"^[\u0590-\u05ff]", t):
        return "he"
    if t[0].isascii() and t[0].isalpha():
        return "latin"
    return "other"


# Categories that are "language-neutral" (not tied to a specific language)
NEUTRAL_CATS = {"special", "markup", "punct", "empty", "other"}

# Latin script is ambiguous — English/French share it. We count it
# separately and decide based on context.


def analyze(lens_path: str) -> None:
    raw_lines = Path(lens_path).read_text().splitlines()

    # Find the header row to determine column geometry
    header_idx = None
    for i, line in enumerate(raw_lines):
        if "pos 0" in line:
            header_idx = i
            break
    if header_idx is None:
        print("Could not find header row", file=sys.stderr)
        sys.exit(1)

    header = raw_lines[header_idx]
    col0 = header.find("pos 0")
    col1 = header.find("pos 1")
    col_width = col1 - col0  # 12
    n_cols = (len(header) - col0) // col_width

    # Merge continuation lines: lines that don't start with a known label
    # are continuations of the previous row.
    known_labels = {"input", "embed", "layer", "final"}
    merged_lines: list[str] = []
    for line in raw_lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("---"):
            continue
        first = stripped.split()[0] if stripped.split() else ""
        if first in known_labels:
            merged_lines.append(line)
        else:
            # Continuation — append to previous line, padded to same width
            if merged_lines:
                merged_lines[-1] = merged_lines[-1] + line

    # Parse each merged row using fixed-width column slicing
    rows: list[tuple[str, list[str]]] = []
    for line in merged_lines:
        label_part = line[:col0].strip()
        if label_part.startswith("layer"):
            parts = label_part.split()
            if len(parts) >= 2:
                label = f"layer {parts[1]}"
            else:
                continue
        elif label_part in ("input", "embed", "final"):
            label = label_part
        else:
            continue

        tokens: list[str] = []
        for c in range(n_cols):
            start = col0 + c * col_width
            end = start + col_width
            if end <= len(line):
                tok = line[start:end].strip()
            elif start < len(line):
                tok = line[start:].strip()
            else:
                tok = ""
            tokens.append(tok)
        rows.append((label, tokens))

    if not rows:
        print("No data rows found", file=sys.stderr)
        sys.exit(1)

    # The last position is the key "prediction" position
    last_idx = n_cols - 1

    print(f"{'Layer':<12} {'Classification':<20} {'Top cats (all pos)':<45} {'Last-pos token':<15} {'Last-pos cat'}")
    print("-" * 110)

    for label, tokens in rows:
        cats = Counter(classify_token(t) for t in tokens)

        # Classify the last position
        last_tok = tokens[last_idx] if last_idx < len(tokens) else "?"
        last_cat = classify_token(last_tok)

        # Determine overall classification
        # Count language-specific vs neutral across all positions
        lang_specific = sum(v for k, v in cats.items() if k not in NEUTRAL_CATS and k != "latin")
        neutral = sum(v for k, v in cats.items() if k in NEUTRAL_CATS)
        latin = cats.get("latin", 0)
        total = sum(cats.values())

        # Decide: if >40% special/markup/punct → neutral
        # If >30% language-specific scripts → language-specific
        # Otherwise transitional
        neutral_frac = neutral / total if total else 0
        lang_frac = lang_specific / total if total else 0

        if neutral_frac > 0.55:
            classification = "language-neutral"
        elif lang_frac > 0.30:
            classification = "language-specific"
        elif lang_frac > 0.15:
            classification = "transitional"
        else:
            classification = "mixed"

        # Build a compact category summary
        cat_summary = ", ".join(f"{k}:{v}" for k, v in cats.most_common(4))

        print(f"{label:<12} {classification:<20} {cat_summary:<45} {last_tok:<15} {last_cat}")

    # Summary
    print()
    print("=" * 110)
    print("Last-position progression (pos {} = prediction after the final '\"'):".format(last_idx))
    print()
    print(f"{'Layer':<12} {'Token':<16} {'Category'}")
    print("-" * 40)
    for label, tokens in rows:
        if label in ("input",):
            continue
        last_tok = tokens[last_idx] if last_idx < len(tokens) else "?"
        last_cat = classify_token(last_tok)
        marker = " <-- correct answer" if last_tok == "花" else ""
        print(f"{label:<12} {last_tok:<16} {last_cat}{marker}")

    print()
    print("=" * 110)
    print("Summary by phase:")
    print()

    phases: list[tuple[str, list[str]]] = []
    current_phase: str | None = None
    current_layers: list[str] = []
    for label, tokens in rows:
        if label in ("input", "embed"):
            continue
        cats = Counter(classify_token(t) for t in tokens)
        total = sum(cats.values())
        neutral_frac = sum(v for k, v in cats.items() if k in NEUTRAL_CATS) / total if total else 0
        lang_frac = sum(v for k, v in cats.items() if k not in NEUTRAL_CATS and k != "latin") / total if total else 0

        if neutral_frac > 0.55:
            phase = "language-neutral"
        elif lang_frac > 0.30:
            phase = "language-specific"
        elif lang_frac > 0.15:
            phase = "transitional"
        else:
            phase = "mixed"

        if phase != current_phase:
            if current_phase and current_layers:
                phases.append((current_phase, current_layers))
            current_phase = phase
            current_layers = [label]
        else:
            current_layers.append(label)
    if current_phase and current_layers:
        phases.append((current_phase, current_layers))

    for phase, layers in phases:
        if len(layers) == 1:
            print(f"  {phase:<20} {layers[0]}")
        else:
            print(f"  {phase:<20} {layers[0]} – {layers[-1]}  ({len(layers)} layers)")


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "lens.txt")
