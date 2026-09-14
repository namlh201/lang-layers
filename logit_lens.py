"""Logit lens hooks for MLX models, in the TransformerLens style.

TransformerLens (PyTorch) cannot run MLX-quantized checkpoints such as
``mlx-community/gemma-4-12B-it-4bit``. This module ports TransformerLens's
logit-lens technique to models loaded with :mod:`mlx_vlm` (or :mod:`mlx_lm`):

* TransformerLens-style *hook points* on the residual stream
  (``blocks.{i}.hook_resid_pre`` / ``blocks.{i}.hook_resid_post``).
* ``run_with_cache`` which runs a forward pass and caches every residual.
* The logit lens itself: apply the model's final layer-norm (with recomputed
  statistics) followed by the unembed (and any logit soft-cap) to each cached
  residual, yielding "what the model would predict" at every layer.

The logit lens mirrors ``ActivationCache.apply_ln_to_stack(..., recompute_ln=True,
layer=n_layers)`` from TransformerLens: each component is normalized through the
final norm and projected through the unembed.
"""

from __future__ import annotations
from sympy.polys.subresultants_qq_zz import res

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, final

try:
    import mlx.core as mx
    import mlx.nn as nn

    NdArray = mx.array
except ImportError:
    import numpy as np
    import torch.nn as nn

    NdArray = np.ndarray

import nnsight

__all__ = [
    "HookedMLXModel",
    "HookedTorchModel",
    "HookedNNsightModel",
    "make_hooked_model",
    "ActivationCache",
    "HookPoint",
    "LayerLens",
    "LogitLensResult",
    "LayerTokenLens",
    "LogitLensGrid",
    "format_logit_lens",
    "format_logit_lens_grid",
    "grid_to_lens_result",
]


HookFunction = Callable[[NdArray, "HookPoint"], NdArray | None]

_ORIGINAL_CALLS: dict[type, Any] = {}


class HookPoint:
    """Context object passed to hook functions, mirroring TransformerLens.

    A hook is ``fn(activation, hook) -> activation | None``. Returning ``None``
    leaves the activation unchanged; returning an array replaces it (and the
    replacement flows forward through the rest of the forward pass).
    """

    __slots__ = ("name", "layer")

    def __init__(self, name: str, layer: int):
        self.name = name
        self.layer = layer

    def __repr__(self) -> str:
        return f"HookPoint(name={self.name!r}, layer={self.layer})"


class _LayerHookCtx:
    """Per-layer runtime state attached to a decoder layer during a cached run.

    Stored as a plain instance attribute (it is *not* an ``mx.array``/``dict``/
    ``list``/``tuple``) so it never pollutes the ``mlx.nn.Module`` parameter
    tree.
    """

    __slots__ = ("layer", "cache", "hooks")

    def __init__(self, layer: int, cache: ActivationCache, hooks: dict):
        self.layer = layer
        self.cache = cache
        self.hooks = hooks

    def apply(self, name: str, value: NdArray) -> NdArray:
        for fn in self.hooks.get(name, ()):  # type: ignore[union-attr]
            result = fn(value, HookPoint(name, self.layer))
            if result is not None:
                value = result
        self.cache[name] = value
        return value


def _hooked_layer_call(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
    """Dispatcher installed on each decoder-layer class.

    When no hook context is attached it is a near-no-op passthrough to the
    original ``__call__``. When a context is present (inside
    ``run_with_cache``) it records/transforms the residual stream entering
    (``hook_resid_pre``) and leaving (``hook_resid_post``) the layer.

    The residual stream is assumed to be the first positional argument and the
    first element of the returned tuple (the convention used by every
    ``mlx_lm`` decoder layer).
    """
    ctx: _LayerHookCtx | None = getattr(self, "_tl_hook_ctx", None)
    original = _ORIGINAL_CALLS.get(type(self))
    if original is None:
        original = type(self).__call__
    if ctx is None:
        return original(self, *args, **kwargs)

    layer = ctx.layer
    residual = args[0] if args else kwargs.get("x")
    residual = ctx.apply(f"blocks.{layer}.hook_resid_pre", residual)
    if args:
        new_args = (residual,) + args[1:]
        out = original(self, *new_args, **kwargs)
    else:
        kwargs = dict(kwargs)
        kwargs["x"] = residual
        out = original(self, **kwargs)

    if isinstance(out, tuple):
        post = out[0]
        post = ctx.apply(f"blocks.{layer}.hook_resid_post", post)
        return (post,) + out[1:]
    post = ctx.apply(f"blocks.{layer}.hook_resid_post", out)
    return post


class ActivationCache(dict):
    """A dict of cached activations keyed by TransformerLens hook names.

    Holds a back-reference to the :class:`HookedMLXModel` so lens helpers can be
    expressed directly on the cache, mirroring TransformerLens's
    ``ActivationCache``.
    """

    def __init__(
        self, *args: Any, model: Any = None, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.model = model

    def apply_final_ln(self, residual: NdArray) -> NdArray:
        """Apply the model's final norm to a residual-stream component."""
        return self.model._final_norm(residual)  # type: ignore[union-attr]

    def lens_logits(self, residual: NdArray) -> NdArray:
        """Full vocab logits for a residual via the logit lens."""
        return self.model.lens_logits(residual)  # type: ignore[union-attr]

    def logit_lens(
        self,
        pos: int = -1,
        top_k: int = 10,
        final_logits: NdArray | None = None,
    ) -> LogitLensResult:
        """Run the logit lens over the cached residuals.

        Args:
            pos: Sequence position to inspect (defaults to the last token).
            top_k: Number of top tokens to materialize per layer.
            final_logits: Optional real model logits to include as the
                "final" row and to rank against. If ``None`` the lens of the
                last layer is used as the reference.
        """
        if self.model is None:
            raise RuntimeError("ActivationCache has no associated HookedMLXModel.")
        return self.model.logit_lens(
            self, pos=pos, top_k=top_k, final_logits=final_logits
        )

    def logit_lens_per_token(
        self,
        input_ids: NdArray,
        final_logits: NdArray | None = None,
        top_k: int = 1,
    ) -> LogitLensGrid:
        """Per-token logit lens: top-k predictions at every position for every layer.

        Args:
            input_ids: The token ids fed to the model, shape ``[batch, seq]``.
            final_logits: Optional real model logits to include as the "final" row.
            top_k: Number of top tokens to keep per position (default 1).
        """
        if self.model is None:
            raise RuntimeError("ActivationCache has no associated HookedMLXModel.")
        return self.model.logit_lens_per_token(
            self, input_ids, final_logits=final_logits, top_k=top_k
        )

    def logit_lens_batch(
        self,
        input_ids: NdArray,
        attention_mask: NdArray,
        top_k: int = 10,
        final_logits: NdArray | None = None,
    ) -> list[LogitLensGrid]:
        """Per-token logit lens for a batch, returning one grid per sample.

        Padded positions (where ``attention_mask == 0``) are excluded
        from the results.  Each returned :class:`LogitLensGrid` is
        identical to what :meth:`logit_lens_per_token` would produce
        for that sample individually.
        """
        if self.model is None:
            raise RuntimeError("ActivationCache has no associated HookedMLXModel.")
        return self.model.logit_lens_batch(
            self, input_ids, attention_mask,
            top_k=top_k, final_logits=final_logits,
        )


class HookedMLXModel:
    """An MLX model instrumented with TransformerLens-style hooks.

    Works with both ``mlx_lm`` (text-only) and ``mlx_vlm`` (multimodal) models.
    For VLM models the language model submodule is used automatically.

    Example::

        from mlx_vlm import load
        from logit_lens import HookedMLXModel

        model, processor = load("mlx-community/gemma-4-12B-it-4bit")
        hooked = HookedMLXModel(model, processor)

        logits, cache = hooked.run_with_cache("The capital of France is")
        result = cache.logit_lens(pos=-1, top_k=5)
        print(format_logit_lens(result, processor))
    """

    def __init__(self, model: nn.Module, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self._fwd_hooks: dict[str, list[HookFunction]] = {}
        self._patched_classes: set[type] = set()
        self._is_vlm = hasattr(model, "language_model")
        self._install_hook_dispatcher()
        self._final_norm, self._unembed, self._softcap = self._detect_lens_components()

    # ------------------------------------------------------------------ setup

    def _install_hook_dispatcher(self) -> None:
        layers = self.layers
        if len(layers) == 0:
            return
        for layer in layers:
            cls = type(layer)
            self._patched_classes.add(cls)
            if cls in _ORIGINAL_CALLS:
                continue  # already instrumented by another HookedMLXModel
            try:
                original = cls.__call__
            except AttributeError:
                continue
            if original is None:
                continue
            _ORIGINAL_CALLS[cls] = original
            cls.__call__ = _hooked_layer_call  # type: ignore[assignment]

    def _detect_lens_components(self):
        if self._is_vlm:
            lm = self.model.language_model
            text_model = lm.model
            final_norm = text_model.norm
            embed = text_model.embed_tokens
            unembed = embed.as_linear
            softcap = getattr(lm, "final_logit_softcapping", None)
        else:
            inner = getattr(self.model, "model", self.model)
            final_norm = (
                getattr(inner, "norm", None)
                or getattr(inner, "final_norm", None)
                or getattr(self.model, "norm", None)
            )
            if final_norm is None:
                raise ValueError(
                    "Could not locate the final norm (expected model.model.norm)."
                )

            tied = bool(getattr(self.model, "tie_word_embeddings", False))
            embed = getattr(inner, "embed_tokens", None)
            if (tied or not hasattr(self.model, "lm_head")) and embed is not None:
                unembed = embed.as_linear
            else:
                unembed = self.model.lm_head  # type: ignore[union-attr]

            softcap = getattr(self.model, "final_logit_softcapping", None)

        if softcap is not None:
            try:
                softcap = float(softcap)
            except (TypeError, ValueError):
                softcap = None
        return final_norm, unembed, softcap

    # --------------------------------------------------------------- properties

    @property
    def layers(self) -> Sequence[nn.Module]:
        return self.model.layers  # type: ignore[union-attr]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def cfg(self) -> Any:
        return getattr(self.model, "args", getattr(self.model, "config", None))

    # ------------------------------------------------------------------- hooks

    def add_hook(self, name: str, hook: HookFunction) -> None:
        """Register a forward hook at a named hook point.

        ``name`` is an exact hook-point name such as
        ``"blocks.3.hook_resid_post"``. The hook ``fn(activation, hook)`` may
        return a replacement activation (which flows forward) or ``None``.
        """
        self._fwd_hooks.setdefault(name, []).append(hook)

    def add_per_layer_hook(self, template: str, hook: HookFunction) -> None:
        """Register ``hook`` on every layer using a ``{layer}`` template.

        e.g. ``template="blocks.{layer}.hook_resid_post"``.
        """
        for i in range(self.n_layers):
            self.add_hook(template.format(layer=i), hook)

    def remove_hooks(self) -> None:
        self._fwd_hooks.clear()

    def restore(self) -> None:
        """Undo the class-level ``__call__`` patches (fully reversible).

        Reverts every decoder-layer class instrumented by this instance back
        to its original ``__call__``.
        """
        for cls in self._patched_classes:
            original = _ORIGINAL_CALLS.pop(cls, None)
            if original is not None:
                cls.__call__ = original  # type: ignore[assignment]
        self._patched_classes.clear()

    # ------------------------------------------------------------- forward/cache

    def _apply_chat_template(self, text: str) -> tuple[str, bool]:
        """Apply the VLM chat template if one is available.

        Returns ``(formatted_text, add_special_tokens)`` matching what
        ``mlx_vlm.stream_generate`` does: for Gemma models with a chat
        template, the template includes BOS so ``add_special_tokens`` is
        ``False``; otherwise the raw text is returned with ``True``.
        """
        if not self._is_vlm or self.tokenizer is None:
            return text, True
        processor = self.tokenizer
        if not hasattr(processor, "chat_template") or processor.chat_template is None:
            return text, True
        try:
            from mlx_vlm.prompt_utils import apply_chat_template

            config = getattr(self.model, "config", None)
            if config is None:
                return text, True
            model_type = getattr(config, "model_type", "")
            fmt = apply_chat_template(processor, config, text, num_images=0)
            # Gemma models: chat template already includes BOS
            add_special = model_type not in (
                "gemma3", "gemma3n", "gemma4", "gemma4_unified",
            )
            return fmt, add_special
        except Exception:
            return text, True

    def _encode(self, text: str, prepend_bos: bool | None = None) -> NdArray:
        tok = self._tokenizer
        if tok is None:
            raise ValueError("A tokenizer is required to encode text.")
        if prepend_bos is None:
            text, add_special = self._apply_chat_template(text)
            ids = tok.encode(text, add_special_tokens=add_special)
        else:
            ids = tok.encode(text, add_special_tokens=prepend_bos)
        if isinstance(ids, mx.array):
            return ids[None] if ids.ndim == 1 else ids
        if not isinstance(ids, list):
            ids = list(ids)
        return mx.array([ids]) if isinstance(ids[0], int) else mx.array(ids)

    @property
    def _tokenizer(self) -> Any:
        """The underlying tokenizer, unwrapped from a VLM processor if needed."""
        tok = self.tokenizer
        if tok is not None and hasattr(tok, "tokenizer"):
            return tok.tokenizer
        return tok

    def run_with_cache(
        self,
        input: str | NdArray,
        names: Iterable[str] | None = None,
        prepend_bos: bool | None = None,
    ) -> tuple[NdArray, ActivationCache]:
        """Run a forward pass and cache the residual stream at every layer.

        Args:
            input: Prompt text or a token-id array.
            names: Optional iterable of hook-point names to restrict caching to
                (defaults to all ``hook_resid_pre``/``hook_resid_post`` points).
            prepend_bos: Override for tokenizer special-token handling.

        Returns:
            ``(logits, cache)`` where ``logits`` are the model's real output
            logits and ``cache`` is an :class:`ActivationCache`.
        """
        ids = self._encode(input, prepend_bos) if isinstance(input, str) else input
        if ids.ndim == 1:
            ids = ids[None]

        name_filter = set(names) if names is not None else None
        cache = ActivationCache(model=self)

        for i, layer in enumerate(self.layers):
            layer._tl_hook_ctx = _LayerHookCtx(i, cache, self._fwd_hooks)

        try:
            out = self.model(ids)
            mx.eval(out)
            logits = out.logits if hasattr(out, "logits") else out
            for key in list(cache.keys()):
                if name_filter is not None and key not in name_filter:
                    del cache[key]
        finally:
            for layer in self.layers:
                layer._tl_hook_ctx = None

        return logits, cache

    # ------------------------------------------------------------------- lens

    def lens_logits(self, residual: NdArray) -> NdArray:
        """Apply the logit lens (final norm -> unembed -> soft-cap) to a residual.

        Equivalent to running the model's own head on an intermediate residual
        stream component, i.e. ``softcap(unembed(ln_final(residual)))``.
        """
        h = self._final_norm(residual)
        logits = self._unembed(h)
        if self._softcap is not None:
            logits = mx.tanh(logits / self._softcap) * self._softcap
        return logits

    def logit_lens(
        self,
        cache: ActivationCache,
        pos: int = -1,
        top_k: int = 10,
        final_logits: NdArray | None = None,
    ) -> LogitLensResult:
        """Compute logit-lens predictions for every layer from a cache.

        Produces one row per residual-stream checkpoint:
        the embedding (``blocks.0.hook_resid_pre``) then the output of every
        layer (``blocks.{i}.hook_resid_post``). Each row reports the ``top_k``
        tokens the lens predicts at ``pos``.
        """
        rows: list[LayerLens] = []
        points: list[tuple[str, str]] = [("embed", "blocks.0.hook_resid_pre")]
        for i in range(self.n_layers):
            points.append((f"layer {i}", f"blocks.{i}.hook_resid_post"))

        for label, name in points:
            if name not in cache:
                continue
            logits = self.lens_logits(cache[name])
            rows.append(LayerLens(label=label, name=name, logits=logits))

        final_tokens: list[tuple[int, str, float, float]] = []
        final_rank: int | None = None
        ref_logits = final_logits if final_logits is not None else (
            rows[-1].logits if rows else None
        )
        if ref_logits is not None:
            ref_pos = _take_pos(ref_logits, pos)
            final_tokens, _ = _top_k_tokens(ref_pos, top_k, self._tokenizer)
            if rows:
                final_rank = _rank_of_token(
                    _take_pos(rows[-1].logits, pos),
                    final_tokens[0][0] if final_tokens else 0,
                )

        for row in rows:
            lp = _take_pos(row.logits, pos)
            row.top_tokens, row.top_indices = _top_k_tokens(lp, top_k, self._tokenizer)
            if final_tokens:
                row.rank_of_final = _rank_of_token(lp, final_tokens[0][0])

        return LogitLensResult(
            rows=rows,
            final_tokens=final_tokens,
            final_rank=final_rank,
            pos=pos,
            top_k=top_k,
        )

    def logit_lens_per_token(
        self,
        cache: ActivationCache,
        input_ids: NdArray,
        final_logits: NdArray | None = None,
        top_k: int = 1,
    ) -> LogitLensGrid:
        """Compute per-token logit-lens predictions for every layer.

        For each residual-stream checkpoint and every sequence position,
        materializes the ``top_k`` predicted tokens. The result is a grid:
        layers × positions.

        Args:
            cache: An :class:`ActivationCache` from ``run_with_cache``.
            input_ids: The token ids fed to the model ``[batch, seq]``. Used to
                show input tokens and determine "correct" predictions.
            final_logits: Optional real model logits as the "final" row.
            top_k: Top tokens to keep per position (default 1).
        """
        layers: list[LayerTokenLens] = []
        points: list[tuple[str, str]] = [("embed", "blocks.0.hook_resid_pre")]
        for i in range(self.n_layers):
            points.append((f"layer {i}", f"blocks.{i}.hook_resid_post"))

        for label, name in points:
            if name not in cache:
                continue
            logits = self.lens_logits(cache[name])  # [batch, seq, vocab]
            layers.append(
                LayerTokenLens(label=label, name=name, logits=logits)
            )

        if final_logits is not None:
            layers.append(
                LayerTokenLens(label="final", name="final", logits=final_logits)
            )

        seq_len = input_ids.shape[-1]
        input_ids_1d = input_ids[0] if input_ids.ndim == 2 else input_ids

        for layer_row in layers:
            lg = layer_row.logits
            if lg.ndim == 3:
                lg = lg[0]  # squeeze batch
            layer_row.top_ids = _top_k_ids_per_position(lg, top_k)
            layer_row.tokens = _decode_ids(
                layer_row.top_ids[:, 0], self._tokenizer, seq_len
            )
            layer_row.probs = _top_k_probs_per_position(lg, top_k)

        input_tokens = _decode_ids(input_ids_1d, self._tokenizer, seq_len)

        return LogitLensGrid(
            layers=layers,
            input_tokens=input_tokens,
            input_ids=input_ids_1d.tolist(),
            n_positions=seq_len,
            top_k=top_k,
        )


# ----------------------------------------------------------------- dataclasses


@dataclass
class LayerLens:
    """A single layer's logit-lens prediction at a position."""

    label: str
    name: str
    logits: NdArray
    top_tokens: list[tuple[int, str, float, float]] = field(default_factory=list)
    top_indices: NdArray = None  # type: ignore[assignment]
    rank_of_final: int | None = None


@dataclass
class LogitLensResult:
    """The full logit-lens output across all layers (single position)."""

    rows: list[LayerLens]
    final_tokens: list[tuple[int, str, float, float]]
    final_rank: int | None
    pos: int
    top_k: int

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@dataclass
class LayerTokenLens:
    """One layer's top-k predictions at every sequence position."""

    label: str
    name: str
    logits: NdArray
    top_ids: NdArray = None  # type: ignore[assignment]  # [seq, k]
    tokens: list[str] = field(default_factory=list)  # top-1 token per position
    probs: NdArray = None  # type: ignore[assignment]  # [seq, k]


@dataclass
class LogitLensGrid:
    """Per-token logit lens: predictions at every position for every layer.

    Each :attr:`layers` entry holds top-k predictions for all positions.
    :attr:`input_tokens` are the actual prompt tokens for reference.
    """

    layers: list[LayerTokenLens]
    input_tokens: list[str]
    input_ids: list[int]
    n_positions: int
    top_k: int

    def __iter__(self):
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]


# ----------------------------------------------------------------- helpers IO


def _take_pos(logits: NdArray, pos: int) -> NdArray:
    """Extract ``pos`` from logits, squeezing batch/seq down to ``[vocab]``."""
    if logits.ndim >= 2:
        return logits[:, pos, :][0] if logits.ndim == 3 else logits[pos]
    return logits


def _top_k_tokens(
    logits_1d: NdArray,
    k: int,
    tokenizer: Any,
) -> tuple[list[tuple[int, str, float, float]], NdArray]:
    """Return ``(tokens, indices)`` for the top-k logits at one position.

    Each token is ``(token_id, decoded_str, logit, prob)`` sorted by logit desc.
    """
    k = min(k, logits_1d.shape[-1])
    idx = mx.argpartition(-logits_1d, kth=k - 1, axis=-1)[:k]
    vals = logits_1d[idx]
    order = mx.argsort(-vals)
    idx = idx[order]
    vals = vals[order]
    probs = mx.softmax(vals)

    idx_np = idx.astype(mx.int32)
    vals_np = vals.astype(mx.float32)
    probs_np = probs.astype(mx.float32)
    mx.eval(idx_np, vals_np, probs_np)

    out: list[tuple[int, str, float, float]] = []
    for tid, lg, pr in zip(
        idx_np.tolist(), vals_np.tolist(), probs_np.tolist(), strict=True
    ):
        if tokenizer is not None:
            try:
                tok = tokenizer.decode([tid])
            except Exception:  # pragma: no cover - tokenizer edge cases
                tok = str(tid)
        else:
            tok = str(tid)
        tok = tok.replace("\n", "\\n").replace("\t", "\\t")
        out.append((int(tid), tok, float(lg), float(pr)))
    return out, idx


def _rank_of_token(logits_1d: Any, token_id: int) -> int | None:
    """1-based rank of ``token_id`` within ``logits_1d`` (1 = top).

    Works with both MLX and PyTorch tensors.
    """
    target = float(logits_1d[token_id])
    # MLX uses mx.sum; torch uses torch.sum — check which is available
    if hasattr(logits_1d, "item") and hasattr(logits_1d, "dtype"):
        try:
            import mlx.core as mx

            if isinstance(logits_1d, mx.array):
                rank = int(mx.sum(logits_1d > target).item()) + 1
                return rank
        except ImportError:
            pass
    # PyTorch path or fallback
    import torch

    if isinstance(logits_1d, torch.Tensor):
        rank = int((logits_1d > target).sum().item()) + 1
        return rank
    # Generic fallback
    rank = sum(1 for v in logits_1d if float(v) > target) + 1
    return rank


def _top_k_ids_per_position(logits_2d: NdArray, k: int) -> NdArray:
    """Top-k token ids per position, sorted by logit descending.

    Args:
        logits_2d: ``[seq, vocab]``.
        k: Number of top tokens.

    Returns:
        ``[seq, k]`` int array.
    """
    k = min(k, logits_2d.shape[-1])
    idx = mx.argpartition(-logits_2d, kth=k - 1, axis=-1)[:, :k]
    vals = mx.take_along_axis(logits_2d, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    mx.eval(idx)
    return idx.astype(mx.int32)


def _top_k_probs_per_position(logits_2d: NdArray, k: int) -> NdArray:
    """Softmax probabilities for the top-k tokens per position.

    Args:
        logits_2d: ``[seq, vocab]``.
        k: Number of top tokens.

    Returns:
        ``[seq, k]`` float array.
    """
    k = min(k, logits_2d.shape[-1])
    idx = _top_k_ids_per_position(logits_2d, k)
    vals = mx.take_along_axis(logits_2d, idx, axis=-1)
    probs = mx.softmax(vals, axis=-1)
    mx.eval(probs)
    return probs.astype(mx.float32)


def _decode_ids(ids: NdArray, tokenizer: Any, max_len: int) -> list[str]:
    """Decode a 1-D array of token ids into a list of token strings."""
    id_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    tokens: list[str] = []
    for tid in id_list[:max_len]:
        if tokenizer is not None:
            try:
                tok = tokenizer.decode([int(tid)])
            except Exception:  # pragma: no cover - tokenizer edge cases
                tok = str(tid)
        else:
            tok = str(tid)
        tok = tok.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        tokens.append(tok)
    return tokens


def format_logit_lens(
    result: LogitLensResult,
    tokenizer: Any = None,
    show_tokens: int = 10,
) -> str:
    """Render a :class:`LogitLensResult` as a readable per-layer table."""
    show = max(1, min(show_tokens, result.top_k))
    lines: list[str] = []
    head = f"{'checkpoint':<11} {'top1':<16} {'p(top1)':>8}  top-{show}"
    lines.append(head)
    lines.append("-" * len(head))

    final_id = result.final_tokens[0][0] if result.final_tokens else None

    for row in result.rows:
        toks = row.top_tokens[:show]
        cells = []
        for tid, tok, _lg, pr in toks:
            mark = "*" if tid == final_id else " "
            cells.append(f"{mark}{tok[:14]:<14}{pr:6.1%}")
        top = toks[0] if toks else (0, "", 0.0, 0.0)
        lines.append(
            f"{row.label:<11} {top[1][:14]:<16} {top[3]:>8.1%}  "
            + "  ".join(cells)
        )

    if result.final_tokens:
        ft = result.final_tokens[0]
        lines.append("-" * len(head))
        lines.append(
            f"{'final':<11} {ft[1][:14]:<16} {ft[3]:>8.1%}  (model output logits)"
        )
    return "\n".join(lines)


def format_logit_lens_grid(
    grid: LogitLensGrid,
    tokenizer: Any = None,
    col_width: int = 12,
    show_probs: bool = False,
) -> str:
    """Render a :class:`LogitLensGrid` as a per-token table.

    Rows are layers (embed, layer 0, ..., final); columns are sequence
    positions. Each cell shows the top-1 predicted token (optionally with
    probability). A ``+`` prefix marks positions where the prediction matches
    the actual next input token.
    """
    tok = tokenizer
    if tok is not None and hasattr(tok, "tokenizer"):
        tok = tok.tokenizer

    n = grid.n_positions
    cw = col_width

    def _fmt_cell(token: str, prob: float, match: bool) -> str:
        prefix = "+" if match else " "
        t = token[: cw - 3]
        if show_probs:
            return f"{prefix}{t:<{cw - 7}}{prob:5.1%}"
        return f"{prefix}{t:<{cw - 1}}"

    lines: list[str] = []
    # Position header
    pos_header = f"{'':>11} "
    for p in range(n):
        pos_header += f"{'pos ' + str(p):<{cw}}"
    lines.append(pos_header)

    # Input tokens row
    input_row = f"{'input':>11} "
    for _p, t in enumerate(grid.input_tokens):
        input_row += _fmt_cell(t, 0.0, False)
    lines.append(input_row)
    lines.append("-" * len(pos_header))

    # Layer rows
    for layer_row in grid.layers:
        row = f"{layer_row.label:>11} "
        top_ids = layer_row.top_ids
        top_probs = layer_row.probs
        if top_ids is None:
            continue
        ids_np = top_ids[:, 0].tolist()
        probs_np = top_probs[:, 0].tolist() if top_probs is not None else [0.0] * n
        for p in range(n):
            pred_id = ids_np[p]
            cell_tok = layer_row.tokens[p] if p < len(layer_row.tokens) else str(pred_id)
            next_id = grid.input_ids[p + 1] if p + 1 < n else None
            match = next_id is not None and pred_id == next_id
            row += _fmt_cell(cell_tok, probs_np[p], match)
        lines.append(row)

    return "\n".join(lines)


# ----------------------------------------------------------------- torch backend
#
# The following classes mirror the MLX ``HookedMLXModel`` interface but use
# PyTorch forward hooks (``register_forward_pre_hook`` /
# ``register_forward_hook``) to capture the residual stream.  This allows the
# logit lens to work with any HuggingFace ``transformers`` model on CUDA, MPS,
# or CPU.


class _TorchLayerHooks:
    """Per-layer pre/post hooks that capture the residual stream."""

    def __init__(self, layer_idx: int, cache: ActivationCache):
        self.layer_idx = layer_idx
        self.cache = cache
        self._pre_handle: Any = None
        self._post_handle: Any = None

    def pre_hook(self, module: Any, args: tuple, kwargs: dict) -> Any:
        residual = args[0] if args else kwargs.get("hidden_states")
        if residual is not None:
            self.cache[f"blocks.{self.layer_idx}.hook_resid_pre"] = residual
        return None

    def post_hook(
        self, module: Any, args: tuple, kwargs: dict, output: Any
    ) -> Any:
        if isinstance(output, tuple):
            residual = output[0]
        else:
            residual = output
        if residual is not None:
            self.cache[f"blocks.{self.layer_idx}.hook_resid_post"] = residual
        return output


class HookedTorchModel:
    """A PyTorch model instrumented with TransformerLens-style hooks.

    Works with any HuggingFace ``transformers`` model (causal LM or VLM) on
    CUDA, MPS, or CPU.  Uses ``register_forward_pre_hook`` and
    ``register_forward_hook`` on each decoder layer to capture the residual
    stream entering (``hook_resid_pre``) and leaving
    (``hook_resid_post``) every layer.

    Example::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from logit_lens import HookedTorchModel

        model = AutoModelForCausalLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)
        hooked = HookedTorchModel(model, tokenizer)

        logits, cache = hooked.run_with_cache("The capital of France is")
        result = cache.logit_lens(pos=-1, top_k=5)
        print(format_logit_lens(result, tokenizer))
    """

    def __init__(self, model: Any, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self._is_vlm = self._detect_vlm(model)
        self._hook_ctxs: list[_TorchLayerHooks] = []
        self._final_norm, self._unembed, self._softcap = (
            self._detect_lens_components()
        )

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _detect_vlm(model: Any) -> bool:
        """Detect whether *model* is (or wraps) a VLM with a language_model."""
        if hasattr(model, "language_model"):
            return True
        inner = getattr(model, "model", None)
        return inner is not None and hasattr(inner, "language_model")

    def _get_inner_model(self) -> Any:
        """Return the inner text model (handles VLM wrappers).

        For a VLM such as ``Gemma4ForConditionalGeneration`` the hierarchy is::

            ForConditionalGeneration
              .model = Gemma4Model
                .language_model = Gemma4TextModel  (has .norm, .embed_tokens, .layers)

        For a plain causal LM such as ``Gemma4ForCausalLM``::

            ForCausalLM
              .model = Gemma4TextModel  (has .norm, .embed_tokens, .layers)

        vLLM uses a slightly different hierarchy for VLMs::

            ForConditionalGeneration
              .language_model = Gemma4ForCausalLM   (ForCausalLM wrapper)
                .model = Gemma4Model  (has .norm, .embed_tokens, .layers)
                .lm_head = ParallelLMHead
        """
        if self._is_vlm:
            lm = getattr(self.model, "language_model", None)
            if lm is None:
                inner = getattr(self.model, "model", None)
                lm = getattr(inner, "language_model", None) if inner is not None else None
            if lm is None:
                lm = getattr(self.model, "text_model", None)
            # vLLM wraps the text model inside a ForCausalLM; unwrap to .model
            # which holds .norm / .embed_tokens / .layers directly.
            if lm is not None and hasattr(lm, "model") and not hasattr(lm, "norm"):
                lm = lm.model
            return lm
        return getattr(self.model, "model", self.model)

    def _detect_lens_components(self) -> tuple[Any, Any, float | None]:
        inner = self._get_inner_model()

        final_norm = getattr(inner, "norm", None)
        if final_norm is None:
            final_norm = getattr(inner, "final_norm", None)
        if final_norm is None:
            final_norm = getattr(self.model, "norm", None)
        if final_norm is None:
            raise ValueError(
                "Could not locate the final norm in the torch model."
            )

        tied = bool(
            getattr(self.model.config, "tie_word_embeddings", False)
        )
        embed = getattr(inner, "embed_tokens", None)
        if embed is None:
            embed = getattr(inner, "wte", None)

        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            # vLLM VLM: lm_head lives on the language_model ForCausalLM wrapper
            lm_wrapper = getattr(self.model, "language_model", None)
            if lm_wrapper is not None:
                lm_head = getattr(lm_wrapper, "lm_head", None)
        if (tied or lm_head is None) and embed is not None:
            unembed = embed
        else:
            unembed = lm_head

        softcap = getattr(self.model, "final_logit_softcapping", None)
        if softcap is None:
            softcap = getattr(self.model.config, "final_logit_softcapping", None)
        if softcap is None:
            softcap = getattr(self.model.config, "logit_softcapping", None)
        if softcap is None:
            text_config_fn = getattr(self.model.config, "get_text_config", None)
            if text_config_fn is not None:
                softcap = getattr(text_config_fn(), "final_logit_softcapping", None)
        if softcap is not None:
            try:
                softcap = float(softcap)
            except (TypeError, ValueError):
                softcap = None
        return final_norm, unembed, softcap

    # --------------------------------------------------------------- properties

    @property
    def layers(self) -> Any:
        inner = self._get_inner_model()
        return getattr(inner, "layers", getattr(inner, "h", []))

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def cfg(self) -> Any:
        return getattr(self.model, "config", None)

    # ------------------------------------------------------------- forward/cache

    @property
    def _tokenizer(self) -> Any:
        tok = self.tokenizer
        if tok is not None and hasattr(tok, "tokenizer"):
            return tok.tokenizer
        return tok

    def _apply_chat_template(self, text: str) -> tuple[str, bool]:
        if not self._is_vlm or self.tokenizer is None:
            return text, True
        processor = self.tokenizer
        if not hasattr(processor, "chat_template"):
            return text, True
        try:
            config = getattr(self.model, "config", None)
            if config is None:
                return text, True
            model_type = getattr(config, "model_type", "")
            from mlx_vlm.prompt_utils import apply_chat_template

            fmt = apply_chat_template(
                processor, config, text, num_images=0
            )
            add_special = model_type not in (
                "gemma3", "gemma3n", "gemma4", "gemma4_unified",
            )
            return fmt, add_special
        except Exception:
            return text, True

    def _encode(self, text: str, prepend_bos: bool | None = None) -> Any:
        import torch

        tok = self._tokenizer
        if tok is None:
            raise ValueError("A tokenizer is required to encode text.")
        if prepend_bos is None:
            text, add_special = self._apply_chat_template(text)
            ids = tok.encode(text, add_special_tokens=add_special)
        else:
            ids = tok.encode(text, add_special_tokens=prepend_bos)
        return torch.tensor([ids], dtype=torch.long)

    def _encode_batch(self, texts: list[str]) -> tuple[Any, Any]:
        """Encode a list of texts with right-padding to equal length.

        Returns ``(input_ids, attention_mask)`` as torch LongTensors
        of shape ``[batch, seq]``.
        """
        import torch

        tok = self._tokenizer
        if tok is None:
            raise ValueError("A tokenizer is required to encode text.")

        encoded: list[list[int]] = []
        for text in texts:
            formatted, add_special = self._apply_chat_template(text)
            ids = tok.encode(formatted, add_special_tokens=add_special)
            encoded.append(ids)

        max_len = max(len(ids) for ids in encoded)
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id
        if pad_id is None:
            pad_id = 0

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def run_with_cache(
        self,
        input: str | Any,
        names: Iterable[str] | None = None,
        prepend_bos: bool | None = None,
        attention_mask: Any | None = None,
    ) -> tuple[Any, ActivationCache]:
        """Run a forward pass and cache the residual stream at every layer.

        Args:
            input: Prompt text or a token-id tensor ``[batch, seq]``.
            names: Optional iterable of hook-point names to restrict to.
            prepend_bos: Override for tokenizer special-token handling.
            attention_mask: Optional ``[batch, seq]`` mask.  Must be
                provided when ``input`` has ``batch > 1`` so padded
                positions do not contaminate real-token activations.

        Returns:
            ``(logits, cache)`` where ``logits`` are the model's real output
            logits and ``cache`` is an :class:`ActivationCache`.
        """
        import torch

        if isinstance(input, str):
            input_ids = self._encode(input, prepend_bos)
        else:
            input_ids = input
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        cache = ActivationCache(model=self)
        name_filter = set(names) if names is not None else None

        hooks: list[_TorchLayerHooks] = []
        for i, layer in enumerate(self.layers):
            ctx = _TorchLayerHooks(i, cache)
            ctx._pre_handle = layer.register_forward_pre_hook(
                ctx.pre_hook, with_kwargs=True
            )
            ctx._post_handle = layer.register_forward_hook(
                ctx.post_hook, with_kwargs=True
            )
            hooks.append(ctx)

        try:
            with torch.no_grad():
                if hasattr(self.model, "compute_logits"):
                    # vLLM models: forward() requires positions and returns
                    # hidden_states, not logits. Compute positions and then
                    # call compute_logits to get the final logits.
                    seq_len = input_ids.shape[1]
                    positions = torch.arange(
                        seq_len, device=input_ids.device
                    ).unsqueeze(0)
                    hidden_states = self.model(input_ids, positions)
                    logits = self.model.compute_logits(hidden_states)
                else:
                    # HuggingFace models: forward(input_ids) returns
                    # ModelOutput with .logits
                    kw: dict[str, Any] = {}
                    if attention_mask is not None:
                        kw["attention_mask"] = attention_mask
                    outputs = self.model(input_ids, **kw)
                    logits = (
                        outputs.logits
                        if hasattr(outputs, "logits")
                        else outputs
                    )
            for key in list(cache.keys()):
                if name_filter is not None and key not in name_filter:
                    del cache[key]
        finally:
            for ctx in hooks:
                ctx._pre_handle.remove()
                ctx._post_handle.remove()

        return logits, cache

    # ------------------------------------------------------------------- lens

    def lens_logits(self, residual: Any) -> Any:
        """Apply the logit lens to a residual-stream component."""
        import torch

        h = self._final_norm(residual)
        weight = getattr(self._unembed, "weight", None)
        if weight is not None:
            logits = torch.nn.functional.linear(h, weight)
        else:
            logits = self._unembed(h)
        if self._softcap is not None:
            logits = torch.tanh(logits / self._softcap) * self._softcap
        return logits

    def logit_lens(
        self,
        cache: ActivationCache,
        pos: int = -1,
        top_k: int = 10,
        final_logits: Any | None = None,
    ) -> LogitLensResult:
        """Compute logit-lens predictions for every layer from a cache."""
        rows: list[LayerLens] = []
        points: list[tuple[str, str]] = [
            ("embed", "blocks.0.hook_resid_pre")
        ]
        for i in range(self.n_layers):
            points.append((f"layer {i}", f"blocks.{i}.hook_resid_post"))

        for label, name in points:
            if name not in cache:
                continue
            logits = self.lens_logits(cache[name])
            rows.append(LayerLens(label=label, name=name, logits=logits))

        final_tokens: list[tuple[int, str, float, float]] = []
        final_rank: int | None = None
        ref_logits = final_logits if final_logits is not None else (
            rows[-1].logits if rows else None
        )
        if ref_logits is not None:
            ref_pos = _torch_take_pos(ref_logits, pos)
            final_tokens, _ = _torch_top_k_tokens(
                ref_pos, top_k, self._tokenizer
            )
            if rows:
                final_rank = _rank_of_token(
                    _torch_take_pos(rows[-1].logits, pos),
                    final_tokens[0][0] if final_tokens else 0,
                )

        for row in rows:
            lp = _torch_take_pos(row.logits, pos)
            row.top_tokens, row.top_indices = _torch_top_k_tokens(
                lp, top_k, self._tokenizer
            )
            if final_tokens:
                row.rank_of_final = _rank_of_token(
                    lp, final_tokens[0][0]
                )

        return LogitLensResult(
            rows=rows,
            final_tokens=final_tokens,
            final_rank=final_rank,
            pos=pos,
            top_k=top_k,
        )

    def logit_lens_per_token(
        self,
        cache: ActivationCache,
        input_ids: Any,
        final_logits: Any | None = None,
        top_k: int = 1,
    ) -> LogitLensGrid:
        """Compute per-token logit-lens predictions for every layer."""
        import torch

        layers: list[LayerTokenLens] = []
        points: list[tuple[str, str]] = [
            ("embed", "blocks.0.hook_resid_pre")
        ]
        for i in range(self.n_layers):
            points.append((f"layer {i}", f"blocks.{i}.hook_resid_post"))

        for label, name in points:
            if name not in cache:
                continue
            logits = self.lens_logits(cache[name])
            layers.append(
                LayerTokenLens(label=label, name=name, logits=logits)
            )

        if final_logits is not None:
            layers.append(
                LayerTokenLens(label="final", name="final", logits=final_logits)
            )

        if isinstance(input_ids, torch.Tensor):
            input_ids_1d = (
                input_ids[0] if input_ids.ndim == 2 else input_ids
            )
            input_ids_list = input_ids_1d.tolist()
        else:
            input_ids_1d = input_ids
            input_ids_list = list(input_ids_1d)
        seq_len = len(input_ids_list)

        for layer_row in layers:
            lg = layer_row.logits
            if isinstance(lg, torch.Tensor):
                if lg.ndim == 3:
                    lg = lg[0]
                lg_np = lg.detach().cpu()
                layer_row.top_ids = _torch_top_k_ids_per_position(lg_np, top_k)
                layer_row.tokens = _decode_ids(
                    layer_row.top_ids[:, 0].tolist(),
                    self._tokenizer,
                    seq_len,
                )
                layer_row.probs = _torch_top_k_probs_per_position(
                    lg_np, top_k
                )

        input_tokens = _decode_ids(
            input_ids_list, self._tokenizer, seq_len
        )

        return LogitLensGrid(
            layers=layers,
            input_tokens=input_tokens,
            input_ids=input_ids_list,
            n_positions=seq_len,
            top_k=top_k,
        )

    def logit_lens_batch(
        self,
        cache: ActivationCache,
        input_ids: Any,
        attention_mask: Any,
        top_k: int = 10,
        final_logits: Any | None = None,
        return_logits: bool = False,
    ) -> list[LogitLensGrid] | tuple[list[LogitLensGrid], list[Any]]:
        """Per-token logit lens for a batch of samples.

        Computes ``lens_logits`` once per layer on the full
        ``[batch, seq, hidden]`` tensor, immediately extracts top-k,
        and discards the large ``[batch, seq, vocab]`` tensor before
        moving to the next layer.  Padded positions are excluded from
        the per-sample results.

        Results are identical to running :meth:`logit_lens_per_token`
        on each sample individually (with a proper ``attention_mask``
        so padded positions do not affect real-token activations).
        """
        import torch

        B = input_ids.shape[0]
        device = input_ids.device

        points: list[tuple[str, str]] = [
            ("embed", "blocks.0.hook_resid_pre")
        ]
        for i in range(self.n_layers):
            points.append((f"layer {i}", f"blocks.{i}.hook_resid_post"))

        seq_lens = attention_mask.sum(dim=1).tolist()

        all_logits_per_sample: list[list[Any]] | None = (
            [[] for _ in range(B)] if return_logits else None
        )

        # Per-layer: (label, name, topk_idx [B,S,k], probs [B,S,k],
        #            tokens [B][list[str]])
        layer_data: list[tuple[str, str, Any, Any, list[list[str]]]] = []

        for label, name in points:
            if name not in cache:
                continue
            h = cache[name]  # [B, S, H]
            lens_lg = self.lens_logits(h)  # [B, S, V]

            if return_logits and all_logits_per_sample is not None:
                for b in range(B):
                    rl = seq_lens[b]
                    all_logits_per_sample[b].append(
                        lens_lg[b, :rl].cpu()
                    )

            k = min(top_k, lens_lg.shape[-1])
            topk_vals, topk_idx = torch.topk(lens_lg, k, dim=-1)
            del lens_lg
            probs = torch.softmax(topk_vals, dim=-1)
            del topk_vals

            top1_ids = topk_idx[:, :, 0]  # [B, S]
            tokens_per_sample: list[list[str]] = []
            for b in range(B):
                rl = seq_lens[b]
                ids_b = top1_ids[b, :rl].tolist()
                tokens_b = _decode_ids(ids_b, self._tokenizer, rl)
                tokens_per_sample.append(tokens_b)

            layer_data.append((label, name, topk_idx, probs, tokens_per_sample))

        if final_logits is not None:
            k = min(top_k, final_logits.shape[-1])
            all_probs = torch.softmax(final_logits, dim=-1)
            topk_vals, topk_idx = torch.topk(final_logits, k, dim=-1)
            probs, _ = torch.topk(all_probs, k, dim=-1)
            # probs = torch.softmax(topk_vals, dim=-1)
            del topk_vals

            top1_ids = topk_idx[:, :, 0]
            tokens_per_sample: list[list[str]] = []
            for b in range(B):
                rl = seq_lens[b]
                ids_b = top1_ids[b, :rl].tolist()
                tokens_b = _decode_ids(ids_b, self._tokenizer, rl)
                tokens_per_sample.append(tokens_b)

            layer_data.append(
                ("final", "final", topk_idx, probs, tokens_per_sample)
            )

        results: list[LogitLensGrid] = []
        for b in range(B):
            rl = seq_lens[b]
            input_ids_b = input_ids[b, :rl]
            input_tokens = _decode_ids(
                input_ids_b.tolist(), self._tokenizer, rl
            )

            layers: list[LayerTokenLens] = []
            for label, name, tk_all, pr_all, toks_all in layer_data:
                layers.append(LayerTokenLens(
                    label=label,
                    name=name,
                    logits=None,
                    top_ids=tk_all[b, :rl],
                    tokens=toks_all[b],
                    probs=pr_all[b, :rl],
                ))

            results.append(LogitLensGrid(
                layers=layers,
                input_tokens=input_tokens,
                input_ids=input_ids_b.tolist(),
                n_positions=rl,
                top_k=top_k,
            ))

        if return_logits and all_logits_per_sample is not None:
            stacked = [torch.stack(ls) for ls in all_logits_per_sample]
            return results, stacked
        return results

    def run_lens_batch(
        self,
        texts: list[str],
        top_k: int = 10,
        return_logits: bool = False,
    ) -> list[LogitLensGrid] | list[tuple[LogitLensGrid, Any]]:
        """Full lens pipeline on a batch of texts.

        Encodes all texts with right-padding, runs one forward pass
        with ``attention_mask``, computes the logit lens for all
        layers, and returns per-sample grids with padded positions
        excluded.
        """
        import torch

        input_ids, attention_mask = self._encode_batch(texts)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits, cache = self.run_with_cache(
            input_ids, attention_mask=attention_mask
        )
        if return_logits:
            grids, logits_list = self.logit_lens_batch(
                cache, input_ids, attention_mask,
                top_k=top_k, final_logits=logits,
                return_logits=True,
            )
            return list(zip(grids, logits_list))
        return self.logit_lens_batch(
            cache, input_ids, attention_mask,
            top_k=top_k, final_logits=logits,
        )

    def remove_hooks(self) -> None:
        """No-op for torch (hooks are registered per-call)."""
        pass

    def restore(self) -> None:
        """No-op for torch (no class-level patches are made)."""
        pass


# ----------------------------------------------------------------- nnsight backend
#
# The following class mirrors the ``HookedTorchModel`` interface but uses
# nnsight's vLLM tracing API to capture intermediate activations.


class HookedNNsightModel:
    """Logit lens via nnsight's vLLM tracing interface.

    All module access (layers, final norm, unembed) and lens computation
    happens **inside** the nnsight trace so Envoy proxies resolve
    correctly on the worker process where the real weights live.  Only
    the small top-k results are ``.save()``-ed back to the client.
    """

    def __init__(self, nnsight_model: Any, tokenizer: Any = None):
        self.model = nnsight_model
        self.tokenizer = tokenizer

        raw_tok = tokenizer
        if raw_tok is not None and hasattr(raw_tok, "tokenizer"):
            raw_tok = raw_tok.tokenizer
        self._raw_tok = raw_tok

        self._n_layers = self._detect_n_layers()

    # --------------------------------------------------------------- structure

    def _detect_n_layers(self) -> int:
        """Get layer count from model config without touching Envoys."""
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
            for attr in ("num_hidden_layers", "num_layers", "n_layers"):
                val = getattr(tc, attr, None)
                if val is not None:
                    return int(val)
        return 0

    def _get_inner_model(self) -> Any:
        """Return the Envoy wrapping the text model, accessed inside trace."""
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model
            if hasattr(lm, "layers"):
                return lm
        if hasattr(m, "model") and hasattr(m.model, "model") and hasattr(m.model.model, "layers"):
            return m.model.model
        return m

    def _get_unembed(self, inner: Any) -> Any:
        """Return the Envoy for the unembedding (lm_head or embed_tokens)."""
        m = self.model
        lm_head = getattr(m, "lm_head", None)
        if lm_head is not None:
            return lm_head
        lm_wrapper = getattr(m, "language_model", None)
        if lm_wrapper is not None:
            lm_head = getattr(lm_wrapper, "lm_head", None)
            if lm_head is not None:
                return lm_head
        return getattr(inner, "embed_tokens", None)

    def _get_tied(self) -> bool:
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
            return bool(getattr(tc, "tie_word_embeddings", False))
        return False

    def _get_softcap(self) -> float | None:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return None
        tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        for attr in ("final_logit_softcapping", "logit_softcapping"):
            sc = getattr(tc, attr, None)
            if sc is not None:
                try:
                    return float(sc)
                except (TypeError, ValueError):
                    pass
        return None

    def _cohere_layernorm(self, hidden_states: Any, norm_weight: Any, norm_eps: Any) -> Any:
        import torch

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + norm_eps)
        hidden_states = norm_weight.to(torch.float32) * hidden_states
        hidden_states = hidden_states.to(input_dtype)

        return hidden_states

    # --------------------------------------------------------------- properties

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def _tokenizer(self) -> Any:
        return self._raw_tok

    # --------------------------------------------------------------- encoding

    def _encode(self, text: str) -> Any:
        import torch

        tok = self._raw_tok
        ids = tok.encode(text, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long)

    # --------------------------------------------------------------- batch lens

    def run_lens_batch(
        self,
        texts: list[str],
        top_k: int = 10,
        return_logits: bool = False,
    ) -> list[LogitLensGrid] | list[tuple[LogitLensGrid, Any]]:
        """Run logit lens on a batch of texts via nnsight tracing.

        All texts are processed in a **single trace** with one
        ``tracer.invoke(text)`` per prompt.  vLLM batches all invokes
        into one forward pass for GPU efficiency.  For each text the
        full lens computation (norm + unembed + top-k + softmax) runs
        inside the trace on the worker; only the small top-k results are
        downloaded.
        """
        import torch
        import torch.nn.functional as F

        tok = self._raw_tok
        n_layers = self._n_layers
        softcap = self._get_softcap()
        tied = self._get_tied()

        all_ids: list[list[int]] = [
            tok.encode(t, add_special_tokens=True) for t in texts
        ]

        all_saved_logits = None
        with self.model.trace() as tracer:
            all_saved: list[list[tuple[Any, Any]]] = nnsight.save(
                [list() for _ in texts]
            )
            if return_logits:
                all_saved_logits = nnsight.save(
                    [list() for _ in texts]
                )
            for i, text in enumerate(texts):
                with tracer.invoke(text):
                    # print(i, text)
                    inner = self._get_inner_model()
                    # print(type(inner._module).__name__)
                    layers = inner.layers
                    # print("num_layers", len(layers))
                    final_norm = inner.norm
                    # print("final_norm", final_norm, final_norm.weight, final_norm.weight.shape)
                    unembed = self._get_unembed(inner)
                    # print("unembed", unembed, unembed.weight, unembed.weight.shape)

                    # print(f"layers[0]-{i}", layers[0].inputs[0])

                    # Collect all residual stream points.
                    # In vLLM, decoder layer input is (positions,
                    # hidden_states, residual, ...) — .input returns
                    # positions (int64), so use .inputs[0][1] for the
                    # hidden states entering layer 0 (the embed
                    # residual).
                    if type(inner._module).__name__ in ["Gemma4Model", "LlamaModel", "CohereModel"]:
                        embed_resid = layers[0].inputs[0][1]
                    elif type(inner._module).__name__ in ["Qwen3_5Model"]:
                        embed_resid = layers[0].inputs[1]["hidden_states"]

                    # print(f"embed_resid-{i}", embed_resid, embed_resid.shape)

                    # if type(inner._module).__name__ in ["CohereModel"]:
                    #     norm_weight = final_norm.weight
                    #     norm_eps = (
                    #         getattr(final_norm, "variance_epsilon", None)
                    #         or getattr(final_norm, "eps", None)
                    #         or 1e-6
                    #     )
                    #     normed = self._cohere_layernorm(embed_resid, norm_weight, norm_eps)
                    # else:
                    #     normed = final_norm(embed_resid)

                    # # print(f"normed-i{i}-j{j}", normed)

                    # logits = F.linear(normed, unembed.weight)
                    # if softcap is not None:
                    #     logits = torch.tanh(logits / softcap) * softcap
                    # k = min(top_k, logits.shape[-1])
                    # topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
                    # probs = F.softmax(topk_vals, dim=-1)
                    # print(f"topk_vals-i{i}-j{j}", topk_vals)
                    # print(f"probs-i{i}-j{j}", probs)
                    # all_saved[i].append(
                    #     (topk_idx.save(), probs.save())
                    # )

                    residuals = [embed_resid]
                    # for j in range(n_layers):
                    for j, layer in enumerate(layers):
                        # print(f"layer-i{i}-j{j}", layer.output)
                        out = layer.output[0]
                        # print(f"out-i{i}-j{j}", out)
                        residuals.append(out)

                        # resid = out

                        # print(f"resid-i{i}-j{j}", resid, resid.shape)
                        # if type(inner._module).__name__ in ["CohereModel"]:
                        #     norm_weight = final_norm.weight
                        #     norm_eps = (
                        #         getattr(final_norm, "variance_epsilon", None)
                        #         or getattr(final_norm, "eps", None)
                        #         or 1e-6
                        #     )
                        #     normed = self._cohere_layernorm(resid, norm_weight, norm_eps)
                        # else:
                        #     normed = final_norm(resid)

                        # print(f"normed-i{i}-j{j}", normed)

                        # logits = F.linear(normed, unembed.weight)
                        # if softcap is not None:
                        #     logits = torch.tanh(logits / softcap) * softcap
                        # k = min(top_k, logits.shape[-1])
                        # topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
                        # probs = F.softmax(topk_vals, dim=-1)
                        # print(f"topk_vals-i{i}-j{j}", topk_vals)
                        # print(f"probs-i{i}-j{j}", probs)
                        # all_saved[i].append(
                        #     (topk_idx.save(), probs.save())
                        # )

                    assert len(residuals) == len(layers) + 1, f"len(residuals) = {len(residuals)} != len(layers) + 1 = {len(layers)}"

                    # print("residuals", residuals)

                    # Apply lens to each residual inside the trace.
                    # Don't call final_norm(resid) directly — vLLM fused
                    # norms (LayerNorm, GemmaRMSNorm) have different
                    # forward() signatures (expecting hidden_states +
                    # residual) that crash when called with one arg.
                    # Instead, read the norm's weight and eps and apply
                    # normalization manually.
                    # norm_weight = final_norm.weight
                    # norm_eps = (
                    #     getattr(final_norm, "variance_epsilon", None)
                    #     or getattr(final_norm, "eps", None)
                    #     or 1e-6
                    # )

                    for j, resid in enumerate(residuals):
                        # print(f"resid-i{i}-j{j}", resid, resid.shape)
                        if type(inner._module).__name__ in ["CohereModel"]:
                            norm_weight = final_norm.weight
                            norm_eps = (
                                getattr(final_norm, "variance_epsilon", None)
                                or getattr(final_norm, "eps", None)
                                or 1e-6
                            )
                            normed = self._cohere_layernorm(resid, norm_weight, norm_eps)
                        else:
                            normed = final_norm(resid)

                        # print(f"normed-i{i}-j{j}", normed)

                        logits = F.linear(normed, unembed.weight)

                        # print(f"logits-i{i}-j{j}", logits)

                        if softcap is not None:
                            logits = torch.tanh(logits / softcap) * softcap
                        k = min(top_k, logits.shape[-1])
                        # k = min(10, logits.shape[-1])

                        # topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
                        # probs = F.softmax(topk_vals, dim=-1)

                        all_probs = F.softmax(logits, dim=-1)
                        probs, _ = torch.topk(all_probs, k, dim=-1)
                        _, topk_idx = torch.topk(logits, k, dim=-1)

                        # print(f"topk_idx-i{i}-j{j}", topk_idx)
                        # print(f"probs-i{i}-j{j}", probs)
                        all_saved[i].append(
                            (topk_idx.save(), probs.save())
                        )
                        if return_logits and all_saved_logits is not None:
                            all_saved_logits[i].append(logits.save())

        # Build LogitLensGrid for each text from saved results
        labels = ["embed"] + [f"layer {j}" for j in range(n_layers)]
        results: list[Any] = []

        assert all([len(all_saved[i]) > 0 for i in range(len(all_saved))]), [i for i in range(len(all_saved)) if len(all_saved[i]) == 0]

        # print("all_saved", all_saved)

        for i, input_ids_list in enumerate(all_ids):
            seq_len = len(input_ids_list)
            saved = all_saved[i]

            layer_data: list[tuple[str, str, Any, Any, list[str]]] = []
            for label, (tk_proxy, pr_proxy) in zip(labels, saved):
                tk = tk_proxy
                pr = pr_proxy
                if tk.ndim == 3:
                    tk = tk[0]
                if pr.ndim == 3:
                    pr = pr[0]
                top1_ids = tk[:, 0]
                tokens = _decode_ids(top1_ids.tolist(), self._raw_tok, seq_len)
                layer_data.append((label, "", tk, pr, tokens))

            input_tokens = _decode_ids(input_ids_list, self._raw_tok, seq_len)

            layers_out: list[LayerTokenLens] = []
            for label, name, tk, pr, toks in layer_data:
                layers_out.append(LayerTokenLens(
                    label=label,
                    name=name,
                    logits=None,
                    top_ids=tk,
                    tokens=toks,
                    probs=pr,
                ))

            results.append(LogitLensGrid(
                layers=layers_out,
                input_tokens=input_tokens,
                input_ids=input_ids_list,
                n_positions=seq_len,
                top_k=top_k,
            ))

            if return_logits and all_saved_logits is not None:
                saved_lg = all_saved_logits[i]
                lg_tensors = []
                for sl in saved_lg:
                    if sl.ndim == 3:
                        sl = sl[0]
                    lg_tensors.append(sl)
                stacked_lg = torch.stack(lg_tensors)
                results[-1] = (results[-1], stacked_lg)

        return results

    def restore(self) -> None:
        """No-op for nnsight (no persistent hooks)."""
        pass


# ----------------------------------------------------------------- torch helpers


def _torch_take_pos(logits: Any, pos: int) -> Any:
    """Extract ``pos`` from a torch logits tensor, squeezing to ``[vocab]``."""
    if logits.ndim >= 2:
        return logits[:, pos, :][0] if logits.ndim == 3 else logits[pos]
    return logits


def _torch_top_k_tokens(
    logits_1d: Any,
    k: int,
    tokenizer: Any,
) -> tuple[list[tuple[int, str, float, float]], Any]:
    """Return ``(tokens, indices)`` for the top-k logits at one position."""
    import torch

    k = min(k, logits_1d.shape[-1])
    vals, idx = torch.topk(logits_1d, k)
    probs = torch.softmax(vals, dim=-1)

    idx_list = idx.tolist()
    vals_list = vals.tolist()
    probs_list = probs.tolist()

    out: list[tuple[int, str, float, float]] = []
    for tid, lg, pr in zip(
        idx_list, vals_list, probs_list, strict=True
    ):
        if tokenizer is not None:
            try:
                tok = tokenizer.decode([tid])
            except Exception:
                tok = str(tid)
        else:
            tok = str(tid)
        tok = tok.replace("\n", "\\n").replace("\t", "\\t")
        out.append((int(tid), tok, float(lg), float(pr)))
    return out, idx


def _torch_top_k_ids_per_position(logits_2d: Any, k: int) -> Any:
    """Top-k token ids per position for a torch tensor ``[seq, vocab]``."""
    import torch

    k = min(k, logits_2d.shape[-1])
    _, idx = torch.topk(logits_2d, k, dim=-1)
    return idx


def _torch_top_k_probs_per_position(logits_2d: Any, k: int) -> Any:
    """Softmax probabilities for the top-k tokens per position."""
    import torch

    k = min(k, logits_2d.shape[-1])
    idx = _torch_top_k_ids_per_position(logits_2d, k)
    vals = torch.gather(logits_2d, -1, idx)
    probs = torch.softmax(vals, dim=-1)
    return probs


# ----------------------------------------------------------------- batch helpers


def grid_to_lens_result(
    grid: LogitLensGrid,
    pos: int,
    top_k: int,
    tokenizer: Any,
) -> LogitLensResult:
    """Extract a single-position :class:`LogitLensResult` from a grid.

    Allows reusing batch-computed lens results to produce the same
    output as :meth:`HookedTorchModel.logit_lens` at a specific
    position.  ``pos`` may be negative (standard Python indexing).

    The ``logit`` value in each ``top_tokens`` tuple is set to 0.0
    (callers only use the token id, string, and probability).
    ``rank_of_final`` and ``final_rank`` are ``None`` because the
    full vocab logits are not retained in batch mode.
    """
    raw_tok = tokenizer
    if raw_tok is not None and hasattr(raw_tok, "tokenizer"):
        raw_tok = raw_tok.tokenizer

    k = min(top_k, grid.top_k)

    if pos < 0:
        pos = grid.n_positions + pos

    rows: list[LayerLens] = []
    final_tokens: list[tuple[int, str, float, float]] = []

    for layer in grid.layers:
        if layer.top_ids is None:
            continue
        if pos < 0 or pos >= layer.top_ids.shape[0]:
            continue

        ids = layer.top_ids[pos].tolist()
        probs = (
            layer.probs[pos].tolist()
            if layer.probs is not None
            else [0.0] * len(ids)
        )

        top_tokens: list[tuple[int, str, float, float]] = []
        for j in range(min(k, len(ids))):
            tid = int(ids[j])
            pr = float(probs[j])
            if raw_tok is not None:
                try:
                    tok_str = raw_tok.decode([tid])
                except Exception:
                    tok_str = str(tid)
            else:
                tok_str = str(tid)
            tok_str = tok_str.replace("\n", "\\n").replace("\t", "\\t")
            top_tokens.append((tid, tok_str, 0.0, pr))

        if layer.label == "final":
            final_tokens = top_tokens
        else:
            rows.append(LayerLens(
                label=layer.label,
                name=layer.name,
                logits=None,
                top_tokens=top_tokens,
                top_indices=layer.top_ids[pos],
            ))

    if not final_tokens and rows:
        final_tokens = rows[-1].top_tokens

    return LogitLensResult(
        rows=rows,
        final_tokens=final_tokens,
        final_rank=None,
        pos=pos,
        top_k=k,
    )


# ----------------------------------------------------------------- factory


def _is_torch_model(model: Any) -> bool:
    """Return True if *model* is a PyTorch ``nn.Module``."""
    try:
        import torch.nn as nn_torch

        return isinstance(model, nn_torch.Module)
    except ImportError:
        return False


def _is_mlx_model(model: Any) -> bool:
    """Return True if *model* is an MLX ``nn.Module``."""
    try:
        import mlx.nn as nn_mlx

        return isinstance(model, nn_mlx.Module)
    except ImportError:
        return False


def make_hooked_model(model: Any, tokenizer: Any = None) -> Any:
    """Factory: detect model type and return the appropriate hooked wrapper.

    Returns a :class:`HookedMLXModel` for MLX models or a
    :class:`HookedTorchModel` for PyTorch models.  The returned object
    implements the same interface (``run_with_cache``, ``lens_logits``,
    ``logit_lens``, ``logit_lens_per_token``, ``remove_hooks``, ``restore``).
    """
    if _is_mlx_model(model):
        return HookedMLXModel(model, tokenizer)
    if _is_torch_model(model):
        return HookedTorchModel(model, tokenizer)
    raise TypeError(
        f"Cannot create hooked model from {type(model).__name__}: "
        "expected an MLX nn.Module or a torch nn.Module."
    )
