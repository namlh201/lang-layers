"""GPU detection, model loading, and dataset loading utilities.

This module centralises all backend-related code: detecting the available
GPU, loading models with the appropriate backend (MLX, vLLM, or
transformers), and loading translation datasets (FLORES+, WMT24++).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langcodes import Language

# --------------------------------------------------------------------------- #
#  GPU / backend detection
# --------------------------------------------------------------------------- #


def detect_gpu() -> str:
    """Detect the best available GPU and return the backend name.

    Priority: ``cuda`` > ``mlx`` > ``mps`` > ``cpu``.

    * ``cuda``   -- NVIDIA GPU via PyTorch CUDA
    * ``mlx``    -- Apple Silicon via MLX (native Metal)
    * ``mps``    -- Apple Silicon via PyTorch MPS
    * ``cpu``    -- CPU fallback
    """
    # 1. CUDA (NVIDIA)
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # 2. MLX (Apple Silicon native)
    try:
        import mlx.core as mx  # noqa: F401

        return "mlx"
    except ImportError:
        pass

    # 3. MPS (Apple Silicon via PyTorch)
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


# --------------------------------------------------------------------------- #
#  Backend detection and model loading
# --------------------------------------------------------------------------- #

# model_type values that mlx_lm supports natively (text-only).
# Everything else (or unknown) goes through mlx_vlm.
_LM_MODEL_TYPES: set[str] = {
    "gemma2_text", "gemma3_text", "gemma4_text",
    "llama", "qwen2", "qwen3", "mistral", "phi3", "phi",
    "gemma", "gemma2", "gpt2", "gpt_neox", "starcoder2",
    "cohere", "internlm2", "openelm", "olmo", "olmo2",
    "stablelm", "solar",
}


def _hf_config_model_type(model_path: str) -> str | None:
    """Try to read ``model_type`` from the HF config of ``model_path``."""
    import json as _json
    from pathlib import Path as _Path

    try:
        from huggingface_hub import hf_hub_download

        local = _Path(model_path)
        if local.is_dir():
            cfg_file = local / "config.json"
            if cfg_file.exists():
                with open(cfg_file) as f:
                    return _json.load(f).get("model_type")
        path = hf_hub_download(repo_id=model_path, filename="config.json")
        with open(path) as f:
            return _json.load(f).get("model_type")
    except Exception:
        return None


def detect_backend(model_path: str, forced: str | None = None) -> str:
    """Return ``"vlm"`` or ``"lm"`` for the given model path."""
    if forced is not None:
        return forced
    mt = _hf_config_model_type(model_path)
    if mt is None:
        return "vlm"
    if mt in _LM_MODEL_TYPES:
        return "lm"
    return "vlm"


def load_nnsight_model(
    model_path: str,
    device: str | None = None,
) -> tuple:
    """Load a model via nnsight's VLLM backend.

    Uses :class:`nnsight.modeling.vllm.VLLM` which wraps vLLM's engine
    and injects nnsight's interleaving machinery (custom worker class,
    model runner monkey-patching, mediator transport via
    ``SamplingParams.extra_args``) so interventions can observe and
    modify intermediate activations during vLLM's forward pass.

    Pass ``dispatch=True`` so real weights are loaded (not meta tensors).
    ``enforce_eager=True`` avoids CUDA graph compilation which is
    incompatible with nnsight's interleaving hooks.  Tensor-parallel size
    is taken from ``SLURM_GPUS`` when available.

    Returns ``(hooked, tokenizer)`` where ``hooked`` is a
    :class:`~logit_lens.HookedNNsightModel`.
    """
    import sys
    import torch
    from nnsight.modeling.vllm import VLLM

    # Disable vLLM's multiprocessing so the engine core, executor, and
    # worker all run in the main process.  This is required because
    # nnsight monkey-patches GPUModelRunner via NNsightGPUWorker.__init__,
    # which must execute in the same process where init_device() creates
    # the model_runner.  With multiprocessing enabled (vLLM 0.26+), the
    # engine core runs in a subprocess and the monkey-patch may not
    # propagate correctly.
    # os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Force V1 model runner — nnsight only patches
    # vllm.v1.worker.gpu_model_runner.GPUModelRunner (V1 path).
    # vLLM 0.26+ defaults to V2 for non-MoE models, which imports from
    # vllm.v1.worker.gpu.model_runner (different module, unpatched).
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    kwargs: dict = {}
    if torch.cuda.is_available():
        kwargs["tensor_parallel_size"] = int(os.environ.get("SLURM_GPUS", 1))
        kwargs["dtype"] = "bfloat16"
        kwargs["gpu_memory_utilization"] = 0.85

    print(f"  Loading {model_path} via nnsight VLLM", file=sys.stderr)

    model = VLLM(model_path, dispatch=True, **kwargs)
    tokenizer = model.tokenizer

    # print(model)

    from logit_lens import HookedNNsightModel

    hooked = HookedNNsightModel(model, tokenizer)
    return hooked, tokenizer


def load_model(
    model_path: str,
    backend: str | None = None,
    use_vllm: bool = True,
    device: str | None = None,
) -> tuple:
    """Load a model with the appropriate backend.

    Auto-detects GPU and dispatches to the right loader:

    * **MLX** (Apple Silicon): uses ``vllm_mlx`` (if *use_vllm* and available)
      or ``mlx_lm`` / ``mlx_vlm``.
    * **CUDA / MPS / CPU** (NVIDIA or Apple-via-torch): uses ``vllm``
      (if *use_vllm* and available) or ``transformers``.

    Returns ``(model, tokenizer_or_processor, backend_kind, vllm_wrapper)``.

    * ``backend_kind`` is one of ``"mlx-lm"``, ``"mlx-vlm"``, ``"torch"``,
      ``"vllm"``.
    * ``vllm_wrapper`` is the vllm / vllm-mlx object used for generation
      (or ``None`` if using the raw model).
    * ``model`` is always the raw MLX or torch model suitable for
      :func:`make_hooked_model`.
    """
    gpu = device or detect_gpu()

    # ---- MLX path (Apple Silicon native) ----
    if gpu == "mlx":
        mlx_backend = detect_backend(model_path, backend)

        if use_vllm:
            try:
                if mlx_backend == "vlm":
                    from vllm_mlx.models import MLXMultimodalLM

                    vllm_model = MLXMultimodalLM(model_path)
                    vllm_model.load()
                    return (
                        vllm_model.model,
                        vllm_model.processor,
                        "mlx-vlm",
                        vllm_model,
                    )
                else:
                    from vllm_mlx.models import MLXLanguageModel

                    vllm_model = MLXLanguageModel(model_path)
                    vllm_model.load()
                    return (
                        vllm_model.model,
                        vllm_model.tokenizer,
                        "mlx-lm",
                        vllm_model,
                    )
            except ImportError:
                pass  # fall through to mlx_lm / mlx_vlm

        if mlx_backend == "vlm":
            from mlx_vlm import load

            model, processor = load(model_path)
            return model, processor, "mlx-vlm", None
        else:
            from mlx_lm import load

            model, tokenizer = load(model_path)
            return model, tokenizer, "mlx-lm", None

    # ---- torch / vLLM path (CUDA, MPS, CPU) ----
    import torch

    dtype = torch.bfloat16 # if gpu in ("cuda", "mps") else torch.float32
    torch_device = gpu

    if use_vllm:
        try:
            from vllm import LLM

            llm = LLM(
                model=model_path,
                dtype="bfloat16", # if gpu == "cuda" else "float32",
                tensor_parallel_size=int(os.environ.get("SLURM_GPUS", 1)),
                # gpu_memory_utilization=0.75,
            )
            hf_model = None
            # vLLM exposes the HF model internally for lens extraction
            # try:
            #     # vLLM V1 API (apply_model works in both uni- and multi-proc)
            #     hf_model = llm.apply_model(lambda m: m)[0]
            # except Exception as e:
            #     print("vLLM v1", e)
            #     # vLLM V0 fallback
            #     hf_model = llm.llm_engine.model_executor.driver_worker.model_runner.model  # type: ignore[attr-defined]
            tokenizer = llm.get_tokenizer()
            return hf_model, tokenizer, "vllm", llm
        except ImportError as e:
            print("ImportError", e)
            pass
        except Exception as e:
            print("Exception", e)
            pass  # fall through to transformers

    # transformers fallback
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=torch_device if torch_device != "cpu" else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    return hf_model, tokenizer, "torch", None


def load_hf_model(
    model_path: str,
    device: str | None = None,
) -> tuple:
    """Load a model in HuggingFace transformers format.

    Returns ``(model, tokenizer, device)``.
    """
    import sys
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "auto" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 # if device in ("cuda", "mps") else torch.float32

    print(f"  Loading {model_path} (device={device}, dtype={dtype})",
          file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    return model, tokenizer, device


# --------------------------------------------------------------------------- #
#  Dataset loading
# --------------------------------------------------------------------------- #


@dataclass
class TranslationSample:
    """A single source/target translation pair."""

    source: str
    target: str
    source_lang: str
    target_lang: str
    sample_id: str


_FLORES_ALIASES: dict[str, str] = {
    "zho_Hans": "cmn_Hans",
    "zho_Hant": "cmn_Hant",
    "zh_Hans": "cmn_Hans",
    "zh_Hant": "cmn_Hant",
    "zh": "cmn_Hans",
    "fra": "fra_Latn",
    "eng": "eng_Latn",
    "deu": "deu_Latn",
    "spa": "spa_Latn",
    "jpn": "jpn_Jpan",
    "kor": "kor_Hang",
    "rus": "rus_Cyrl",
    "ara": "arb_Arab",
    "hin": "hin_Deva",
    "por": "por_Latn",
    "ita": "ita_Latn",
    "nld": "nld_Latn",
    "tur": "tur_Latn",
    "vie": "vie_Latn",
    "tha": "tha_Thai",
}


def _resolve_flores_lang(code: str) -> str:
    """Map common aliases to FLORES+ ISO 639-3 config names."""
    return _FLORES_ALIASES.get(code, code)


def load_flores(
    src_lang: str,
    tgt_lang: str,
    n_shot: int,
    n_samples: int,
    split: str = "devtest",
) -> tuple[list[TranslationSample], list[TranslationSample]]:
    """Load FLORES+ parallel data for few-shot examples and test samples.

    Returns ``(few_shot_examples, test_samples)``.
    """
    from datasets import load_dataset

    src_lang = _resolve_flores_lang(src_lang)
    tgt_lang = _resolve_flores_lang(tgt_lang)

    src_ds = load_dataset("openlanguagedata/flores_plus", src_lang, split=split)
    tgt_ds = load_dataset("openlanguagedata/flores_plus", tgt_lang, split=split)

    # Align by id
    src_by_id = {r["id"]: r["text"] for r in src_ds}
    tgt_by_id = {r["id"]: r["text"] for r in tgt_ds}
    common_ids = sorted(set(src_by_id) & set(tgt_by_id))

    all_samples = [
        TranslationSample(
            source=src_by_id[i],
            target=tgt_by_id[i],
            source_lang=src_lang,
            target_lang=tgt_lang,
            sample_id=str(i),
        )
        for i in common_ids
    ]

    n_samples = len(all_samples) - n_shot if n_samples == -1 else n_samples

    few_shot = all_samples[:n_shot]
    test = all_samples[n_shot : n_shot + n_samples]
    return few_shot, test


def load_wmt24pp(
    tgt_lang: str,
    n_shot: int,
    n_samples: int,
    same_lang: bool = False,
    src_lang: str | None = None,
) -> tuple[list[TranslationSample], list[TranslationSample]]:
    """Load WMT24++ data for few-shot examples and test samples.

    ``tgt_lang`` is the locale suffix, e.g. ``zh_CN``, ``fr_FR``.

    Modes:
    - **en→X** (default): source is English, target is ``tgt_lang`` text.
    - **Same-lang X→X** (``same_lang=True``): source and target are both
      ``tgt_lang`` text.
    - **Cross-lingual X→Y** (``src_lang`` set, not "en", != ``tgt_lang``):
      loads ``en-{src_lang}`` and ``en-{tgt_lang}`` configs, matches by
      ``segment_id``, and uses src text as source, tgt text as target.
    """
    from datasets import load_dataset

    if src_lang is not None and src_lang != "en" and src_lang != tgt_lang:
        src_config = f"en-{src_lang}"
        tgt_config = f"en-{tgt_lang}"

        src_ds = load_dataset("google/wmt24pp", src_config, split="train")
        tgt_ds = load_dataset("google/wmt24pp", tgt_config, split="train")

        src_rows = [r for r in src_ds if not r.get("is_bad_source", False)]
        tgt_rows = [r for r in tgt_ds if not r.get("is_bad_source", False)]

        src_map = {str(r.get("segment_id", i)): r["target"] for i, r in enumerate(src_rows)}
        tgt_map = {str(r.get("segment_id", i)): r["target"] for i, r in enumerate(tgt_rows)}

        common_ids = [sid for sid in src_map if sid in tgt_map]

        all_samples = [
            TranslationSample(
                source=src_map[sid],
                target=tgt_map[sid],
                source_lang=src_lang,
                target_lang=tgt_lang,
                sample_id=sid,
            )
            for sid in common_ids
        ]
    else:
        config = f"en-{tgt_lang}"
        ds = load_dataset("google/wmt24pp", config, split="train")

        rows = [r for r in ds if not r.get("is_bad_source", False)]

        all_samples = [
            TranslationSample(
                source=r["target"] if same_lang else r["source"],
                target=r["target"],
                source_lang=tgt_lang if same_lang else "en",
                target_lang=tgt_lang,
                sample_id=str(r.get("segment_id", idx)),
            )
            for idx, r in enumerate(rows)
        ]

    n_samples = len(all_samples) - n_shot if n_samples == -1 else n_samples

    few_shot = all_samples[:n_shot]
    test = all_samples[n_shot : n_shot + n_samples]

    return few_shot, test


# --------------------------------------------------------------------------- #
#  Language names and prompt construction
# --------------------------------------------------------------------------- #

# Human-readable language names for prompt formatting
_LANG_NAMES: dict[str, str] = {
    "fra_Latn": "Français",
    "eng_Latn": "English",
    "cmn_Hans": "中文",
    "cmn_Hant": "中文",
    "zho_Hans": "中文",
    "zho_Hant": "中文",
    "deu_Latn": "Deutsch",
    "spa_Latn": "Español",
    "jpn_Jpan": "日本語",
    "kor_Hang": "한국어",
    "rus_Cyrl": "Русский",
    "ara_Arab": "العربية",
    "hin_Deva": "हिन्दी",
    "por_Latn": "Português",
    "ita_Latn": "Italiano",
    "nld_Latn": "Nederlands",
    "tur_Latn": "Türkçe",
    "vie_Latn": "Tiếng Việt",
    "tha_Thai": "ไทย",
    "fra": "Français",
    "eng": "English",
    "zho": "中文",
    "de": "Deutsch",
    "es": "Español",
    "ja": "日本語",
    "ko": "한국어",
    "ru": "Русский",
    "ar": "العربية",
    "hi": "हिन्दी",
    "pt": "Português",
    "it": "Italiano",
    "nl": "Nederlands",
    "tr": "Türkçe",
    "vi": "Tiếng Việt",
    "th": "ไทย",
    "en": "English",
    "fr_FR": "Français",
    "zh_CN": "中文",
    "zh_TW": "中文",
    "de_DE": "Deutsch",
    "es_MX": "Español",
    "ja_JP": "日本語",
    "ko_KR": "한국어",
    "ru_RU": "Русский",
    "ar_EG": "العربية",
    "ar_SA": "العربية",
    "hi_IN": "हिन्दी",
    "pt_BR": "Português",
    "pt_PT": "Português",
    "it_IT": "Italiano",
    "nl_NL": "Nederlands",
    "tr_TR": "Türkçe",
    "vi_VN": "Tiếng Việt",
    "th_TH": "ไทย",
}


def lang_name(code: str) -> str:
    return Language.get(code).display_name(code)
    # return _LANG_NAMES.get(code, code)
    # lang = _LANG_NAMES.get(code, code)

    # if lang == code:
    #     lang = Language.get(code).display_name(code)

    # return lang
