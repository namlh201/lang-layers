from __future__ import annotations

from utils.load import TranslationSample, lang_name

def build_few_shot_prompt(
    examples: list[TranslationSample],
    test: TranslationSample,
) -> str:
    """Build a few-shot translation prompt.

    Format::

        <src_lang>: "<sentence1>"
        <tgt_lang>: "<translation1>"
        <src_lang>: "<sentence2>"
        <tgt_lang>: "<translation2>"
        ...
        <src_lang>: "<test_source>"
        <tgt_lang>: "
    """
    lines: list[str] = []
    for ex in examples:
        src_name = lang_name(ex.source_lang)
        tgt_name = lang_name(ex.target_lang)
        lines.append(f'{src_name}: "{ex.source}"')
        lines.append(f'{tgt_name}: "{ex.target}"')

    test_src_name = lang_name(test.source_lang)
    test_tgt_name = lang_name(test.target_lang)
    lines.append(f'{test_src_name}: "{test.source}"')
    lines.append(f'{test_tgt_name}: "')
    return "\n".join(lines)

# --------------------------------------------------------------------------- #
#  Generation
# --------------------------------------------------------------------------- #


def generate_translation(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    is_vlm: bool = False,
    vllm_wrapper=None,
    backend_kind: str = "mlx-lm",
    stop: list[str] | None = None,
) -> str:
    """Generate a single translation from the prompt.

    Dispatches to the appropriate generation backend:

    * ``backend_kind="vllm"``: NVIDIA vLLM (fastest on CUDA).
    * ``backend_kind="mlx-lm"`` / ``"mlx-vlm"`` with ``vllm_wrapper``:
      vllm-mlx (fastest on Apple Silicon).
    * ``backend_kind="mlx-lm"`` / ``"mlx-vlm"`` without ``vllm_wrapper``:
      mlx_lm / mlx_vlm directly.
    * ``backend_kind="torch"``: HuggingFace ``model.generate()``.

    Args:
        stop: Stop sequences that halt generation (default: ``["\n"]``).
    """
    return generate_translations(
        model,
        tokenizer,
        [prompt],
        max_tokens=max_tokens,
        temperature=temperature,
        is_vlm=is_vlm,
        vllm_wrapper=vllm_wrapper,
        backend_kind=backend_kind,
        stop=stop,
    )[0]


def generate_translations(
    model,
    tokenizer,
    prompts: list[str],
    max_tokens: int = 128,
    temperature: float = 0.0,
    is_vlm: bool = False,
    vllm_wrapper=None,
    backend_kind: str = "mlx-lm",
    stop: list[str] | None = None,
) -> list[str]:
    """Generate translations for multiple prompts (batch).

    For vLLM (NVIDIA CUDA), all prompts are sent in a single
    ``generate()`` call so the engine can batch them efficiently.
    For all other backends, prompts are processed sequentially via
    :func:`generate_translation`'s per-backend logic.

    Args:
        prompts: List of prompt strings.
        stop: Stop sequences that halt generation (default: ``["\\n"]``).

    Returns:
        List of generated translation strings, same length and order as
        *prompts*.
    """
    if stop is None:
        stop = ["\n"]

    if len(prompts) == 0:
        return []

    # --- vLLM (NVIDIA CUDA) — native batch ---
    if backend_kind == "vllm" and vllm_wrapper is not None:
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        outputs = vllm_wrapper.generate(prompts, sampling)
        return [o.outputs[0].text.strip() for o in outputs]

    # --- vllm-mlx (Apple Silicon) — native batch ---
    if vllm_wrapper is not None and backend_kind in ("mlx-lm", "mlx-vlm"):
        fmt_prompts: list[str] = []
        for p in prompts:
            if is_vlm:
                from mlx_vlm.prompt_utils import apply_chat_template

                config = getattr(model, "config", None)
                fmt_prompts.append(
                    apply_chat_template(tokenizer, config, p, num_images=0)
                )
            else:
                fmt_prompts.append(p)
        outputs = [
            vllm_wrapper.generate(
                prompt=fp,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            for fp in fmt_prompts
        ]
        return [o.text.strip() for o in outputs]

    # --- mlx_vlm direct (Apple Silicon, no vllm-mlx) ---
    if backend_kind == "mlx-vlm":
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        config = getattr(model, "config", None)
        results: list[str] = []
        for p in prompts:
            fmt_prompt = apply_chat_template(
                tokenizer, config, p, num_images=0
            )
            result = generate(
                model,
                tokenizer,
                fmt_prompt,
                image=None,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            text = result.text
            for s in stop:
                text = text.split(s)[0]
            results.append(text.strip())
        return results

    # --- mlx_lm direct (Apple Silicon, no vllm-mlx) ---
    if backend_kind == "mlx-lm":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        results: list[str] = []
        for p in prompts:
            text = generate(
                model,
                tokenizer,
                p,
                max_tokens=max_tokens,
                sampler=sampler,
                stop=stop,
                verbose=False,
            )
            results.append(text.strip())
        return results

    # --- torch / transformers (CUDA, MPS, CPU) ---
    import torch

    device = next(model.parameters()).device
    stop_token_ids = [
        tid
        for s in stop
        for tid in tokenizer.encode(s, add_special_tokens=False)
    ]
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnStrings(StoppingCriteria):
        def __init__(self, stop_ids: list[int]):
            self.stop_ids = set(stop_ids)

        def __call__(self, ids, scores, **kwargs):
            return ids[0, -1].item() in self.stop_ids

    stopping = StoppingCriteriaList([_StopOnStrings(stop_token_ids)])
    gen_kwargs: dict = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "stopping_criteria": stopping,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    results: list[str] = []
    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, **gen_kwargs)
            new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
            results.append(
                tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            )
    return results
