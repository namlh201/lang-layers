"""Load a model and generate text from a prompt.

Auto-detects the best available GPU and dispatches accordingly:

* **Apple Silicon (MLX)**: uses ``vllm-mlx`` or ``mlx_lm`` / ``mlx_vlm``.
* **NVIDIA CUDA**: uses ``vllm`` or ``transformers``.
* **Apple Silicon (MPS)** / **CPU**: uses ``transformers``.

Usage::

    # Apple Silicon (auto-detected)
    python generate.py --model mlx-community/gemma-4-12B-it-4bit \\
        --prompt "The capital of France is"

    # NVIDIA CUDA (auto-detected)
    python generate.py --model meta-llama/Llama-3.2-3B-Instruct \\
        --prompt "The capital of France is"

    # Force a specific device
    python generate.py --model <repo> --prompt "..." --device cuda
    python generate.py --model <repo> --prompt "..." --device cpu

    # Disable vllm/vllm-mlx
    python generate.py --model <repo> --prompt "..." --no-vllm

    # With an image (VLM only, MLX backend):
    python generate.py --model mlx-community/gemma-4-12B-it-4bit \\
        --prompt "Describe this image" --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from utils.load import detect_gpu, load_model

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a model and generate text (auto GPU detection)."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-12B-it-4bit",
        help="Model path or HF repo",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument(
        "--image", default=None, help="Image path or URL (VLM models only)"
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["lm", "vlm"],
        default=None,
        help="Force MLX sub-backend (auto-detected by default)",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=True,
        help="Use vllm-mlx (Apple) or vllm (NVIDIA) for generation "
        "(default: True)",
    )
    parser.add_argument(
        "--no-vllm",
        dest="use_vllm",
        action="store_false",
        help="Disable vllm/vllm-mlx, use mlx_lm/mlx_vlm or "
        "transformers directly",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mlx", "mps", "cpu"],
        help="Force device (auto-detected by default)",
    )
    args = parser.parse_args()

    gpu = args.device or detect_gpu()
    print(f"[device: {gpu}]  model: {args.model}", file=sys.stderr)

    model, tokenizer, backend_kind, vllm_wrapper = load_model(
        args.model,
        backend=args.backend,
        use_vllm=args.use_vllm,
        device=gpu,
    )
    is_vlm = backend_kind in ("mlx-vlm", "vlm")

    if args.image and not is_vlm:
        print(
            "Image provided but backend is text-only — ignoring image.",
            file=sys.stderr,
        )

    from utils.generation import generate_translation

    try:
        text = generate_translation(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            is_vlm=is_vlm,
            vllm_wrapper=vllm_wrapper,
            backend_kind=backend_kind,
        )
    except Exception as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.verbose:
        print(text)


if __name__ == "__main__":
    main()
