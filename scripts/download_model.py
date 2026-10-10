"""Download only the fp16 diffusers files a model needs (not the large single-file checkpoints in the same repos).

    python scripts/download_model.py sd-turbo     # ~2.6 GB (estimate)
    python scripts/download_model.py sdxl         # ~7 GB (estimate)

Requires huggingface_hub (installed by requirements-live.txt). Sizes are estimates; check the model cards.
"""

import sys
from pathlib import Path

MODELS = {
    "sd-turbo": ("stabilityai/sd-turbo",
                 ["text_encoder/model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors",
                  "vae/diffusion_pytorch_model.fp16.safetensors"]),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0",
             ["text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
              "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
              # diffusers' completeness check also expects the repo's alternative VAE folder (~170 MB)
              "vae_1_0/diffusion_pytorch_model.fp16.safetensors"]),
}
COMMON = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "*/config.json"]


def download(key: str) -> Path:
    from huggingface_hub import snapshot_download

    repo, weights = MODELS[key]
    root = Path(snapshot_download(repo, allow_patterns=COMMON + weights))
    try:
        # Second pass with diffusers' own file selection, so the loader's completeness check passes even if the
        # repository layout differs from the list above. Already-downloaded files are not fetched again.
        from diffusers import DiffusionPipeline  # type: ignore

        root = Path(DiffusionPipeline.download(repo, variant="fp16", use_safetensors=True))
    except (ImportError, AttributeError):
        print("note: diffusers not available for the completeness pass; using the file list only")
    missing = [w for w in ["model_index.json", *weights] if not (root / w).exists()]
    size_gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
    print(f"{repo}: {root} ({size_gb:.2f} GB)")
    if missing:
        raise SystemExit(f"MISSING (the repository layout may have changed): {missing}")
    return root


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        raise SystemExit(f"usage: python scripts/download_model.py {{{'|'.join(MODELS)}}}")
    download(sys.argv[1])
