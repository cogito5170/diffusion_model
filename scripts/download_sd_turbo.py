"""Download only the files the diffusers-local engine needs for stabilityai/sd-turbo (fp16 variant).

The model repository also holds a ~5 GB single-file checkpoint and fp32 copies that this project does not use.
Expected download: roughly 2.6 GB (estimate; check the model card). Requires `pip install huggingface_hub`.

    python scripts/download_sd_turbo.py
"""

import sys

from huggingface_hub import snapshot_download

REPO = "stabilityai/sd-turbo"
PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "*/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]

if __name__ == "__main__":
    path = snapshot_download(REPO, allow_patterns=PATTERNS)
    from pathlib import Path

    root = Path(path)
    needed = ["model_index.json", "unet/diffusion_pytorch_model.fp16.safetensors",
              "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors"]
    missing = [n for n in needed if not (root / n).exists()]
    size_gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
    print(f"downloaded to {root} ({size_gb:.2f} GB)")
    if missing:
        print("MISSING (the repository layout may have changed):", missing)
        sys.exit(1)
