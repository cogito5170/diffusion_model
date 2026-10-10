# Optional live generation (clearly separated from the offline pipeline)

Nothing in this repository installs packages, downloads weights or runs a real model on its own. The steps below are
deliberate, manual opt-ins.

## What the live engine is
`atelier/engines/diffusers_local.py`: local inference with Hugging Face `diffusers`.

| profile | steps | CFG / negative prompt | approx. download* | license (verify on the model card) |
|---|---|---|---|---|
| `stabilityai/sd-turbo` (default) | 1–4 | guidance 0 → negative prompt has **no effect** | ~2.5–5 GB | Stability AI license; commercial use restricted, verify |
| `stabilityai/sdxl-turbo` | 1–4 | guidance 0 → no effect | ~7 GB | as above |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | 5–50 | 1–15 → negative prompts work | ~2–4 GB | CreativeML OpenRAIL-M |

\* Approximations, **not measured** in the environment this was built in (no GPU, packages not installed).

All profiles use a CLIP text encoder: about 75 usable prompt tokens. The compiler drops lower-priority
sections and records each drop in `generation_request.compilation.dropped`.

## Resource expectations (estimates, unverified)
- **CPU only** (e.g. 4 cores, 16 GB RAM): sd-turbo at 512 px, 2 steps is plausibly tens of seconds per image. A
  4-image batch × 3 iterations could take 5–20 minutes. Set `generation.timeout_s` and `budget.max_engine_seconds`
  accordingly.
- **NVIDIA GPU (≥ 6–8 GB VRAM)**: typically well under a few seconds per image for turbo models.
- Monetary cost: no API charge. `cost_usd_per_image` defaults to 0.0 and the report labels cost as *estimated*,
  with electricity and hardware unmetered. If you add a paid engine later, set a real per-image estimate so
  `max_cost_usd` is enforced.

## Procedure
1. Offline suite first: `python3 -m unittest discover -s tests -t .` (must pass).
2. Inspect the machine: `python3 -m atelier doctor`.
3. Decide and approve the cost: disk for weights, time per image, license terms.
4. Install deliberately: `pip install -r requirements-live.txt` (pick the right torch build for your CUDA/CPU).
5. Fetch weights deliberately, e.g. `huggingface-cli download stabilityai/sd-turbo` (or set
   `"allow_download": true` in a *copy* of `configs/live_sd_turbo.json`; the default is false).
6. Check: `python3 -m atelier caps --config configs/live_sd_turbo.json`. Expect `"available": true`.
7. One-image smoke test (disabled by default):
   `ATELIER_LIVE_TEST=1 python3 -m unittest tests.test_live_diffusers`
8. A real run with Claude Code agents (see `CLAUDE.md`):
   `python3 -m atelier run --brief examples/briefs/metallic_identity.json --config configs/live_sd_turbo.json --run-id run_live-01`

Budget limits in `configs/live_sd_turbo.json`: 4 images/batch, ≤ 2 refinements, ≤ 12 images, ≤ 16 attempts,
300 s timeout per image, ≤ 1800 engine seconds. Lower them for a first trial.
