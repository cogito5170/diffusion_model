# Limitations, assumptions and unresolved decisions

## Verified vs. not verified (as of this commit)
| item | status |
|---|---|
| Offline test suite (mock engine, offline agents) | **executed, passing** (see README for the exact command) |
| Reproducible mock end-to-end run (`scripts/mock_e2e.sh`) | **executed**; images and decisions byte-identical across two runs |
| diffusers adapter code path | **unit-tested with an injected fake pipeline only** |
| Real diffusers inference, sd-turbo | executed by the user on an Apple-silicon Mac (MPS, float16): about 2.4–4.1 s per image; first image 10–19 s including model load |
| Real diffusers inference, SDXL | first attempts on the user's 16 GB Mac: a missing-file download bug (fixed) and then a Python crash during high-quality generation (cause not yet diagnosed; mitigations: isolated worker, VAE tiling, ~0.8 MP default size). **No successful SDXL image yet** |
| Mac installer, studio | tested here only with the mock engine and syntax checks; **not yet run on a clean Mac** |
| Claude Code subagents producing artifacts | protocol tested end to end with stub answers; **no real agent run recorded yet** |
| Image-quality improvement from the agent workflow | **unverified**; no human ratings collected |

## Engine limitations
- **~75-token CLIP prompt budget.** With realistic art directions the compiler drops palette, composition and mood
  (recorded per request). Consequence observed in the mock demo: iterations can only influence dimensions whose
  sections reach the prompt. The curator stopped after one refinement because material and technical
  ratings were already at the ceiling. Options: long-prompt chunking/weighting (e.g. Compel), an engine with a
  longer text encoder, or the art-director front-loading the visual priorities.
- Turbo models run at guidance 0, so **negative prompts have no effect**. Avoidance constraints are enforced only
  by critique.
- Text-to-image only: `max_reference_images = 0`. Required reference images fail compilation; optional ones are
  dropped with a warning. img2img/IP-Adapter is future work.
- Timeouts: the runner abandons a non-cooperative engine thread (daemon) and signals cancellation. The diffusers
  adapter cooperates via `callback_on_step_end`, but a single long step cannot be pre-empted.
- Token counts are estimated without the real tokenizer.

## Evaluation limitations
- The offline **heuristic critic** measures pixel statistics (highlight fraction, saturation, occupancy, hue spread,
  palette distance, within-batch feature distance). It is a test fixture, not an aesthetic judge. Reports say so.
- The art-critic agent's ratings are model opinions. They need calibration against blinded human ratings
  (`experiments/`) before anyone uses them to claim improvement.
- Integer 1–5 ratings are coarse; ties are common and broken by candidate id.
- The diversity feature (luma thumbnail + colour histograms) is cheap and weak; it catches near-duplicates, not
  semantic sameness. A CLIP/DINO embedding would be better but adds a model download.
- The default weights (0.25/0.20/0.15/0.15/0.15/0.10) are hypotheses.

## Schema limitations
- `materials[].type` covers metals only. Glass, acrylic, water and concrete (e.g. in
  `docs/prompts/memory_of_space.md`) must be described in subject/environment text, so they are not parameterised and
  the plausibility checks do not cover them. A `material_spec.v2` with non-metal classes is a candidate change.

## Assumptions
- One concept is developed per run (the recommended one). Multi-concept batches would add diversity at higher cost.
- Seeds are kept fixed per slot across refinements, to isolate the planned change.
- The brief's domain enum currently has only `metallic_fine_art_editorial`; new domains need a schema version bump.
- Cost for local inference is recorded as 0 USD (estimated), with electricity and hardware unmetered.

## Unresolved decisions (need an owner)
1. **Model license**: turbo models carry Stability AI licenses with commercial restrictions. Choose a model whose
   license fits the intended use before any publication.
2. **Live engine choice beyond sd-turbo**: local GPU vs. a paid API. Adding a provider is deliberately deferred
   until the first integration is tested live.
3. **Prompt budget strategy**: chunking/weighting vs. stricter art-direction length limits.
4. **Human rating protocol**: number of raters, rater instructions, preference ranking vs. absolute ratings.
5. **Whether a human "reject" may trigger a new bounded iteration** (currently it ends the run).
6. **Where research may come from** (web access for concept-researcher, citation policy).
