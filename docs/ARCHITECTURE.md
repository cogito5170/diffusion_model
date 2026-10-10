# Architecture

## Principles
- **Python decides; agents judge.** Everything deterministic (state, validation, compilation, generation, QA,
  ranking, budget, records, reports) is code. Claude Code subagents handle only judgement: research, concepts,
  art direction, materials, critique, curation.
- **Contracts, not conversations.** Every hand-off is a JSON artifact with `schema_version` and an id, validated
  against `atelier/schemas/*.v1.json` (`additionalProperties: false` everywhere). Invalid output is rejected and
  re-requested with the errors attached. Nothing is coerced.
- **Engine-independent spec.** The art direction describes intent, composition, materials and lighting in
  physical/semantic terms. Only `prompt_compiler.py` knows about prompts, token limits or CFG.
- **Honest records.** Results store actual paths, hashes, model ids, seeds, runtimes, attempts and structured
  errors. Mock and heuristic outputs are labelled as such everywhere.

## Workflow (state machine, `atelier/state_machine.py`)

```
CREATED ─► BRIEF_ACCEPTED ─► RESEARCHED ─► CONCEPTS_READY ─► ART_DIRECTION_DRAFTED ─► SPEC_VALIDATED
 (brief      (intent hash    (concept-      (concept-        (art-director)           (material-specialist
  frozen)     recorded)       researcher)    generator)                                + merge + checks)
                                                                                          │
   ┌──────────────────────────── bounded iteration (new art-direction version) ◄────────┐  ▼
   │                                                                                    │ PROMPTS_COMPILED
   ▼                                                                                    │  (prompt_compiler)
SPEC_VALIDATED ─► PROMPTS_COMPILED ─► GENERATED ─► QA_DONE ─► CRITIQUED ─► SELECTED ────┤
                   (engine runner)    (evaluator)  (art-critic) (selector + curator)    │
                                                                                         ▼
                                    AWAITING_HUMAN_APPROVAL ─► APPROVED / REJECTED ─► DONE (final report)
Any non-terminal state ─► FAILED | CANCELLED   (a partial report is still written)
```

A run is persisted after every unit of work (`runs/<id>/state.json`, atomic writes). `resume` continues from
the persisted state and never regenerates a candidate whose result is recorded. A run pauses (does not block)
while it waits for an agent artifact or a human decision.

## Modules
| module | responsibility |
|---|---|
| `orchestrator.py` | drives states, persists after each unit of work, pauses/resumes, approval gate, failure/cancel handling |
| `state_machine.py` | states and legal transitions |
| `schemas.py` + `schemas/` | versioned JSON Schemas (Draft 2020-12, cross-file `$ref`), strict validation |
| `artifact_store.py` | immutable artifacts by id, sha256 manifest, atomic writes, event log, `verify()` |
| `budget.py` | hard limits (images, attempts, estimated USD, engine seconds) and cancellation tokens (incl. a `CANCEL` file) |
| `prompt_compiler.py` | art direction → engine request; capability negotiation; token budgeting by priority; seeds |
| `spec_checks.py` | semantic checks beyond JSON Schema (aspect ratio, material plausibility, constraint conflicts, engine-syntax leakage) |
| `engines/base.py` | `ImageEngine`, `EngineCapabilities`, `EngineOutput`, `EngineError` |
| `engines/runner.py` | timeout, bounded retries, budget charging, malformed-output detection; always returns a valid result |
| `engines/diffusers_local.py` | **the one real adapter**: local Hugging Face diffusers (default `stabilityai/sd-turbo`), `local_files_only` by default |
| `engines/mock.py` | deterministic procedural test double with failure injection |
| `evaluator.py` | technical QA (decode, size, hash, blank, clipping, sharpness), weighted score, improvement criteria |
| `selector.py` | ranking + Maximal-Marginal-Relevance shortlist; near-duplicate demotion; diversity metrics |
| `iteration.py` | plan validation (≤ N major variables, frozen sections, stale/no-op/unreachable changes), plan application, stopping rules |
| `agents_io.py` | file-drop hand-off to Claude Code subagents; offline backend |
| `offline_agents.py` | deterministic stubs + heuristic critic (labelled, low confidence) for tests and mock runs |
| `report.py` | review sheet for the approval gate, final report (md + schema-valid json), contact sheet |

## Agent hand-off protocol
1. The orchestrator writes `tasks/<step>__<key>.task.json` (`atelier.agent_task.v1`): agent name, input file
   paths, output path, schema path, `required_values` (ids the output must use), instructions, previous errors.
2. The main Claude Code session invokes the named subagent; it writes `inbox/<step>__<key>.json`.
3. `atelier resume` validates: schema, required values, step-specific semantic checks. On failure the file
   moves to `rejected/` and the task is re-issued (bounded by `agents.max_rejections_per_step`).

| agent | tools | writes |
|---|---|---|
| concept-researcher | Read, Write, WebSearch, WebFetch | research_notes |
| concept-generator | Read, Write | concept_set |
| art-director | Read, Write | art_direction_draft |
| material-specialist | Read, Write | material_spec (merged into art_direction) |
| prompt-compiler | Read, Bash (atelier compile/caps/validate) | nothing (review only) |
| image-operator | Read, Bash (atelier CLI) | nothing (operates the CLI) |
| art-critic | Read, Write | critique (one per candidate) |
| curator | Read, Write | curation (+ iteration_plan) |

## Scoring, selection and iteration
- Weighted score = Σ wᵢ·ratingᵢ with the configured weights (default 0.25 / 0.20 / 0.15 / 0.15 / 0.15 / 0.10).
  Weights must sum to 1 and are **hypotheses**, recorded in every selection artifact.
- Shortlist: MMR with λ (default 0.7) over a cheap perceptual feature (luma thumbnail + chroma/grey histograms).
  Near-duplicates (distance < threshold) are demoted. QA failures are ineligible. A human may pick any candidate
  with an image. Overrides are recorded with reasons.
- Each refinement is an `iteration_plan` with what failed / what will change / what stays fixed / why / an
  improvement criterion. Validation rejects plans that change more than `max_major_variables_per_iteration`
  sections, touch frozen sections, cite stale values, set unattainable criteria, or **do not change the compiled
  engine input**: the orchestrator recompiles to check, so no budget is spent on variables the engine never sees.
- Stop rules (first match): no eligible candidate → target reached → max iterations (default 2, ceiling 5) →
  plateau (best score gained < `min_improvement`) → next batch unaffordable → curator recommends stop.
- Seeds are fixed per slot across iterations by default, so a refinement batch differs from its parent only
  by the planned change.
