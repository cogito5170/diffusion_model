# Run report: run_mock-demo

- **Final state:** DONE
- **Stop reason:** curator_recommended_stop: no remaining deficit in dimensions the engine input can change (technical_quality mean is 5.0)
- **Final candidate:** cand_mock-demo-i0-s0
- **Decision:** approved by scripts/mock_e2e.sh (script)
- **Engine:** mock / mock-procedural-v1

## Brief

**Original intent (verbatim, user-authored):**

> A fine-art editorial image about identity dissolving into material. A human presence that exists only as a reflection or contour on liquid chrome or brushed steel, in a cold industrial space under hard directional light. It should feel like the metal remembers a person who is no longer there: sculptural curvature, reflective titanium, quiet tension between the body and the surface. No faces, no robots, no sci-fi.

**System interpretation** (by offline-stub, offline_stub): Offline keyword interpretation of the brief: engineered permanence; facelessness; identity as surface; industrial endurance; industrial perfection; industrial space; instability and flux; reflection and self-recognition; the body as material.

## Concept

**Thesis:** A perfectly reflective form returns the whole industrial room to the viewer but withholds the face: identity appears only as the distortion it causes in a surface.

**Visual hypothesis:** A single liquid-chrome sculptural mass, implied human contour, hard key light; the reflection carries more information than the object.

## Iterations

### Iteration 0 (ad_mock-demo-v1)
- requested 4, succeeded 4, best weighted 4.15, mean 4.0
- **Experiment plan_mock-demo-i1 (for iteration 1)**
  - What failed: material_plausibility: batch mean 4.00/5 (largest weighted deficit)
  - What will change: `/materials/0/imperfections`: [] -> ['fine micro-scratches']
  - What remains fixed: avoidance_constraints, composition, concept_thesis, engine and model, engine params, lighting, mood, output size, palette, seeds per slot, semantic_constraints, subject, visual_hypothesis
  - Why it should help: Perfect surfaces read as CG; sparse micro-scratches are a physical cue of real metal.
  - Improvement means: mean(material_plausibility) > 4.0

### Iteration 1 (ad_mock-demo-v2)
- requested 4, succeeded 4, best weighted 4.15, mean 4.0
- previous plan criterion `mean(material_plausibility) > 4.0`: measured 4.0 (baseline 4.0) -> **not met**

## Candidates

| candidate | iter | status | QA | weighted | critic | runtime s | image |
|---|---|---|---|---|---|---|---|
| cand_mock-demo-i0-s0 | 0 | succeeded | True | 4.15 | heuristic | 0.385 | images/cand_mock-demo-i0-s0/a1/image_0.png |
| cand_mock-demo-i0-s1 | 0 | succeeded | True | 4.00 | heuristic | 0.1461 | images/cand_mock-demo-i0-s1/a1/image_0.png |
| cand_mock-demo-i0-s2 | 0 | succeeded | True | 3.85 | heuristic | 0.3595 | images/cand_mock-demo-i0-s2/a1/image_0.png |
| cand_mock-demo-i0-s3 | 0 | succeeded | True | 4.00 | heuristic | 0.2458 | images/cand_mock-demo-i0-s3/a1/image_0.png |
| cand_mock-demo-i1-s0 | 1 | succeeded | True | 4.15 | heuristic | 0.1551 | images/cand_mock-demo-i1-s0/a1/image_0.png |
| cand_mock-demo-i1-s1 | 1 | succeeded | True | 4.00 | heuristic | 0.3706 | images/cand_mock-demo-i1-s1/a1/image_0.png |
| cand_mock-demo-i1-s2 | 1 | succeeded | True | 3.85 | heuristic | 0.1852 | images/cand_mock-demo-i1-s2/a1/image_0.png |
| cand_mock-demo-i1-s3 | 1 | succeeded | True | 4.00 | heuristic | 0.1846 | images/cand_mock-demo-i1-s3/a1/image_0.png |

## Budget ledger

- images (measured): 8 / 12
- generation attempts (measured): 8 / 24
- engine seconds (measured wall-clock): 2.0202 / 600
- cost USD (**estimated**, basis: mock engine: no compute or API cost): 0.0 / 0.0

## Evidence quality

All ratings come from the offline HEURISTIC critic: pixel-statistic proxies with low confidence. They are not aesthetic judgements and must not be cited as evidence of image quality.

## Integrity

All artifacts and images re-hashed and re-validated: OK


Contact sheet: `report/contact_sheet.png`
