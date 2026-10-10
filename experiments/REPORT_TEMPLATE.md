# Experiment report: <name>

> Fill every cell. Label each number **measured**, **estimated**, **proxy** or **unverified**. Do not claim that the
> agent workflow improves quality unless the blinded human comparison supports it (CI excludes 0).

## 1. Setup
| item | value |
|---|---|
| Date / operator | |
| Brief(s) | `examples/briefs/...` (n = ) |
| Engine / model / version | (from `automatic_metrics.json → design.engine`; library versions from result `engine_metadata`) |
| Hardware | (`python -m atelier doctor`) |
| Config | `configs/...` (commit hash: ) |
| Images per condition | (default = images C generated) |
| Seeds | base_seed = ; per-slot seeds shared by A and B |
| Critic in C | agent / heuristic (from `design.critic_kinds_in_C`) |
| Raters | number, expertise, blinding procedure |

Conditions: **A** raw intent text as prompt · **B** structured art direction + compiler (C's iteration-0 spec),
no refinement · **C** full workflow (critic + curator + ≤ 2 bounded refinements).

## 2. Procedure
1. `python experiments/run_comparison.py run --brief ... --config ... --out experiments/out/<name>`
2. Raters fill `blind/ratings.csv` without opening `blind/key.json`.
3. `python experiments/run_comparison.py analyze --out experiments/out/<name>`

## 3. Results
### 3.1 Human ratings (measured, blinded)
| metric | A mean (n) | B mean (n) | C mean (n) | B−A [95% CI] | C−B [95% CI] | C−A [95% CI] |
|---|---|---|---|---|---|---|
| overall preference | | | | | | |
| intent alignment | | | | | | |
| material plausibility | | | | | | |

### 3.2 Automatic metrics
| metric | label | A | B | C |
|---|---|---|---|---|
| success rate | measured | | | |
| runtime per candidate (s) | measured | | | |
| cost per candidate (USD) | estimated | | | |
| diversity (mean pairwise feature distance) | measured proxy | | | |
| heuristic intent / material | proxy | | | |

### 3.3 Process observations (from run C)
- iterations executed / plan criteria met:
- stop reason:
- sections dropped by the compiler:
- agent output rejections:

## 4. Findings
| finding | evidence type (measured / estimated / unverified) | supported? |
|---|---|---|
| | | |

## 5. Threats to validity
- single brief / small n; images within a condition are not independent (shared prompts, seeds)
- rater expertise and fatigue; order effects (the packet is shuffled, not counterbalanced)
- critic is a model and may share biases with the art-director
- engine prompt budget limits what refinement can change

## 6. Decision / next step
