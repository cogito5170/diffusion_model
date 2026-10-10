# atelier: agent-orchestrated aesthetic image generation

> **처음 쓰시나요? 코딩을 몰라도 됩니다 → [시작하기.md](시작하기.md)** (맥 설치 한 줄 + 바탕화면 아이콘 · 미드저니식 '이미지 스튜디오' · [PDF 설명서](docs/manual/이미지_스튜디오_설명서.pdf))

A small, modular system that turns a philosophical/artistic intention into a **validated, engine-independent
visual specification**, generates candidate images with a **pretrained** model behind an adapter, evaluates them,
and improves them through a **bounded** feedback loop. A **human approves** the final selection.

Initial domain: metallic fine-art editorial imagery (liquid chrome, brushed steel, reflective titanium, sculptural
curvature, industrial space, hard directional light, tension between human identity and material surface).
No model is trained here.

- Python orchestrates, validates, evaluates and records: `atelier/`
- Claude Code subagents do the judgement steps: `.claude/agents/` (8 agents, narrow tools)
- Every hand-off is a versioned JSON Schema contract: `atelier/schemas/*.v1.json`
- One real engine adapter: local `diffusers` (`stabilityai/sd-turbo` default). Plus a mock test double.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Live generation: [docs/LIVE_GENERATION.md](docs/LIVE_GENERATION.md) ·
Limitations: [docs/LIMITATIONS.md](docs/LIMITATIONS.md) · Orchestrating with Claude Code: [CLAUDE.md](CLAUDE.md) ·
Experiment: [experiments/](experiments/) · Example prompt work: [docs/prompts/memory_of_space.md](docs/prompts/memory_of_space.md)

## Setup
Python ≥ 3.10 (developed and tested on 3.13). Offline dependencies only: `jsonschema>=4.18`, `pillow`, `numpy`.

```bash
pip install -r requirements.txt          # offline pipeline + tests; no GPU, no network at runtime
python3 -m atelier doctor                # what this machine can do (no network calls)
```

## Run the offline tests
```bash
python3 -m unittest discover -s tests -t .      # also works with: python -m pytest tests
```
No GPU, API key or network needed. `tests/test_live_diffusers.py` is skipped unless `ATELIER_LIVE_TEST=1`.

## Reproducible mock end-to-end run
```bash
scripts/mock_e2e.sh                       # RUNS=... RUN_ID=... to change location
```
It uses the mock engine (procedural renders, **not** model output), offline agent stubs, the heuristic critic
(pixel proxies, **not** aesthetic judgement) and a *scripted* approval. A copy of one run is in
[`docs/examples/run_mock-demo/`](docs/examples/run_mock-demo/report/report.md).

## Run step by step
```bash
python3 -m atelier run --brief examples/briefs/metallic_identity.json --config configs/mock_offline.json --run-id run_demo
python3 -m atelier status run_demo
python3 -m atelier approve run_demo --candidate <id from report/review.md> --reviewer <you>
python3 -m atelier verify run_demo
```
With real Claude Code subagents (mock engine, so no image cost): use `configs/mock_claude_code.json` and follow
[CLAUDE.md](CLAUDE.md). The run pauses with a task file for each agent step and continues with `atelier resume`.

Other commands: `resume`, `cancel` (takes effect at the next candidate/step boundary), `report`, `shortlist`,
`caps --config`, `compile --config --brief --art-direction` (dry-run prompt compilation), `validate <file>`.

## Configuration (defaults in `configs/*.json`)
| setting | default | bound |
|---|---|---|
| batch size | 4 | ≤ 8 |
| refinement iterations after the initial batch | 2 | ≤ 5 (schema ceiling) |
| major variables changed per iteration | 1 | ≤ 3 |
| weights (intent, material, composition, coherence, originality, technical) | .25 .20 .15 .15 .15 .10 | must sum to 1; hypotheses |
| budget | ≤ 12 images, ≤ 24 attempts, ≤ $0 estimated, ≤ 600 engine-s (mock) | hard limits |
| per-image timeout / retries | 60 s / 1 (timeouts not retried) | ≤ 3600 s / ≤ 3 |
| human approval | required | `require_human_approval: false` records `config_auto` |

## Layout
```
atelier/                orchestrator, state machine, budget, compiler, evaluator, selector, store, engines/, schemas/
.claude/agents/         concept-researcher, concept-generator, art-director, material-specialist,
                        prompt-compiler, image-operator, art-critic, curator
configs/                mock_offline, mock_claude_code, live_sd_turbo
examples/               briefs/ (metallic_identity, memory_of_space), artifacts/ (one valid example per contract)
experiments/            run_comparison.py (A/B/C + blind rating packet + analysis), REPORT_TEMPLATE.md
tests/                  offline unit + integration tests; optional live test
docs/                   ARCHITECTURE, LIVE_GENERATION, LIMITATIONS, prompts/, examples/
```
