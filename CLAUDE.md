# atelier: instructions for the orchestrating Claude Code session

This repository is an agent-orchestrated aesthetic image pipeline. **The main Claude Code session is the
orchestrator.** Python (`python -m atelier`) owns the deterministic parts: state machine, validation, prompt
compilation, generation, QA, ranking, budget and records. Subagents in `.claude/agents/` are used only for
judgement steps.

## Ground rules
- Inspect before editing; do not touch unrelated files.
- Never install packages, download model weights, enable `allow_download`, or start a live (non-mock) run
  without the user's explicit approval of the expected cost/resources (see `docs/LIVE_GENERATION.md`).
- Never fabricate results. Report statuses, errors and numbers exactly as recorded in `runs/<run>/`.
- **Never approve a final candidate on the user's behalf.** Show `report/review.md` and `report/contact_sheet.png`
  and ask. Use `--scripted` only for demos/CI, never to stand in for a person.
- Do not delegate deterministic work (validation, compilation, ranking, reports) to subagents.

## Running the offline test suite (always first)
    python3 -m unittest discover -s tests -t .

## Driving a run with Claude Code subagents
1. `python3 -m atelier doctor` to check the environment.
2. `python3 -m atelier run --brief <brief.json> --config configs/mock_claude_code.json --run-id run_<name>`
   (mock engine, real agents: no GPU or model cost; agent calls do cost model tokens).
3. Loop until the run is DONE, FAILED, CANCELLED or waiting for a human:
   - `python3 -m atelier status run_<name>` prints `WAITING for agent '<agent>'` plus one or more task files.
   - For each task file, invoke the named subagent with: *"Complete the atelier task at `<task path>`. Read the
     task file first."* Independent `art-critic` tasks may run in parallel.
   - `python3 -m atelier resume run_<name>`. Invalid output is rejected and re-issued with errors attached,
     up to `agents.max_rejections_per_step`; after that the run fails visibly.
4. At `AWAITING_HUMAN_APPROVAL`: present the review sheet, then run the user's decision:
   `python3 -m atelier approve run_<name> --candidate <id> --reviewer <user>` or `... reject ...`.
5. `python3 -m atelier verify run_<name>` and report `runs/<run>/report/report.md`.

Agents and steps: concept-researcher (research) → concept-generator (concepts) → art-director
(art_direction_draft) → material-specialist (material_spec) → [deterministic compile + generate + QA] →
art-critic (critique, one task per candidate) → curator (curation: stop or one bounded experiment).
prompt-compiler and image-operator are optional helpers (read-only review / CLI operation).
