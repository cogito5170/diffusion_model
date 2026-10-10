# Example run (mock)

`run_mock-demo/` is the full output of `RUNS=docs/examples scripts/mock_e2e.sh`, committed so the artifacts can be
inspected without running anything: state, manifest, event log, every validated artifact, images and report.

- Images come from the **procedural mock engine**. They are test renders, not model output.
- Critiques come from the **heuristic critic** (pixel-statistic proxies, confidence 0.15), not from the art-critic agent.
- The final approval was **scripted** (`reviewer.kind = "script"`), not a human decision.
- Rerunning the script reproduces byte-identical images and identical decisions. Timestamps and runtimes differ.

Check integrity: `python3 -m atelier verify docs/examples/run_mock-demo`
