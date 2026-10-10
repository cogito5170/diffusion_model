#!/usr/bin/env bash
# Reproducible offline end-to-end run: mock engine + offline agent stubs + scripted approval.
# No GPU, network, API keys or model downloads. Images are procedural test renders, not model output.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS="${RUNS:-runs}"
RUN_ID="${RUN_ID:-run_mock-demo}"
if [ -e "$RUNS/$RUN_ID" ]; then
  echo "$RUNS/$RUN_ID already exists; choose another RUN_ID or remove it yourself" >&2
  exit 1
fi
python3 -m atelier --runs-dir "$RUNS" run \
  --brief examples/briefs/metallic_identity.json --config configs/mock_offline.json --run-id "$RUN_ID"
CAND="$(python3 -m atelier --runs-dir "$RUNS" shortlist "$RUN_ID" | head -1)"
python3 -m atelier --runs-dir "$RUNS" approve "$RUN_ID" --candidate "$CAND" --scripted \
  --reviewer "scripts/mock_e2e.sh" --note "scripted approval of the top shortlist entry for the reproducible demo; not an aesthetic judgement"
python3 -m atelier --runs-dir "$RUNS" verify "$RUN_ID"
echo "report: $RUNS/$RUN_ID/report/report.md"
