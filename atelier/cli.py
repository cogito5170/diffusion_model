"""Command-line interface: ``python -m atelier <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atelier import doctor, schemas
from atelier.config import ConfigError, load_config
from atelier.engines import create_engine
from atelier.orchestrator import Orchestrator, RunError
from atelier.state_machine import State
from atelier.util import read_json


def _run_dir(arg: str, runs_dir: str) -> Path:
    p = Path(arg)
    if (p / "state.json").exists():
        return p
    q = Path(runs_dir) / arg
    if (q / "state.json").exists():
        return q
    raise SystemExit(f"no run found at {p} or {q}")


def _summary(orch: Orchestrator) -> str:
    st = orch.state
    out = [f"run {st['run_id']}: {st['state']}  (iteration {st['iteration']}, ledger {st['ledger']})"]
    if st["waiting_for"]:
        w = st["waiting_for"]
        if w["kind"] == "agent":
            out.append(f"WAITING for agent '{w['agent']}' on step '{w['step']}' keys={w['keys']}")
            out += [f"  task: {orch.store.root / t}" for t in w["task_paths"]]
            out.append(f"  then: python -m atelier resume {orch.store.root}")
        else:
            out.append("WAITING for human approval; review sheet:")
            out += [f"  {orch.store.root / t}" for t in w["task_paths"]]
    if st["stop_reason"]:
        out.append(f"stop reason: {st['stop_reason']}")
    if st["failure"]:
        out.append(f"FAILED: {st['failure']['code']}: {st['failure']['message']}")
    if st["state"] in ("DONE", "FAILED", "CANCELLED"):
        out.append(f"report: {orch.store.root / 'report' / 'report.md'}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="atelier", description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="inspect python, packages, hardware, model cache (no network)")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("validate", help="validate a JSON artifact against its declared schema")
    s.add_argument("file")
    s.add_argument("--expect", help="required schema_version")

    s = sub.add_parser("caps", help="show engine capabilities and availability for a config")
    s.add_argument("--config", required=True)

    s = sub.add_parser("compile", help="dry-run prompt compilation for an art direction (no generation)")
    s.add_argument("--config", required=True)
    s.add_argument("--brief", required=True)
    s.add_argument("--art-direction", required=True)
    s.add_argument("--slot", type=int, default=0)

    s = sub.add_parser("run", help="create a run and advance it until it finishes or pauses")
    s.add_argument("--brief", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--run-id", required=True, help="e.g. run_demo-mock")
    s.add_argument("--stop-at", choices=["SPEC_VALIDATED"], help="stop once the art direction is validated")

    for name, hlp in (("resume", "continue a paused or interrupted run"), ("status", "show run state"),
                      ("cancel", "request cancellation"), ("verify", "re-hash and re-validate all artifacts"),
                      ("report", "regenerate the report"), ("shortlist", "print the system shortlist")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("run")
        if name == "resume":
            s.add_argument("--stop-at", choices=["SPEC_VALIDATED"], help="stop once the art direction is validated")

    s = sub.add_parser("approve", help="human approval of a final candidate")
    s.add_argument("run")
    s.add_argument("--candidate", required=True)
    s.add_argument("--reviewer", required=True)
    s.add_argument("--note", default="")
    s.add_argument("--scripted", action="store_true", help="record the approval as made by a script, not a human")

    s = sub.add_parser("reject", help="human rejection of all candidates")
    s.add_argument("run")
    s.add_argument("--reviewer", required=True)
    s.add_argument("--note", default="")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "doctor":
            info = doctor.inspect(a.runs_dir)
            print(json.dumps(info, indent=2) if a.json else doctor.format_report(info))
            return 0
        if a.cmd == "validate":
            errors = schemas.validation_errors(read_json(Path(a.file)), a.expect)
            if errors:
                print("INVALID\n" + "\n".join(f"  {e}" for e in errors))
                return 1
            print("valid")
            return 0
        if a.cmd == "caps":
            engine = create_engine(load_config(a.config)["engine"])
            av = engine.availability()
            print(json.dumps({"capabilities": engine.capabilities().describe(),
                              "available": av.ok, "reason": av.reason}, indent=2))
            return 0
        if a.cmd == "compile":
            from atelier.prompt_compiler import compile_request

            cfg = load_config(a.config)
            ad, brief = read_json(Path(a.art_direction)), read_json(Path(a.brief))
            schemas.validate(ad, "atelier.art_direction.v1")
            schemas.validate(brief, "atelier.creative_brief.v1")
            g = cfg["generation"]
            req = compile_request(ad, brief, create_engine(cfg["engine"]).capabilities(), run_id="run_preview",
                                  iteration=0, slot=a.slot, base_seed=g["base_seed"], keep_seeds=True,
                                  engine_params=cfg["engine_params"], timeout_s=g["timeout_s"], max_retries=g["max_retries"])
            print(json.dumps(req, indent=2))
            return 0
        if a.cmd == "run":
            orch = Orchestrator.create(a.runs_dir, a.run_id, read_json(Path(a.brief)), load_config(a.config))
            orch.run(stop_at=State(a.stop_at) if a.stop_at else None)
            print(_summary(orch))
            return 0 if orch.state["state"] != "FAILED" else 2
        run_dir = _run_dir(a.run, a.runs_dir)
        orch = Orchestrator.load(run_dir)
        if a.cmd == "resume":
            orch.run(stop_at=State(a.stop_at) if a.stop_at else None)
        elif a.cmd == "cancel":
            orch.request_cancel()
            print("cancellation requested (takes effect at the next step/candidate boundary)")
        elif a.cmd == "verify":
            problems = orch.store.verify()
            print("integrity OK" if not problems else "\n".join(problems))
            return 0 if not problems else 3
        elif a.cmd == "report":
            from atelier import report

            print(report.write_final_report(orch)[0])
            return 0
        elif a.cmd == "shortlist":
            sel_id = orch.state["iterations"][-1]["selection_id"] if orch.state["iterations"] else None
            print("\n".join(orch.store.get(sel_id)["shortlist"]) if sel_id else "")
            return 0
        elif a.cmd == "approve":
            orch.approve(a.candidate, a.reviewer, a.note, reviewer_kind="script" if a.scripted else "human")
            orch.run()
        elif a.cmd == "reject":
            orch.reject(a.reviewer, a.note)
            orch.run()
        print(_summary(orch))
        return 0 if orch.state["state"] != "FAILED" else 2
    except (ConfigError, RunError, schemas.SchemaValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
