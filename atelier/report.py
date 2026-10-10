"""Human-readable review sheet (for the approval gate) and the final run report.

Reports separate measured facts (runtime, statuses, QA metrics) from estimates
(cost) and from judgements (critique ratings, labelled by critic kind).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

from atelier import schemas
from atelier.evaluator import weighted_score
from atelier.util import atomic_write_bytes, atomic_write_json, utc_now

if TYPE_CHECKING:  # pragma: no cover
    from atelier.orchestrator import Orchestrator


def _rows(orch: "Orchestrator") -> list[dict[str, Any]]:
    rows = []
    for itn, c in orch._all_candidates():
        res = orch.store.get(c["result_id"]) if c["result_id"] else None
        qa = orch.store.get(c["qa_id"]) if c["qa_id"] else None
        crit = orch.store.get(c["critique_id"]) if c["critique_id"] else None
        img = res["images"][0]["path"] if res and res["images"] else None
        rows.append({
            "candidate_id": c["candidate_id"], "iteration": itn,
            "status": res["status"] if res else "not_run",
            "qa_passed": qa["passed"] if qa else None,
            "weighted_score": weighted_score(crit["ratings"], orch.cfg["weights"]) if crit else None,
            "critic_kind": crit["critic"]["kind"] if crit else None,
            "image_path": img,
            "runtime_s": res["runtime_s"] if res else None,
            "_ratings": crit["ratings"] if crit else None,
            "_errors": res["errors"] if res else [],
        })
    return rows


def contact_sheet(orch: "Orchestrator", rows: list[dict[str, Any]], path: Path, thumb: int = 192) -> Path | None:
    shown = [r for r in rows if r["image_path"]]
    if not shown:
        return None
    cols = min(4, len(shown))
    nrows = (len(shown) + cols - 1) // cols
    label_h = 30
    sheet = Image.new("RGB", (cols * thumb, nrows * (thumb + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    final = orch.state["final_candidate_id"]
    for i, r in enumerate(shown):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + label_h)
        with Image.open(orch.store.abs(r["image_path"])) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
            sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        score = f"{r['weighted_score']:.2f}" if r["weighted_score"] is not None else "-"
        tag = " *FINAL*" if r["candidate_id"] == final else ""
        draw.text((x + 4, y + thumb + 2), f"{r['candidate_id'].split('_', 1)[1]}{tag}", fill=(0, 0, 0))
        draw.text((x + 4, y + thumb + 15), f"score {score} qa {r['qa_passed']}", fill=(60, 60, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def _evidence_note(kinds: set[str]) -> str:
    if not kinds:
        return "No critiques were recorded."
    if kinds <= {"heuristic"}:
        return ("All ratings come from the offline HEURISTIC critic: pixel-statistic proxies with low confidence. "
                "They are not aesthetic judgements and must not be cited as evidence of image quality.")
    if "heuristic" in kinds:
        return "Ratings mix agent/human critiques with heuristic proxies; compare only like with like."
    return ("Ratings come from the art-critic agent (vision-language judgement). They are unvalidated model "
            "opinions until checked against blinded human ratings (see experiments/).")


def write_review_sheet(orch: "Orchestrator") -> Path:
    rows = _rows(orch)
    sel = orch.store.get(orch._it()["selection_id"]) if orch._it()["selection_id"] else None
    sheet = contact_sheet(orch, rows, orch.store.root / "report" / "contact_sheet.png")
    lines = [f"# Review: {orch.state['run_id']}", "",
             f"Stop reason: {orch.state['stop_reason']}", "",
             "Final selection needs a human decision. The system shortlist is advice; you may pick any candidate "
             "with an image (overrides are recorded).", ""]
    if sel:
        lines += [f"System shortlist (weighted score + MMR diversity): {', '.join(sel['shortlist']) or '(empty)'}", ""]
    lines += ["| candidate | iter | status | QA | weighted | critic | image |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ws = f"{r['weighted_score']:.2f}" if r["weighted_score"] is not None else "-"
        lines.append(f"| {r['candidate_id']} | {r['iteration']} | {r['status']} | {r['qa_passed']} | {ws} | "
                     f"{r['critic_kind'] or '-'} | {r['image_path'] or '-'} |")
    lines += ["", _evidence_note({r["critic_kind"] for r in rows if r["critic_kind"]}), ""]
    if sheet:
        lines += [f"Contact sheet: {orch.store.rel(sheet)}", ""]
    lines += ["Approve:  python -m atelier approve " + str(orch.store.root) + " --candidate <id> --reviewer <name> [--note ...]",
              "Reject:   python -m atelier reject " + str(orch.store.root) + " --reviewer <name> [--note ...]"]
    path = orch.store.root / "report" / "review.md"
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))
    return path


def write_final_report(orch: "Orchestrator") -> tuple[Path, Path]:
    st = orch.state
    rows = _rows(orch)
    brief = orch.store.get(st["pointers"]["brief"])
    interp_id = st["pointers"].get("interpreted_brief")
    interp = orch.store.get(interp_id)["system_interpretation"] if interp_id else None
    problems = orch.store.verify()
    kinds = {r["critic_kind"] for r in rows if r["critic_kind"]}
    iters = []
    for it in st["iterations"]:
        scores = [r["weighted_score"] for r in rows if r["iteration"] == it["iteration"] and r["weighted_score"] is not None]
        iters.append({
            "iteration": it["iteration"], "art_direction_id": it["art_direction_id"], "plan_id": it["plan_id"],
            "plan_outcome": it["plan_outcome"],
            "best_weighted_score": max(scores) if scores else None,
            "mean_weighted_score": round(sum(scores) / len(scores), 4) if scores else None,
            "n_succeeded": sum(1 for r in rows if r["iteration"] == it["iteration"] and r["status"] == "succeeded"),
            "n_requested": len(it["candidates"]),
        })
    caps = orch.engine.capabilities()
    data = {
        "schema_version": "atelier.run_report.v1",
        "report_id": f"report_{orch.stem}",
        "run_id": st["run_id"],
        "final_state": st["state"],
        "stop_reason": st["stop_reason"],
        "final_candidate_id": st["final_candidate_id"],
        "approval_id": st["pointers"].get("approval"),
        "original_intent": brief["original_intent"]["text"],
        "system_interpretation_summary": interp["summary"] if interp else None,
        "engine": {"engine_id": caps.engine_id, "model_id": caps.model_id},
        "critic_kinds": sorted(kinds),
        "iterations": iters,
        "candidates": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "ledger": dict(st["ledger"]),
        "evidence_note": _evidence_note(kinds),
        "integrity": {"verified": not problems, "problems": problems},
        "generated_at": utc_now(),
    }
    schemas.validate(data, "atelier.run_report.v1")
    out = orch.store.root / "report"
    json_path = out / "report.json"
    atomic_write_json(json_path, data)
    sheet = contact_sheet(orch, rows, out / "contact_sheet.png")

    L: list[str] = [f"# Run report: {st['run_id']}", ""]
    L += [f"- **Final state:** {st['state']}", f"- **Stop reason:** {st['stop_reason']}",
          f"- **Final candidate:** {st['final_candidate_id']}"]
    if st["pointers"].get("approval"):
        ap = orch.store.get(st["pointers"]["approval"])
        L.append(f"- **Decision:** {ap['decision']} by {ap['reviewer']['name']} ({ap['reviewer']['kind']})"
                 + (f"; override: {', '.join(ap['override']['reasons'])}" if ap["override"]["is_override"] else ""))
    if st["failure"]:
        L.append(f"- **Failure:** `{st['failure']['code']}`: {st['failure']['message']}")
    L.append(f"- **Engine:** {caps.engine_id} / {caps.model_id}")
    L += ["", "## Brief", "", "**Original intent (verbatim, user-authored):**", "", "> " + brief["original_intent"]["text"], ""]
    if interp:
        L += [f"**System interpretation** (by {interp['interpreted_by']['name']}, {interp['interpreted_by']['kind']}): "
              + interp["summary"], ""]
    if st["iterations"]:
        ad0 = orch.store.get(st["iterations"][0]["art_direction_id"])
        L += ["## Concept", "", f"**Thesis:** {ad0['concept_thesis']}", "", f"**Visual hypothesis:** {ad0['visual_hypothesis']}", ""]
    L += ["## Iterations", ""]
    for i, it in enumerate(st["iterations"]):
        L.append(f"### Iteration {it['iteration']} ({it['art_direction_id']})")
        L.append(f"- requested {iters[i]['n_requested']}, succeeded {iters[i]['n_succeeded']}, best weighted "
                 f"{iters[i]['best_weighted_score']}, mean {iters[i]['mean_weighted_score']}")
        if it["plan_outcome"]:
            po = it["plan_outcome"]
            L.append(f"- previous plan criterion `{po['aggregate']}({po['metric']}) {po['comparator']} {po['threshold']}`: "
                     f"measured {po['measured']} (baseline {po['baseline']}) -> **{'met' if po['met'] else 'not met'}**")
        if it["plan_id"]:
            plan = orch.store.get(it["plan_id"])
            executed = plan["iteration"] < len(st["iterations"])
            L += [f"- **Experiment {plan['plan_id']} (for iteration {plan['iteration']})"
                  + ("" if executed else " - proposed, NOT executed (run stopped)") + "**",
                  "  - What failed: " + "; ".join(f"{w['dimension']}: {w['evidence']}" for w in plan["what_failed"]),
                  "  - What will change: " + "; ".join(f"`{c['path']}`: {c['from']!r} -> {c['to']!r}" for c in plan["what_will_change"]),
                  "  - What remains fixed: " + ", ".join(plan["what_remains_fixed"]),
                  "  - Why it should help: " + plan["why_it_should_help"],
                  "  - Improvement means: " + f"{plan['improvement_criterion']['aggregate']}({plan['improvement_criterion']['metric']}) "
                  f"{plan['improvement_criterion']['comparator']} {plan['improvement_criterion']['threshold']}"]
        L.append("")
    L += ["## Candidates", "", "| candidate | iter | status | QA | weighted | critic | runtime s | image |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ws = f"{r['weighted_score']:.2f}" if r["weighted_score"] is not None else "-"
        L.append(f"| {r['candidate_id']} | {r['iteration']} | {r['status']} | {r['qa_passed']} | {ws} | "
                 f"{r['critic_kind'] or '-'} | {r['runtime_s']} | {r['image_path'] or '-'} |")
    errs = [(r["candidate_id"], e) for r in rows for e in r["_errors"]]
    if errs:
        L += ["", "### Recorded errors", ""] + [f"- {cid}: `{e['code']}` (attempt {e.get('attempt')}): {e['message']}" for cid, e in errs]
    led = st["ledger"]
    L += ["", "## Budget ledger", "",
          f"- images (measured): {led['images']} / {orch.cfg['budget']['max_total_images']}",
          f"- generation attempts (measured): {led['attempts']} / {orch.cfg['budget']['max_generation_attempts']}",
          f"- engine seconds (measured wall-clock): {led['engine_seconds']} / {orch.cfg['budget']['max_engine_seconds']}",
          f"- cost USD (**estimated**, basis: {caps.cost_basis}): {led['cost_usd_estimated']} / {orch.cfg['budget']['max_cost_usd']}",
          "", "## Evidence quality", "", _evidence_note(kinds), "",
          "## Integrity", "", "All artifacts and images re-hashed and re-validated: " + ("OK" if not problems else "PROBLEMS"), ""]
    L += [f"- {p}" for p in problems]
    if sheet:
        L += ["", f"Contact sheet: `{orch.store.rel(sheet)}`"]
    md_path = out / "report.md"
    atomic_write_bytes(md_path, ("\n".join(L) + "\n").encode("utf-8"))
    return md_path, json_path
