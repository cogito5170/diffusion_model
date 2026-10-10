"""Resumable, deterministic workflow driver.

The orchestrator is a state machine persisted in ``runs/<run_id>/state.json``. Each call
to ``step()`` performs the work for the current state, saves, and either advances,
pauses (waiting for an agent artifact or a human decision), or ends (DONE / FAILED /
CANCELLED). Re-running after a crash or a pause continues from the persisted state.
Work already recorded (requests, results, QA, critiques) is never redone.
"""

from __future__ import annotations

import copy
import re
import traceback
from pathlib import Path
from typing import Any

from atelier import report, schemas, spec_checks
from atelier.agents_io import AgentBackend, AgentOutputError, AgentRequest, make_backend
from atelier.artifact_store import ArtifactStore, IntegrityError
from atelier.budget import Budget, Cancelled, CancelToken
from atelier.config import validate_config
from atelier.engines import create_engine
from atelier.engines.base import ImageEngine
from atelier.engines.runner import execute_request
from atelier.evaluator import evaluate_criterion, run_qa, weighted_score
from atelier.iteration import apply_plan, stopping_decision, validate_plan
from atelier.prompt_compiler import CompilationError, compile_request, engine_input
from atelier.selector import CandidateView, select
from atelier.state_machine import TERMINAL, State, check_transition
from atelier.util import atomic_write_json, read_json, sha256_json, utc_now

ADVANCED, PAUSED, ENDED = "advanced", "paused", "ended"
RUN_ID_RE = re.compile(r"^run_[A-Za-z0-9][A-Za-z0-9.-]{0,60}$")


class RunError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class Orchestrator:
    # ---------------------------------------------------------------- set-up
    def __init__(self, store: ArtifactStore, config: dict[str, Any], engine: ImageEngine | None = None,
                 agents: AgentBackend | None = None):
        self.store = store
        self.cfg = config
        self.state = store.load_state()
        if sha256_json(config) != self.state["config_sha256"]:
            raise RunError("config_changed", "run config differs from the one frozen at creation")
        self.engine = engine or create_engine(config["engine"])
        self.agents = agents or make_backend(config["agent_backend"])
        self.cancel_token = CancelToken(store.cancel_flag)
        self.budget = Budget(config["budget"], self.state["ledger"])
        self.stem = self.state["run_id"].split("_", 1)[1]

    @classmethod
    def create(cls, runs_root: str | Path, run_id: str, brief: dict[str, Any], config: dict[str, Any],
               engine: ImageEngine | None = None, agents: AgentBackend | None = None) -> "Orchestrator":
        validate_config(config)
        schemas.validate(brief, "atelier.creative_brief.v1")
        if brief["system_interpretation"] is not None:
            raise RunError("invalid_brief", "system_interpretation must be null on intake; the system fills it")
        if not RUN_ID_RE.match(run_id):
            raise RunError("invalid_run_id", f"run id must match {RUN_ID_RE.pattern}, got {run_id!r}")
        root = Path(runs_root) / run_id
        if (root / "state.json").exists():
            raise RunError("run_exists", f"{root} already contains a run; use resume")
        store = ArtifactStore(root)
        atomic_write_json(root / "config.json", config)
        store.put(brief)
        now = utc_now()
        state = {
            "schema_version": "atelier.run_state.v1",
            "run_id": run_id,
            "state": State.CREATED.value,
            "waiting_for": None,
            "iteration": 0,
            "pointers": {"brief": brief["brief_id"]},
            "iterations": [],
            "ledger": Budget.empty_ledger(),
            "agent_rejections": {},
            "stop_reason": None,
            "failure": None,
            "final_candidate_id": None,
            "original_intent_sha256": sha256_json(brief["original_intent"]),
            "config_sha256": sha256_json(config),
            "history": [],
            "created_at": now,
            "updated_at": now,
        }
        store.save_state(state)
        store.log_event("run_created", run_id=run_id, engine=config["engine"], agent_backend=config["agent_backend"])
        return cls(store, config, engine, agents)

    @classmethod
    def load(cls, run_dir: str | Path, engine: ImageEngine | None = None,
             agents: AgentBackend | None = None) -> "Orchestrator":
        store = ArtifactStore(run_dir)
        config = validate_config(read_json(Path(run_dir) / "config.json"))
        return cls(store, config, engine, agents)

    # --------------------------------------------------------------- helpers
    @property
    def current(self) -> State:
        return State(self.state["state"])

    def _save(self) -> None:
        self.store.save_state(self.state)

    def transition(self, target: State, note: str = "") -> None:
        check_transition(self.current, target)
        self.state["history"].append({"from": self.current.value, "to": target.value, "at": utc_now(), "note": note})
        self.state["state"] = target.value
        self.state["waiting_for"] = None
        self._save()
        self.store.log_event("transition", to=target.value, note=note)

    def _it(self) -> dict[str, Any]:
        return self.state["iterations"][self.state["iteration"]]

    def _brief(self) -> dict[str, Any]:
        brief = self.store.get(self.state["pointers"]["brief"])
        if sha256_json(brief["original_intent"]) != self.state["original_intent_sha256"]:
            raise IntegrityError("original intent no longer matches the hash recorded at intake")
        return brief

    def _ad(self) -> dict[str, Any]:
        return self.store.get(self._it()["art_direction_id"])

    def _caps(self):
        return self.engine.capabilities()

    def _paths(self, **ids: str) -> dict[str, str]:
        return {k: self.store.rel(self.store.path_of(v)) for k, v in ids.items()}

    def _all_candidates(self) -> list[tuple[int, dict[str, Any]]]:
        return [(it["iteration"], c) for it in self.state["iterations"] for c in it["candidates"]]

    def _image_path(self, cand: dict[str, Any]) -> Path | None:
        if not cand["result_id"]:
            return None
        res = self.store.get(cand["result_id"])
        return self.store.abs(res["images"][0]["path"]) if res["status"] == "succeeded" else None

    def _pause(self, kind: str, step: str, keys: list[str], agent: str | None, task_paths: list[str]) -> str:
        self.state["waiting_for"] = {"kind": kind, "step": step, "keys": keys, "agent": agent, "task_paths": task_paths}
        self._save()
        self.store.log_event("paused", waiting_for=self.state["waiting_for"])
        return PAUSED

    # ---------------------------------------------------------- agent steps
    def _obtain(self, req: AgentRequest, extra_check=None) -> dict[str, Any] | None:
        """Fetch + strictly validate an agent artifact. Invalid output is rejected (never
        coerced); after too many rejections the run fails."""
        slug = req.slug
        err_path = self.store.root / "tasks" / f"{slug}.errors.json"
        while True:
            attempt = self.state["agent_rejections"].get(slug, 0) + 1
            previous = read_json(err_path)["errors"] if err_path.exists() else []
            try:
                art = self.agents.obtain(req, self.store, previous, attempt)
            except AgentOutputError as exc:
                art, errors = None, [str(exc)]
            else:
                if art is None:
                    return None
                errors = schemas.validation_errors(art, req.schema_version)
                if not errors:
                    for k, v in req.required_values.items():
                        if art.get(k) != v:
                            errors.append(f"/{k}: must be {v!r}, got {art.get(k)!r}")
                if not errors and extra_check is not None:
                    errors = extra_check(art)
            if not errors:
                return art
            self.state["agent_rejections"][slug] = attempt
            atomic_write_json(err_path, {"errors": errors})
            self.agents.on_rejected(req, self.store, errors)
            self.store.log_event("agent_output_rejected", step=req.step, key=req.key, attempt=attempt, errors=errors[:10])
            self._save()
            if attempt > self.cfg["agents"]["max_rejections_per_step"]:
                raise RunError("agent_output_rejected", f"{req.agent} output for {slug} rejected {attempt} time(s)",
                               {"errors": errors[:20]})
            if not self.agents.synchronous:
                # re-issue the task with the errors attached, then wait for a corrected artifact
                self.agents.obtain(req, self.store, errors, attempt + 1)
                return None

    def _agent_or_pause(self, req: AgentRequest, extra_check=None):
        art = self._obtain(req, extra_check)
        if art is None:
            tp = self.agents.task_path(req, self.store)
            return None, self._pause("agent", req.step, [req.key], req.agent, [self.store.rel(tp)] if tp else [])
        return art, None

    # ----------------------------------------------------------------- run
    def run(self, max_steps: int = 500, stop_at: State | None = None) -> dict[str, Any]:
        """Advance until terminal, paused, or (optionally) the run reaches ``stop_at``.

        ``stop_at=State.SPEC_VALIDATED`` lets judgement steps run here and generation run elsewhere
        (e.g. design the art direction in a cloud session, generate on a local GPU machine)."""
        for _ in range(max_steps):
            if self.current in TERMINAL or (stop_at is not None and self.current == stop_at):
                break
            if self.cancel_token.is_cancelled():
                self._end(State.CANCELLED, "cancellation requested")
                break
            if self.step() != ADVANCED:
                break
        return self.state

    def step(self) -> str:
        handler = getattr(self, f"_on_{self.current.value.lower()}")
        try:
            return handler()
        except Cancelled:
            self._end(State.CANCELLED, "cancellation requested")
        except RunError as exc:
            self._fail(exc.code, str(exc), exc.details)
        except CompilationError as exc:
            self._fail("compilation_failed", str(exc))
        except IntegrityError as exc:
            self._fail("integrity_error", str(exc))
        except Exception as exc:  # noqa: BLE001 - graceful failure with full record
            self._fail("internal_error", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()[-4000:]})
        return ENDED

    def _fail(self, code: str, message: str, details: dict | None = None) -> None:
        self.state["failure"] = {"code": code, "message": message[:4000], "retryable": False, "details": details or {}}
        self._end(State.FAILED, f"{code}: {message[:200]}")

    def _end(self, target: State, note: str) -> None:
        if self.current in TERMINAL:
            return
        self.transition(target, note)
        try:
            report.write_final_report(self)
        except Exception as exc:  # noqa: BLE001 - report is best effort on failure paths
            self.store.log_event("report_failed", error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------- handlers
    def _on_created(self) -> str:
        self._brief()
        self.transition(State.BRIEF_ACCEPTED, "original intent hash recorded")
        return ADVANCED

    def _on_brief_accepted(self) -> str:
        brief = self._brief()
        req = AgentRequest(
            "research", "brief", self._paths(brief=brief["brief_id"]),
            {"research_id": f"research_{self.stem}", "brief_id": brief["brief_id"]},
            "Research artistic, philosophical and visual-cultural context for the brief. Mark sources verified "
            "only if actually opened. Put your interpretation in `interpretation`; never rewrite the user's intent.",
            {"brief": brief},
        )
        art, paused = self._agent_or_pause(req)
        if paused:
            return paused
        self.store.put(art)
        interpreted = copy.deepcopy(brief)
        interpreted["brief_id"] = f"{brief['brief_id']}.interpreted"
        interpreted["system_interpretation"] = {
            "interpreted_by": art["produced_by"],
            "interpreted_at": utc_now(),
            "source_artifact_id": art["research_id"],
            **{k: art["interpretation"][k] for k in ("summary", "themes", "tensions", "open_questions")},
        }
        if sha256_json(interpreted["original_intent"]) != self.state["original_intent_sha256"]:
            raise IntegrityError("original intent changed during interpretation")
        self.store.put(interpreted)
        self.state["pointers"].update(research=art["research_id"], interpreted_brief=interpreted["brief_id"])
        self.transition(State.RESEARCHED)
        return ADVANCED

    def _on_researched(self) -> str:
        brief = self._brief()
        p = self.state["pointers"]

        def check(cs):
            ids = [c["concept_id"] for c in cs["concepts"]]
            errs = []
            if len(set(ids)) != len(ids):
                errs.append("concept_id values must be unique")
            if len({c["title"].strip().lower() for c in cs["concepts"]}) != len(ids):
                errs.append("concept titles must be distinct")
            if len({c["thesis"].strip().lower() for c in cs["concepts"]}) != len(ids):
                errs.append("concept theses must be distinct")
            if cs.get("recommended_concept_id") not in (None, *ids):
                errs.append("recommended_concept_id is not one of the concepts")
            return errs

        req = AgentRequest(
            "concepts", "brief",
            self._paths(brief=p["interpreted_brief"], research=p["research"]),
            {"concept_set_id": f"concepts_{self.stem}", "brief_id": brief["brief_id"], "research_id": p["research"]},
            "Produce 2-6 clearly distinct concepts, each with an explicit conceptual tension (pole_a vs pole_b). "
            "Use brief_id exactly as in required_values (the original brief), even though the input is its "
            "'.interpreted' copy.",
            {"brief": brief, "research": self.store.get(p["research"])},
        )
        art, paused = self._agent_or_pause(req, check)
        if paused:
            return paused
        self.store.put(art)
        p["concepts"] = art["concept_set_id"]
        self.transition(State.CONCEPTS_READY)
        return ADVANCED

    def _on_concepts_ready(self) -> str:
        brief = self._brief()
        p = self.state["pointers"]
        cset = self.store.get(p["concepts"])

        def check(draft):
            errs = []
            if draft["concept_id"] not in {c["concept_id"] for c in cset["concepts"]}:
                errs.append(f"/concept_id {draft['concept_id']!r} is not in the concept set")
            e, _ = spec_checks.check_art_direction(draft, brief)
            return errs + e

        req = AgentRequest(
            "art_direction_draft", "v1",
            self._paths(brief=p["interpreted_brief"], research=p["research"], concepts=p["concepts"]),
            {"draft_id": f"draft_{self.stem}", "brief_id": brief["brief_id"]},
            "Develop ONE concept (prefer recommended_concept_id) into engine-independent art direction: thesis, "
            "visual hypothesis, composition, palette, mood, semantic and avoidance constraints. No prompt syntax. "
            "Leave materials and lighting to material-specialist (write notes for them). Use brief_id exactly as in "
            "required_values (the original brief), not the '.interpreted' input id.",
            {"brief": brief, "concepts": cset},
        )
        art, paused = self._agent_or_pause(req, check)
        if paused:
            return paused
        self.store.put(art)
        p["art_direction_draft"] = art["draft_id"]
        self.transition(State.ART_DIRECTION_DRAFTED)
        return ADVANCED

    def _merge(self, draft: dict[str, Any], mspec: dict[str, Any]) -> dict[str, Any]:
        keys = ("concept_thesis", "visual_hypothesis", "subject", "composition", "palette", "mood",
                "semantic_constraints", "avoidance_constraints", "output")
        return {
            "schema_version": "atelier.art_direction.v1",
            "art_direction_id": f"ad_{self.stem}-v1",
            "version": 1,
            "parent_id": None,
            "brief_id": draft["brief_id"],
            "concept_id": draft["concept_id"],
            "provenance": {"draft_id": draft["draft_id"], "material_spec_id": mspec["material_spec_id"],
                           "derived_from_plan_id": None},
            **{k: copy.deepcopy(draft[k]) for k in keys},
            "materials": copy.deepcopy(mspec["materials"]),
            "lighting": copy.deepcopy(mspec["lighting"]),
        }

    def _on_art_direction_drafted(self) -> str:
        brief = self._brief()
        p = self.state["pointers"]
        draft = self.store.get(p["art_direction_draft"])
        merged_box: dict[str, Any] = {}

        def check(mspec):
            merged = self._merge(draft, mspec)
            errs = [f"merged art direction: {e}" for e in schemas.validation_errors(merged, "atelier.art_direction.v1")]
            if not errs:
                e, warns = spec_checks.check_art_direction(merged, brief)
                errs += e
                merged_box.update(ad=merged, warnings=warns)
            return errs

        req = AgentRequest(
            "material_spec", "v1",
            self._paths(brief=p["interpreted_brief"], art_direction_draft=p["art_direction_draft"]),
            {"material_spec_id": f"matspec_{self.stem}", "draft_id": draft["draft_id"]},
            "Specify metal types, reflectivity, roughness, anisotropy, curvature, imperfections and lighting so "
            "the materials are physically plausible. Values are engine-independent.",
            {"brief": brief, "draft": draft},
        )
        art, paused = self._agent_or_pause(req, check)
        if paused:
            return paused
        self.store.put(art)
        ad = merged_box["ad"]
        self.store.put(ad)
        if merged_box["warnings"]:
            self.store.log_event("spec_warnings", art_direction_id=ad["art_direction_id"], warnings=merged_box["warnings"])
        p.update(material_spec=art["material_spec_id"], art_direction=ad["art_direction_id"])
        self.state["iterations"] = [{"iteration": 0, "art_direction_id": ad["art_direction_id"], "candidates": [],
                                     "selection_id": None, "curation_id": None, "plan_id": None, "plan_outcome": None}]
        self.transition(State.SPEC_VALIDATED, "schema + semantic checks passed")
        return ADVANCED

    def _on_spec_validated(self) -> str:
        it = self._it()
        if not it["candidates"]:
            ad, brief, caps = self._ad(), self._brief(), self._caps()
            keep = True
            if self.state["iteration"] > 0:
                prev_plan = self.state["iterations"][self.state["iteration"] - 1]["plan_id"]
                keep = self.store.get(prev_plan)["keep_seeds"]
            gen = self.cfg["generation"]
            for slot in range(self.cfg["batch_size"]):
                req = compile_request(
                    ad, brief, caps, run_id=self.state["run_id"], iteration=self.state["iteration"], slot=slot,
                    base_seed=gen["base_seed"], keep_seeds=keep, engine_params=self.cfg["engine_params"],
                    timeout_s=gen["timeout_s"], max_retries=gen["max_retries"],
                )
                self.store.put(req)
                it["candidates"].append({"candidate_id": req["candidate_id"], "slot": slot, "request_id": req["request_id"],
                                         "result_id": None, "qa_id": None, "critique_id": None})
            self._save()
        self.transition(State.PROMPTS_COMPILED, f"{len(it['candidates'])} requests for {it['art_direction_id']}")
        return ADVANCED

    def _on_prompts_compiled(self) -> str:
        it = self._it()
        for cand in it["candidates"]:
            if cand["result_id"]:
                continue  # resume: never regenerate a recorded candidate
            self.cancel_token.raise_if_cancelled()
            request = self.store.get(cand["request_id"])
            result = execute_request(self.engine, request, self.store, self.budget, self.cancel_token,
                                     retry_on_timeout=self.cfg["generation"]["retry_on_timeout"])
            self.store.put(result)
            cand["result_id"] = result["result_id"]
            self._save()
            self.store.log_event("generation_result", candidate_id=cand["candidate_id"], status=result["status"],
                                 runtime_s=result["runtime_s"])
            if result["status"] == "cancelled":
                raise Cancelled("cancelled during generation")
        if not any(self._image_path(c) for _, c in self._all_candidates()):
            errs = [e for c in it["candidates"] for e in self.store.get(c["result_id"])["errors"]]
            raise RunError("no_images_generated", "no candidate image was produced in this run",
                           {"errors": errs[:20]})
        self.transition(State.GENERATED)
        return ADVANCED

    def _on_generated(self) -> str:
        for cand in self._it()["candidates"]:
            if cand["qa_id"] or not self._image_path(cand):
                continue
            qa = run_qa(self.store.get(cand["result_id"]), self.store.get(cand["request_id"]), self.store, self.cfg["qa"])
            self.store.put(qa)
            cand["qa_id"] = qa["qa_id"]
            self._save()
        self.transition(State.QA_DONE)
        return ADVANCED

    def _on_qa_done(self) -> str:
        it = self._it()
        ad = self._ad()
        brief_id = self.state["pointers"]["interpreted_brief"]
        passed = [c for c in it["candidates"] if c["qa_id"] and self.store.get(c["qa_id"])["passed"]]
        images = {c["candidate_id"]: self._image_path(c) for c in passed}
        pending, tasks = [], []
        for cand in passed:
            if cand["critique_id"]:
                continue
            cid = cand["candidate_id"]

            def check(cr):
                obs_types = {o["evidence_type"] for o in cr["observations"]}
                return [] if obs_types <= set(cr["evidence_types"]) else ["evidence_types must include every observation's evidence_type"]

            req = AgentRequest(
                "critique", cid,
                {"image": self.store.rel(images[cid]),
                 **self._paths(art_direction=ad["art_direction_id"], brief=brief_id, qa=cand["qa_id"],
                               request=cand["request_id"])},
                {"critique_id": f"crit_{cid.split('_', 1)[1]}", "candidate_id": cid, "art_direction_id": ad["art_direction_id"]},
                "Look at the image. Rate intent alignment, material plausibility, composition, visual coherence, "
                "originality and technical quality (integers 1-5) against the art direction. Ground every "
                "observation in what is visible; state confidence honestly; propose one next experiment.",
                {"image_path": images[cid], "art_direction": ad, "qa": self.store.get(cand["qa_id"]),
                 "peer_paths": [p for k, p in images.items() if k != cid]},
            )
            art = self._obtain(req, check)
            if art is None:
                pending.append(cid)
                tp = self.agents.task_path(req, self.store)
                if tp:
                    tasks.append(self.store.rel(tp))
                continue
            self.store.put(art)
            cand["critique_id"] = art["critique_id"]
            self._save()
        if pending:
            return self._pause("agent", "critique", pending, "art-critic", tasks)
        self.transition(State.CRITIQUED)
        return ADVANCED

    def _engine_input(self, ad: dict[str, Any]) -> dict[str, Any]:
        g = self.cfg["generation"]
        req = compile_request(ad, self._brief(), self._caps(), run_id=self.state["run_id"], iteration=0, slot=0,
                              base_seed=g["base_seed"], keep_seeds=True, engine_params=self.cfg["engine_params"],
                              timeout_s=g["timeout_s"], max_retries=g["max_retries"])
        return engine_input(req)

    def _views(self) -> list[CandidateView]:
        views = []
        for itn, c in self._all_candidates():
            crit = self.store.get(c["critique_id"]) if c["critique_id"] else None
            views.append(CandidateView(
                candidate_id=c["candidate_id"], iteration=itn, image_path=self._image_path(c),
                qa_passed=self.store.get(c["qa_id"])["passed"] if c["qa_id"] else None,
                ratings=crit["ratings"] if crit else None, critic_kind=crit["critic"]["kind"] if crit else None))
        return views

    def _critiques(self, iteration: int) -> list[dict[str, Any]]:
        return [self.store.get(c["critique_id"]) for c in self.state["iterations"][iteration]["candidates"] if c["critique_id"]]

    def _on_critiqued(self) -> str:
        it, n = self._it(), self.state["iteration"]
        ad = self._ad()
        if not it["selection_id"]:
            sel = select(self._views(), self.cfg["weights"], self.cfg["selection"], run_id=self.state["run_id"],
                         iteration=n, requires_human_approval=self.cfg["require_human_approval"])
            self.store.put(sel)
            it["selection_id"] = sel["selection_id"]
            if n > 0 and it["plan_outcome"] is None:
                prev_plan = self.store.get(self.state["iterations"][n - 1]["plan_id"])
                it["plan_outcome"] = evaluate_criterion(prev_plan, self._critiques(n - 1), self._critiques(n),
                                                        self.cfg["weights"])
            self._save()
        sel = self.store.get(it["selection_id"])
        known = {c["candidate_id"] for _, c in self._all_candidates()}
        brief = self._brief()

        def check(cur):
            errs = [f"candidate_notes: unknown candidate {x['candidate_id']}" for x in cur["candidate_notes"]
                    if x["candidate_id"] not in known]
            if cur["iteration_plan"] is not None:
                errs += [f"iteration_plan: {e}" for e in validate_plan(
                    cur["iteration_plan"], ad, current_iteration=n, selection_id=sel["selection_id"],
                    max_major_variables=self.cfg["max_major_variables_per_iteration"],
                    max_iterations=self.cfg["max_iterations"], brief=brief, engine_input_fn=self._engine_input)]
            return errs

        inputs = self._paths(selection=sel["selection_id"], art_direction=ad["art_direction_id"],
                             brief=self.state["pointers"]["interpreted_brief"],
                             compiled_request_example=it["candidates"][0]["request_id"])
        inputs["state"] = "state.json"
        req = AgentRequest(
            "curation", f"i{n}", inputs,
            {"curation_id": f"cur_{self.stem}-i{n}", "run_id": self.state["run_id"], "iteration": n,
             "selection_id": sel["selection_id"]},
            f"Compare candidates, explain the shortlist, and either recommend 'stop' or propose ONE bounded "
            f"experiment (iteration {n + 1} of max {self.cfg['max_iterations']}; change at most "
            f"{self.cfg['max_major_variables_per_iteration']} art-direction section(s)). State what failed, what "
            "will change, what remains fixed, why it should help, and the improvement criterion. Do not optimize "
            "only for the weighted score; protect diversity.",
            {"selection": sel, "art_direction": ad, "iteration": n, "max_iterations": self.cfg["max_iterations"],
             "batch_critiques": self._critiques(n), "run_stem": self.stem,
             "changes_engine_input": lambda new_ad: self._engine_input(new_ad) != self._engine_input(ad)},
        )
        art, paused = self._agent_or_pause(req, check)
        if paused:
            return paused
        self.store.put(art)
        it["curation_id"] = art["curation_id"]
        if art["iteration_plan"] is not None:
            self.store.put(art["iteration_plan"])
            it["plan_id"] = art["iteration_plan"]["plan_id"]
        self.transition(State.SELECTED)
        return ADVANCED

    def _on_selected(self) -> str:
        it, n = self._it(), self.state["iteration"]
        sel = self.store.get(it["selection_id"])
        cur = self.store.get(it["curation_id"])
        eligible = [r for r in sel["ranking"] if r["eligible"]]
        best = eligible[0] if eligible else None
        best_min = None
        if best:
            crit_id = next(c["critique_id"] for _, c in self._all_candidates() if c["candidate_id"] == best["candidate_id"])
            best_min = min(self.store.get(crit_id)["ratings"].values())
        prior = [r["weighted_score"] for r in eligible if r["iteration"] < n]
        this = [r["weighted_score"] for r in eligible if r["iteration"] == n]
        reason = stopping_decision(
            iteration=n, max_iterations=self.cfg["max_iterations"],
            best_overall=best["weighted_score"] if best else None, best_overall_min_rating=best_min,
            best_before_this_iteration=max(prior) if prior else None, best_this_iteration=max(this) if this else None,
            stopping_cfg=self.cfg["stopping"], curator_recommendation=cur["recommendation"],
            curator_stop_reason=cur["stop_reason"],
            can_afford_next_batch=self.budget.can_afford_batch(self.cfg["batch_size"], self._caps().cost_usd_per_image),
        )
        if reason is None and it["plan_id"] is None:
            reason = "no_plan: curator did not propose an experiment"
        if reason is None:
            new_ad = apply_plan(self.store.get(it["plan_id"]), self._ad())
            self.store.put(new_ad)
            self.state["iteration"] = n + 1
            self.state["iterations"].append({"iteration": n + 1, "art_direction_id": new_ad["art_direction_id"],
                                             "candidates": [], "selection_id": None, "curation_id": None,
                                             "plan_id": None, "plan_outcome": None})
            self.state["pointers"]["art_direction"] = new_ad["art_direction_id"]
            self.transition(State.SPEC_VALIDATED, f"bounded iteration {n + 1}: {it['plan_id']}")
            return ADVANCED
        self.state["stop_reason"] = reason
        if self.cfg["require_human_approval"]:
            self.transition(State.AWAITING_HUMAN_APPROVAL, reason)
            return ADVANCED
        if not sel["shortlist"]:
            self.transition(State.AWAITING_HUMAN_APPROVAL, reason + " (nothing eligible to auto-approve)")
            return ADVANCED
        self._record_approval("approved", sel["shortlist"][0], {"name": "config", "kind": "config_auto"},
                              "require_human_approval=false in run config; top of shortlist auto-selected")
        self.transition(State.APPROVED, reason)
        return ADVANCED

    def _on_awaiting_human_approval(self) -> str:
        review = report.write_review_sheet(self)
        return self._pause("human", "approval", [], None, [self.store.rel(review)])

    def _on_approved(self) -> str:
        report.write_final_report(self)
        self.transition(State.DONE, "final report written")
        report.write_final_report(self)  # refresh with final state
        return ENDED

    _on_rejected = _on_approved

    # ------------------------------------------------------------- decisions
    def _record_approval(self, decision: str, candidate_id: str | None, reviewer: dict[str, str], note: str) -> dict:
        reasons: list[str] = []
        if candidate_id is not None:
            cands = {c["candidate_id"]: c for _, c in self._all_candidates()}
            if candidate_id not in cands:
                raise RunError("unknown_candidate", f"{candidate_id} is not a candidate of this run")
            if not self._image_path(cands[candidate_id]):
                raise RunError("candidate_has_no_image", f"{candidate_id} has no generated image")
            sel = self.store.get(self._it()["selection_id"]) if self._it()["selection_id"] else None
            if sel:
                ranked = [r["candidate_id"] for r in sel["ranking"] if r["eligible"]]
                if candidate_id not in sel["shortlist"]:
                    reasons.append("not in system shortlist")
                if ranked and candidate_id != ranked[0]:
                    reasons.append("not the top weighted score")
                if candidate_id not in ranked:
                    reasons.append("ineligible in system ranking (QA failure or no critique)")
        n = len(self.store.ids("approval")) + 1
        approval = {
            "schema_version": "atelier.approval.v1",
            "approval_id": f"approval_{self.stem}-{n}",
            "run_id": self.state["run_id"],
            "decision": decision,
            "candidate_id": candidate_id,
            "reviewer": reviewer,
            "note": note,
            "override": {"is_override": bool(reasons), "reasons": reasons},
            "decided_at": utc_now(),
        }
        self.store.put(approval)
        self.state["pointers"]["approval"] = approval["approval_id"]
        self.state["final_candidate_id"] = candidate_id if decision == "approved" else None
        self._save()
        return approval

    def approve(self, candidate_id: str, reviewer: str, note: str = "", reviewer_kind: str = "human") -> dict[str, Any]:
        """``reviewer_kind="script"`` marks non-human approvals (demos, CI) so they are never mistaken for human judgement."""
        if self.current != State.AWAITING_HUMAN_APPROVAL:
            raise RunError("not_awaiting_approval", f"run is in state {self.current.value}")
        approval = self._record_approval("approved", candidate_id, {"name": reviewer, "kind": reviewer_kind}, note)
        self.transition(State.APPROVED, f"approved {candidate_id} by {reviewer}")
        return approval

    def reject(self, reviewer: str, note: str = "") -> dict[str, Any]:
        if self.current != State.AWAITING_HUMAN_APPROVAL:
            raise RunError("not_awaiting_approval", f"run is in state {self.current.value}")
        approval = self._record_approval("rejected", None, {"name": reviewer, "kind": "human"}, note)
        self.transition(State.REJECTED, f"rejected by {reviewer}")
        return approval

    def request_cancel(self) -> None:
        self.store.cancel_flag.write_text(utc_now() + "\n")
        self.store.log_event("cancel_requested")
