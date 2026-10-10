"""Run directory layout, immutable validated artifacts, integrity verification.

runs/<run_id>/
  state.json          persisted run state (validated, atomic)
  config.json         frozen run config
  manifest.json       artifact id -> {kind, path, sha256, created_at}
  events.jsonl        append-only event log
  artifacts/<kind>/<id>.json
  images/<candidate_id>/a<attempt>/...
  tasks/ inbox/ rejected/   agent hand-off (see agents_io.py)
  report/             final report, contact sheets
  CANCEL              presence requests cancellation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from atelier import schemas
from atelier.util import atomic_write_bytes, atomic_write_json, read_json, sha256_bytes, sha256_file, utc_now


class IntegrityError(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for sub in ("artifacts", "images", "tasks", "inbox", "rejected", "report"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._manifest: dict[str, dict[str, Any]] = (
            read_json(self._manifest_path)["artifacts"] if self._manifest_path.exists() else {}
        )
        self._validated: set[str] = set()  # digests validated by this process (content is hash-pinned)

    # ------------------------------------------------------------------ paths
    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def cancel_flag(self) -> Path:
        return self.root / "CANCEL"

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def abs(self, rel_path: str) -> Path:
        p = (self.root / rel_path).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise IntegrityError(f"path escapes run directory: {rel_path}")
        return p

    def candidate_image_dir(self, candidate_id: str, attempt: int) -> Path:
        d = self.root / "images" / candidate_id / f"a{attempt}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -------------------------------------------------------------- artifacts
    def put(self, artifact: dict[str, Any]) -> str:
        """Validate and persist an artifact. Ids are immutable: re-putting identical content
        is a no-op, different content under an existing id raises IntegrityError."""
        atype = schemas.validate(artifact)
        art_id = artifact[atype.id_field]
        data = (json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        digest = sha256_bytes(data)
        existing = self._manifest.get(art_id)
        if existing is not None:
            if existing["sha256"] == digest:
                return art_id
            raise IntegrityError(f"artifact {art_id} already exists with different content (artifacts are immutable)")
        path = self.root / "artifacts" / atype.kind / f"{art_id}.json"
        atomic_write_bytes(path, data)
        self._validated.add(digest)
        self._manifest[art_id] = {
            "kind": atype.kind,
            "schema_version": atype.schema_version,
            "path": self.rel(path),
            "sha256": digest,
            "created_at": utc_now(),
        }
        self._write_manifest()
        self.log_event("artifact_stored", artifact_id=art_id, kind=atype.kind)
        return art_id

    def get(self, art_id: str, revalidate: bool = False) -> dict[str, Any]:
        entry = self._manifest.get(art_id)
        if entry is None:
            raise KeyError(f"unknown artifact {art_id}")
        data = self.abs(entry["path"]).read_bytes()
        if sha256_bytes(data) != entry["sha256"]:
            raise IntegrityError(f"artifact {art_id} was modified on disk (sha256 mismatch)")
        artifact = json.loads(data)
        if revalidate or entry["sha256"] not in self._validated:
            schemas.validate(artifact, entry["schema_version"])
            self._validated.add(entry["sha256"])
        return artifact

    def has(self, art_id: str) -> bool:
        return art_id in self._manifest

    def ids(self, kind: str | None = None) -> list[str]:
        items = sorted(self._manifest.items(), key=lambda kv: kv[1]["created_at"])
        return [k for k, v in items if kind is None or v["kind"] == kind]

    def path_of(self, art_id: str) -> Path:
        return self.abs(self._manifest[art_id]["path"])

    def _write_manifest(self) -> None:
        atomic_write_json(self._manifest_path, {"artifacts": self._manifest})

    # ------------------------------------------------------------------ state
    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        schemas.validate(state, "atelier.run_state.v1")
        atomic_write_json(self.state_path, state)

    def load_state(self) -> dict[str, Any]:
        state = read_json(self.state_path)
        schemas.validate(state, "atelier.run_state.v1")
        return state

    # ----------------------------------------------------------------- events
    def log_event(self, event: str, **data: Any) -> None:
        line = json.dumps({"at": utc_now(), "event": event, **data}, ensure_ascii=False, default=str)
        with open(self.root / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def events(self) -> Iterable[dict[str, Any]]:
        path = self.root / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # -------------------------------------------------------------- integrity
    def verify(self) -> list[str]:
        """Re-hash every artifact and every recorded image; re-validate every artifact."""
        problems: list[str] = []
        for art_id, entry in self._manifest.items():
            try:
                artifact = self.get(art_id, revalidate=True)
            except (IntegrityError, schemas.SchemaValidationError, OSError, ValueError) as exc:
                problems.append(f"{art_id}: {exc}")
                continue
            if entry["kind"] == "generation_result":
                for img in artifact["images"]:
                    try:
                        p = self.abs(img["path"])
                        if not p.exists():
                            problems.append(f"{art_id}: missing image {img['path']}")
                        elif sha256_file(p) != img["sha256"]:
                            problems.append(f"{art_id}: image {img['path']} sha256 mismatch")
                    except IntegrityError as exc:
                        problems.append(f"{art_id}: {exc}")
        return problems
