"""Run an engine in a separate worker process.

Why: a native crash inside PyTorch/Metal (e.g. an MPS assertion) or the OS killing the process for memory takes
down whatever process holds the model. With isolation only the worker dies; the caller (studio, orchestrator)
gets a structured EngineError that says how it died, and the next request starts a fresh worker. It also makes
timeouts real: a request that overruns can be terminated, not just abandoned.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
from pathlib import Path
from typing import Any

from atelier.budget import Cancelled, CancelToken
from atelier.engines.base import Availability, EngineCapabilities, EngineError, EngineOutput, ImageEngine

_SIGNALS = {
    -signal.SIGKILL: "killed by the operating system (SIGKILL), most often because memory ran out",
    -signal.SIGABRT: "aborted (SIGABRT), typically a GPU/Metal (MPS) assertion inside PyTorch",
    -signal.SIGSEGV: "crashed (SIGSEGV) inside native code",
    -signal.SIGTERM: "terminated (SIGTERM)",
}


def describe_exit(code: int | None) -> str:
    if code is None:
        return "stopped responding"
    return _SIGNALS.get(code, f"exited with code {code}")


class _EventToken(CancelToken):
    """Cancellation shared across processes: the parent sets an Event, the child's pipeline sees it."""

    def __init__(self, event):
        super().__init__()
        self._mp_event = event

    def is_cancelled(self) -> bool:
        return self._mp_event.is_set() or super().is_cancelled()


def _worker(conn, engine_cfg: dict[str, Any], cancel_event) -> None:  # pragma: no cover - runs in the child process
    from atelier.engines import create_engine

    engine = create_engine(engine_cfg)
    engine.progress_callback = lambda stage, step, total: conn.send(("progress", stage, step, total))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        request, out_dir = msg
        try:
            out = engine.generate(request, Path(out_dir), _EventToken(cancel_event))
            conn.send(("ok", {"images": [str(p) for p in out.images], "model_id": out.model_id,
                              "seed_used": out.seed_used, "metadata": out.metadata}))
        except Cancelled:
            conn.send(("cancelled",))
        except EngineError as exc:
            conn.send(("engine_error", exc.code, str(exc), exc.retryable, exc.details))
        except Exception as exc:  # noqa: BLE001 - reported to the parent, which records it
            conn.send(("error", type(exc).__name__, str(exc)))


class IsolatedEngine(ImageEngine):
    CANCEL_GRACE_S = 45.0  # time for the worker to stop at a step boundary (keeps the model loaded) before a kill

    def __init__(self, engine_cfg: dict[str, Any], start_method: str = "spawn"):
        from atelier.engines import create_engine

        self.cfg = engine_cfg
        self.engine_id = engine_cfg["engine_id"]
        self._shadow = create_engine(engine_cfg)  # never generates; answers capabilities / availability only
        self._ctx = mp.get_context(start_method)
        self._proc = None
        self._conn = None
        self._cancel_event = None

    def capabilities(self) -> EngineCapabilities:
        return self._shadow.capabilities()

    def availability(self) -> Availability:
        return self._shadow.availability()

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        parent, child = self._ctx.Pipe()
        self._cancel_event = self._ctx.Event()
        self._proc = self._ctx.Process(target=_worker, args=(child, self.cfg, self._cancel_event), daemon=True,
                                       name="atelier-engine")
        self._proc.start()
        child.close()
        self._conn = parent

    def _kill_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(5)
        self._proc, self._conn = None, None

    def generate(self, request: dict[str, Any], out_dir: Path, cancel: CancelToken) -> EngineOutput:
        import time

        cancel.raise_if_cancelled()
        self._ensure_worker()
        self._cancel_event.clear()
        cancel_sent_at = None
        try:
            self._conn.send((request, str(out_dir)))
            while True:
                if self._conn.poll(0.2):
                    reply = self._conn.recv()
                    if reply[0] == "progress":
                        self._progress(*reply[1:])
                        continue
                    break
                if cancel.is_cancelled():
                    if cancel_sent_at is None:
                        self._cancel_event.set()  # ask the pipeline to stop at the next step, keeping the model
                        cancel_sent_at = time.monotonic()
                    elif time.monotonic() - cancel_sent_at > self.CANCEL_GRACE_S:
                        self._kill_worker()  # did not stop in time: a real stop, unlike abandoning a thread
                        raise Cancelled("generation stopped; the worker did not respond and was terminated")
                if not self._proc.is_alive():
                    if self._conn.poll(0):
                        continue
                    raise EOFError
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
            code = None
            if self._proc is not None:
                self._proc.join(5)  # the pipe can close before the OS has reaped the process
                code = self._proc.exitcode
            self._proc, self._conn = None, None
            raise EngineError("engine_process_crashed", f"the image engine process {describe_exit(code)}",
                              retryable=False, details={"exit_code": code}) from None
        kind = reply[0]
        if kind == "ok":
            data = reply[1]
            return EngineOutput(images=[Path(p) for p in data["images"]], model_id=data["model_id"],
                                seed_used=data["seed_used"], metadata=dict(data["metadata"], isolated_worker=True))
        if kind == "cancelled":
            raise Cancelled("generation stopped at a step boundary; the model stays loaded")
        if kind == "engine_error":
            _, code, message, retryable, details = reply
            raise EngineError(code, message, retryable=retryable, details=details)
        raise EngineError("engine_exception", f"{reply[1]}: {reply[2]}", retryable=False)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        if self._proc is not None:
            self._proc.join(10)
        self._kill_worker()
