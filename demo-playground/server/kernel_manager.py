"""Wraps jupyter_client to manage kernel sessions."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import jupyter_client

from output_parse import classify_rich, postprocess_outputs

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent.parent
_KERNEL_INIT_CODE = f"""
import sys
_p = {repr(str(SERVER_DIR))}
if _p not in sys.path:
    sys.path.insert(0, _p)
try:
    from playground_helpers import display_lean_cert, emit_lean_marker
except ImportError:
    pass
del _p
"""


_PREFERRED_KERNEL = "alkahest-dev"


def _alkahest_libs_dir() -> Path | None:
    """Native libs bundled with the +full wheel in the repo .venv."""
    libs = (
        REPO_ROOT
        / ".venv"
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/alkahest.libs"
    )
    return libs if libs.is_dir() else None


def _kernel_env() -> dict[str, str]:
    """Kernel env: drop dev PYTHONPATH (shadows +full JIT) and set LD_LIBRARY_PATH."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    libs = _alkahest_libs_dir()
    if libs:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{libs}:{prev}" if prev else str(libs)
    return env


class KernelSession:
    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        # Prefer the alkahest-dev kernelspec (main venv with 2.2.x + LD_LIBRARY_PATH).
        # Fall back to the default kernel if not registered.
        try:
            ks = jupyter_client.kernelspec.KernelSpecManager()
            available = list(ks.get_all_specs().keys())
            kernel_name = _PREFERRED_KERNEL if _PREFERRED_KERNEL in available else None
        except Exception:
            kernel_name = None
        km_kwargs: dict[str, Any] = {"env": _kernel_env()}
        if kernel_name:
            km_kwargs["kernel_name"] = kernel_name
        self.km = jupyter_client.KernelManager(**km_kwargs)
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=30)
        self._helpers_loaded = False

    def shutdown(self) -> None:
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass

    def _ensure_helpers(self) -> None:
        if self._helpers_loaded:
            return
        # Set flag first to prevent recursion if _raw_execute calls back here.
        self._helpers_loaded = True
        self._raw_execute(_KERNEL_INIT_CODE)

    async def execute(self, code: str) -> list[dict[str, Any]]:
        """Execute code synchronously and return all outputs."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sync, code)

    def _execute_sync(self, code: str) -> list[dict[str, Any]]:
        self._ensure_helpers()
        return self._raw_execute(code)

    def _raw_execute(self, code: str) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        self.kc.execute(code)

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=60)
            except Exception:
                break

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                outputs.append(
                    {"type": "text", "stream": content.get("name", "stdout"), "text": content["text"]}
                )

            elif msg_type in ("display_data", "execute_result"):
                data: dict[str, str] = content.get("data", {})
                out = classify_rich(data)
                if out:
                    outputs.append(out)

            elif msg_type == "error":
                outputs.append(
                    {
                        "type": "error",
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    }
                )

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        return postprocess_outputs(outputs)

    async def execute_streaming(self, code: str):
        """Async generator that yields output dicts as they arrive."""
        self._ensure_helpers()
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def _run():
            exec_count = 0
            self.kc.execute(code)
            while True:
                try:
                    msg = self.kc.get_iopub_msg(timeout=60)
                except Exception:
                    break

                msg_type = msg["msg_type"]
                content = msg["content"]

                if msg_type == "stream":
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"type": "stream", "name": content.get("name", "stdout"), "text": content["text"]},
                    )

                elif msg_type in ("display_data", "execute_result"):
                    data = content.get("data", {})
                    out = classify_rich(data)
                    if out:
                        if msg_type == "execute_result":
                            exec_count = content.get("execution_count", 0)
                        loop.call_soon_threadsafe(queue.put_nowait, out)

                elif msg_type == "error":
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "error",
                            "ename": content.get("ename", "Error"),
                            "evalue": content.get("evalue", ""),
                            "traceback": content.get("traceback", []),
                        },
                    )

                elif msg_type == "status":
                    if content.get("execution_state") == "idle":
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"type": "done", "execution_count": exec_count},
                        )
                        break

            loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _run)

        pending_text: list[str] = []

        while True:
            item = await queue.get()
            if item is None:
                if pending_text:
                    combined = "".join(pending_text)
                    for out in postprocess_outputs(
                        [{"type": "text", "stream": "stdout", "text": combined}]
                    ):
                        yield out
                break

            if item.get("type") == "stream":
                pending_text.append(item.get("text") or "")
                continue

            if pending_text:
                combined = "".join(pending_text)
                pending_text.clear()
                for out in postprocess_outputs(
                    [{"type": "text", "stream": "stdout", "text": combined}]
                ):
                    yield out

            if item.get("type") == "done":
                yield item
            else:
                yield item
