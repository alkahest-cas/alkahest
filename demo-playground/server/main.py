"""FastAPI server providing Jupyter kernel sessions for the alkahest demo playground."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from kernel_manager import KernelSession
from lean_verify import lean_status, verify_lean_source

app = FastAPI(title="alkahest-demo-server")

# Allow all origins for local development / demo recording
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session registry
sessions: dict[str, KernelSession] = {}

SERVER_TOKEN = os.environ.get("ALKAHEST_SERVER_TOKEN", "").strip()


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.query_params.get("token")


def _require_token(request: Request) -> None:
    if not SERVER_TOKEN:
        return
    token = _extract_token(request)
    if token != SERVER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions), "lean": lean_status()}


@app.get("/lean/status")
async def lean_status_endpoint(request: Request):
    _require_token(request)
    return lean_status()


class VerifyLeanRequest(BaseModel):
    source: str
    timeout_sec: int = 120


@app.post("/verify-lean")
async def verify_lean(req: VerifyLeanRequest, request: Request):
    _require_token(request)
    if not req.source.strip():
        raise HTTPException(status_code=400, detail="Empty Lean source")
    return verify_lean_source(req.source, timeout_sec=req.timeout_sec)


# ── Sessions ───────────────────────────────────────────────────────────────

@app.post("/sessions")
async def create_session(request: Request):
    _require_token(request)
    session = KernelSession()
    sessions[session.id] = session
    return {"session_id": session.id}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    _require_token(request)
    session = sessions.pop(session_id, None)
    if session:
        session.shutdown()
    return {"ok": True}


# ── WebSocket streaming execution ──────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def ws_execute(websocket: WebSocket, session_id: str):
    if SERVER_TOKEN:
        token = websocket.query_params.get("token")
        if token != SERVER_TOKEN:
            await websocket.close(code=1008)
            return
    await websocket.accept()
    session = sessions.get(session_id)
    if not session:
        await websocket.send_text(json.dumps({"type": "error", "ename": "NoSession", "evalue": "Session not found", "traceback": []}))
        await websocket.close()
        return

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
        code = payload.get("code", "")

        async for msg in session.execute_streaming(code):
            await websocket.send_text(json.dumps(msg))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "ename": "ServerError", "evalue": str(e), "traceback": []}))
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Synchronous execution (for agent tool calls) ───────────────────────────

class RunRequest(BaseModel):
    code: str


@app.post("/sessions/{session_id}/run")
async def run_sync(session_id: str, req: RunRequest, request: Request):
    _require_token(request)
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    outputs = await session.execute(req.code)
    return {"outputs": outputs}


# ── Wheel install ──────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/install-wheel")
async def install_wheel(session_id: str, request: Request, wheel: Annotated[UploadFile, File()]):
    _require_token(request)
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as f:
        f.write(await wheel.read())
        tmp_path = f.name

    # pip install the wheel into the current Python environment
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", tmp_path],
        capture_output=True,
        text=True,
    )
    Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)

    # Restart the kernel so it picks up the new package
    session.shutdown()
    new_session = KernelSession()
    new_session.id = session_id
    sessions[session_id] = new_session

    return {"ok": True, "stdout": result.stdout}


# ── CLI control endpoint (used by the demo CLI) ────────────────────────────

class ControlRequest(BaseModel):
    action: str
    payload: dict = {}


@app.post("/control")
async def control(req: ControlRequest):
    """Internal control endpoint for the CLI orchestrator."""
    if req.action == "list_sessions":
        return {"sessions": list(sessions.keys())}
    if req.action == "ping":
        return {"pong": True}
    raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
