"""A localhost-only HTTP server for the Math playground (standard library only).

``python -m alkahest.playground`` starts it on ``http://127.0.0.1:8765``.  The
hosted playground at https://alkahest-cas.github.io/playground/ looks for it
there and, when it answers, sends each line of maths to it, so every
calculation runs on the visitor's own machine.

Endpoints (the same JSON contract as the demo server in
``demo-playground/server``):

* ``GET  /health``          → ``{"status": "ok", "math": true, ...}``
* ``POST /math``            ``{input, definitions}`` → result or coded refusal
* ``POST /math/interpret``  ``{input, definitions}`` → interpretation only

Security model.  The server never executes Python from a request: input is a
small maths language (:mod:`alkahest.playground.mathlang`) mapped onto a
whitelist of Alkahest calls, and each calculation runs in a fresh, resource-
limited child process (:mod:`alkahest.playground.service`).  On top of that:

* it binds the loopback interface only, so nothing off this machine can reach it;
* a request whose ``Host`` is not a loopback name is refused (403), which
  defeats DNS rebinding (a hostile page re-pointing its own domain at
  127.0.0.1 still sends its own domain as ``Host``);
* a request carrying an ``Origin`` outside the allowlist is refused (403) and
  gets no CORS headers, so other web pages can neither read answers nor make
  the server compute.  Requests without ``Origin`` come from non-browser
  programs on this machine, which could run Alkahest directly anyway.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import service

DEFAULT_PORT = 8765
BIND_HOST = "127.0.0.1"
PUBLIC_PLAYGROUND = "https://alkahest-cas.github.io/playground/"

#: Origins allowed to call the server from a browser.
DEFAULT_ALLOWED_ORIGINS = (
    "https://alkahest-cas.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

_LOOPBACK_HOST = re.compile(r"^(?:127\.0\.0\.1|localhost|\[::1\])(?::\d{1,5})?$", re.IGNORECASE)
_ALLOWED_HEADERS = "Content-Type, Authorization"
_MAX_AGE = "600"
_DEFAULT_SET = frozenset(DEFAULT_ALLOWED_ORIGINS)


def _alkahest_version() -> str:
    try:
        import alkahest

        return str(getattr(alkahest, "__version__", "unknown"))
    except Exception:
        return "unknown"


def normalize_origin(origin: str) -> str:
    """``scheme://host[:port]`` without a trailing slash, lower-cased."""
    return origin.strip().rstrip("/").lower()


def is_loopback_host(host: str | None) -> bool:
    """True when an HTTP ``Host`` header names this machine's loopback interface."""
    return host is None or bool(_LOOPBACK_HOST.match(host.strip()))


class PlaygroundHandler(BaseHTTPRequestHandler):
    """Routes ``/health``, ``/math`` and ``/math/interpret``."""

    protocol_version = "HTTP/1.1"
    server_version = "alkahest-playground"

    # ── plumbing ────────────────────────────────────────────────────────────

    def log_message(self, format: str, *args: Any) -> None:
        if not getattr(self.server, "quiet", False):
            sys.stderr.write(f"[alkahest.playground] {format % args}\n")

    def _allowed_origins(self) -> frozenset[str]:
        return getattr(self.server, "allowed_origins", _DEFAULT_SET)

    def _origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return normalize_origin(origin) if origin is not None else None

    def _origin_allowed(self) -> bool:
        origin = self._origin()
        return origin is None or origin in self._allowed_origins()

    def _cors_headers(self) -> None:
        origin = self._origin()
        if origin is not None and origin in self._allowed_origins():
            self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "").strip())
        self.send_header("Vary", "Origin")

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _discard_body(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if 0 < length <= service.MAX_BODY_BYTES:
            self.rfile.read(length)
        elif length:
            self.close_connection = True

    def _guard(self) -> bool:
        """Refuse foreign Hosts (DNS rebinding) and Origins outside the allowlist."""
        if not is_loopback_host(self.headers.get("Host")):
            self._discard_body()
            self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Host not allowed"})
            return False
        if not self._origin_allowed():
            self._discard_body()
            self._send_json(HTTPStatus.FORBIDDEN, {"detail": "Origin not allowed"})
            return False
        return True

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    # ── methods ─────────────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:
        """CORS preflight, including Chrome's Private Network Access header."""
        if not self._guard():
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", _ALLOWED_HEADERS)
        self.send_header("Access-Control-Max-Age", _MAX_AGE)
        if (self.headers.get("Access-Control-Request-Private-Network") or "").lower() == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._guard():
            return
        path = self._path()
        if path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "math": True,
                    "server": "alkahest.playground",
                    "local": True,
                    "version": _alkahest_version(),
                },
            )
        elif path in ("/math", "/math/interpret"):
            self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"detail": "Method Not Allowed"})
        elif path == "/":
            self._send_json(
                HTTPStatus.OK,
                {
                    "detail": "Alkahest Math playground server. Open "
                    f"{PUBLIC_PLAYGROUND} to use it; endpoints: "
                    "GET /health, POST /math, POST /math/interpret."
                },
            )
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})

    do_HEAD = do_GET

    def do_POST(self) -> None:
        if not self._guard():
            return
        path = self._path()
        if path not in ("/math", "/math/interpret"):
            self._discard_body()
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self.close_connection = True
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Bad Content-Length"})
            return
        if length > service.MAX_BODY_BYTES:
            self.close_connection = True
            # Drain a moderately oversized body before answering: replying while
            # the client is still sending makes some TCP stacks (macOS) reset the
            # connection, so the client never sees the 413.
            if length <= 16 * service.MAX_BODY_BYTES:
                self.rfile.read(length)
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"detail": "Request too large"})
            return
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Body must be JSON"})
            return
        if path == "/math":
            result = service.run_math(payload)
        else:
            result = service.interpret_payload(payload)
        self._send_json(HTTPStatus.OK, result)

    def _unsupported(self) -> None:
        self._discard_body()
        self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"detail": "Method Not Allowed"})

    do_PUT = do_DELETE = do_PATCH = _unsupported


class PlaygroundServer(ThreadingHTTPServer):
    daemon_threads = True
    # Do not let a second copy silently share the port (SO_REUSEADDR still lets
    # us restart immediately after a stop on POSIX).
    allow_reuse_address = sys.platform != "win32"
    #: Normalised origins whose browser requests are answered (and get CORS headers).
    allowed_origins: frozenset[str] = frozenset()
    #: Suppress the per-request log line.
    quiet = False


def make_server(
    port: int = DEFAULT_PORT,
    *,
    allow_origins: list[str] | tuple[str, ...] = (),
    quiet: bool = False,
    host: str = BIND_HOST,
) -> PlaygroundServer:
    """Build (but do not start) the server; ``port=0`` picks a free port."""
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("the playground server only binds the loopback interface")
    origins = {normalize_origin(o) for o in (*DEFAULT_ALLOWED_ORIGINS, *allow_origins) if o}
    httpd = PlaygroundServer((host, port), PlaygroundHandler)
    # Also accept the server's own origin (e.g. a page served from it in tests).
    own = f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.allowed_origins = frozenset({*origins, own, own.replace("127.0.0.1", "localhost")})
    httpd.quiet = quiet
    return httpd


def serve_in_thread(port: int = 0, **kw: Any) -> tuple[PlaygroundServer, threading.Thread]:
    """Start a server on a background thread (for tests); returns ``(server, thread)``."""
    httpd = make_server(port, quiet=kw.pop("quiet", True), **kw)
    thread = threading.Thread(target=httpd.serve_forever, name="alkahest-playground", daemon=True)
    thread.start()
    return httpd, thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m alkahest.playground",
        description=(
            "Run the Alkahest Math playground's calculation server on this machine "
            f"(127.0.0.1 only). Then open {PUBLIC_PLAYGROUND} — it finds the server "
            "automatically."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="also accept browser requests from this origin, e.g. http://localhost:5173 "
        "(repeatable)",
    )
    parser.add_argument("--quiet", action="store_true", help="do not log each request")
    parser.add_argument(
        "--open", action="store_true", help="open the hosted playground in a web browser"
    )
    args = parser.parse_args(argv)

    try:
        httpd = make_server(args.port, allow_origins=args.allow_origin, quiet=args.quiet)
    except OSError as exc:
        print(
            f"Could not listen on 127.0.0.1:{args.port} ({exc.strerror or exc}).\n"
            f"Is the playground already running? Check http://127.0.0.1:{args.port}/health, "
            "or pick another port with --port.",
            file=sys.stderr,
        )
        return 1
    port = httpd.server_address[1]
    origins = sorted(httpd.allowed_origins)
    print(
        f"Alkahest {_alkahest_version()} Math playground server — computing on this machine.\n"
        f"  Listening on http://127.0.0.1:{port} (this computer only)\n"
        f"  Open {PUBLIC_PLAYGROUND}"
        + ("" if port == DEFAULT_PORT else f"  (set the backend URL to http://127.0.0.1:{port})")
        + "\n"
        f"  Accepting browser requests from: {', '.join(origins)}\n"
        "  Press Ctrl+C to stop.",
        flush=True,
    )
    if args.open:
        import webbrowser

        webbrowser.open(PUBLIC_PLAYGROUND)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0
