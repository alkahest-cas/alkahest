"""The Math playground's sandbox and its localhost server (``python -m alkahest.playground``)."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading

import pytest

pytest.importorskip("alkahest")

from alkahest.playground import mathlang as ml
from alkahest.playground import server, service

PAGES = "https://alkahest-cas.github.io"


@pytest.fixture
def fresh_semaphore(monkeypatch):
    monkeypatch.setattr(service, "_semaphore", None)


@pytest.fixture
def port(fresh_semaphore):
    httpd, thread = server.serve_in_thread(0)
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    hdrs = {"Host": f"127.0.0.1:{port}"}
    hdrs.update(headers or {})
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        payload = json.loads(raw) if raw else None
    except ValueError:
        payload = raw
    return resp.status, {k.lower(): v for k, v in resp.getheaders()}, payload


# ── service: the sandboxed child ─────────────────────────────────────────────


def test_run_math_computes_in_a_child(fresh_semaphore):
    res = service.run_math({"input": "d/dx x^3 sin(x)", "definitions": []})
    assert res["ok"] is True, res
    assert "cos" in res["text"]
    assert res["interpreted_latex"]


def test_input_errors_never_spawn(fresh_semaphore, monkeypatch):
    def boom():
        raise AssertionError("a malformed request must not start a process")

    monkeypatch.setattr(service, "_spawn", boom)
    res = service.run_math({"input": "__import__('os')"})
    assert res["ok"] is False
    assert res["error_source"] == "input"


def test_hard_timeout_kills_worker(fresh_semaphore, monkeypatch):
    # A deadline shorter than interpreter start-up: the worker is killed and the
    # response is a coded timeout, never a hang.
    monkeypatch.setattr(service, "HARD_TIMEOUT_S", 0.01)
    res = service.run_math({"input": "integrate sin(x)^2 dx"})
    assert res["ok"] is False
    assert res["error_code"] == ml.E_TIMEOUT
    assert res["interpreted_latex"]


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX resource limits")
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS ignores RLIMIT_AS")
def test_memory_cap_is_applied(fresh_semaphore, monkeypatch):
    # With an address-space cap too small to even import Alkahest, the child
    # must fail closed with a coded error, not take the server down.
    monkeypatch.setattr(service, "MEMORY_MB", 64)
    res = service.run_math({"input": "1+1", "definitions": []})
    assert res["ok"] is False
    assert res["error_code"] in (ml.E_MEMORY, ml.E_INTERNAL)


def test_busy_when_queue_full(monkeypatch):
    monkeypatch.setattr(service, "QUEUE_WAIT_S", 0.05)
    monkeypatch.setattr(service, "_semaphore", threading.BoundedSemaphore(1))
    service._semaphore.acquire()
    try:
        res = service.run_math({"input": "1+1"})
    finally:
        service._semaphore.release()
    assert res["error_code"] == ml.E_BUSY


def test_worker_ignores_cwd_modules(fresh_semaphore, tmp_path, monkeypatch):
    # A planted module in the server's working directory must not be imported
    # by the child (it runs with `python -I` from the package directory).
    (tmp_path / "alkahest.py").write_text("raise SystemExit('shadowed')\n")
    monkeypatch.chdir(tmp_path)
    res = service.run_math({"input": "1+1"})
    assert res["ok"] is True, res


# ── server: contract ─────────────────────────────────────────────────────────


def test_health(port):
    status, headers, body = request(port, "GET", "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["math"] is True
    assert body["local"] is True
    assert headers["content-type"] == "application/json"


def test_interpret_contract_matches_service(port):
    payload = {"input": "integrate sin(x)^2 dx", "definitions": []}
    status, _, body = request(port, "POST", "/math/interpret", payload)
    assert status == 200
    assert body == service.interpret_payload(payload)
    assert body == {
        "ok": True,
        "interpreted_latex": r"\int \sin^{2}\left(x\right)\,dx",
        "kind": "integrate",
    }


def test_math_computes_and_definitions_round_trip(port):
    _, _, d = request(port, "POST", "/math", {"input": "f = x^2 sin(x)", "definitions": []})
    assert d["ok"] is True, d
    _, _, res = request(
        port, "POST", "/math", {"input": "d/dx f", "definitions": [d["definition"]]}
    )
    assert res["ok"] is True, res
    assert "cos" in res["text"]


def test_malicious_input_refused(port):
    _, _, res = request(port, "POST", "/math", {"input": "().__class__.__base__"})
    assert res["ok"] is False
    assert res["error_code"].startswith("E-PLAYGROUND-")


def test_bad_json_and_size_limit(port):
    status, _, _ = request(port, "POST", "/math", b"not json")
    assert status == 400
    status, _, body = request(port, "POST", "/math", ["a list"])
    assert status == 200
    assert body["error_code"] == ml.E_SYNTAX
    status, _, _ = request(
        port, "POST", "/math", b"{" + b" " * (service.MAX_BODY_BYTES + 10) + b"}"
    )
    assert status == 413


def test_unknown_routes(port):
    assert request(port, "GET", "/nope")[0] == 404
    assert request(port, "POST", "/nope", {})[0] == 404
    assert request(port, "GET", "/math")[0] == 405
    assert request(port, "DELETE", "/math")[0] == 405


# ── server: browser security ─────────────────────────────────────────────────


def test_cors_for_allowed_origin(port):
    status, headers, _ = request(port, "GET", "/health", headers={"Origin": PAGES})
    assert status == 200
    assert headers["access-control-allow-origin"] == PAGES
    assert "origin" in headers["vary"].lower()


@pytest.mark.parametrize("origin", ["http://localhost:3000", "http://127.0.0.1:3000"])
def test_dev_origins_allowed(port, origin):
    status, headers, _ = request(port, "GET", "/health", headers={"Origin": origin})
    assert status == 200
    assert headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize(
    "origin", ["https://evil.example", "null", "https://alkahest-cas.github.io.evil.example"]
)
def test_foreign_origin_refused(port, origin):
    for method, path, body in (("GET", "/health", None), ("POST", "/math", {"input": "1+1"})):
        status, headers, _ = request(port, method, path, body, headers={"Origin": origin})
        assert status == 403
        assert "access-control-allow-origin" not in headers


def test_preflight_with_private_network_access(port):
    status, headers, _ = request(
        port,
        "OPTIONS",
        "/math",
        headers={
            "Origin": PAGES,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert status == 204
    assert headers["access-control-allow-origin"] == PAGES
    assert "POST" in headers["access-control-allow-methods"]
    assert "content-type" in headers["access-control-allow-headers"].lower()
    assert headers["access-control-allow-private-network"] == "true"


def test_preflight_without_pna_header_omits_it(port):
    status, headers, _ = request(
        port, "OPTIONS", "/math", headers={"Origin": PAGES, "Access-Control-Request-Method": "POST"}
    )
    assert status == 204
    assert "access-control-allow-private-network" not in headers


def test_preflight_from_foreign_origin_refused(port):
    status, headers, _ = request(
        port,
        "OPTIONS",
        "/math",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert status == 403
    assert "access-control-allow-private-network" not in headers


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "127.0.0.1.nip.io"])
def test_dns_rebinding_host_refused(port, host):
    status, _, body = request(port, "GET", "/health", headers={"Host": host})
    assert status == 403
    assert body["detail"] == "Host not allowed"


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST:8765", "[::1]:8765", "127.0.0.1"])
def test_loopback_hosts_accepted(port, host):
    assert request(port, "GET", "/health", headers={"Host": host})[0] == 200


def test_allow_origin_flag_extends_the_allowlist(fresh_semaphore):
    httpd, thread = server.serve_in_thread(0, allow_origins=["http://localhost:5173/"])
    try:
        p = httpd.server_address[1]
        status, headers, _ = request(
            p, "GET", "/health", headers={"Origin": "http://localhost:5173"}
        )
        assert status == 200
        assert headers["access-control-allow-origin"] == "http://localhost:5173"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_only_loopback_bind():
    with pytest.raises(ValueError):
        server.make_server(0, host="0.0.0.0")


def test_module_entry_point_help():
    out = subprocess.run(
        [sys.executable, "-m", "alkahest.playground", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "--port" in out.stdout
    assert "--allow-origin" in out.stdout


def test_not_reexported_from_alkahest():
    import alkahest

    assert "playground" not in getattr(alkahest, "__all__", [])
