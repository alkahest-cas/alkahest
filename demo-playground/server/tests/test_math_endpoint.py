"""HTTP tests for the FastAPI wrapper's POST /math and /math/interpret.

The translator, executor and sandbox themselves live in ``alkahest.playground``
and are tested at the repository level (tests/test_playground_*.py); these
tests check that the demo server exposes them with the same contract.
"""

from __future__ import annotations

import pytest

pytest.importorskip("alkahest")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import main
from fastapi.testclient import TestClient
from playground_math import math_service, ml


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "SERVER_TOKEN", "")
    monkeypatch.setattr(main, "MATH_PUBLIC", False)
    monkeypatch.setattr(math_service, "_semaphore", None)
    return TestClient(main.app)


def post(client, text, definitions=None, path="/math"):
    r = client.post(path, json={"input": text, "definitions": definitions or []})
    assert r.status_code == 200, r.text
    return r.json()


EXAMPLES = [
    "integrate sin(x)^2 dx",
    "∫ x e^x dx",
    "integrate 1/(1+x^2) from 0 to oo",
    "d/dx x^3 sin(x)",
    "derivative of tan(x)",
    "diff(x^3 sin(x), x, 2)",
    "d²/dx² x^3 sin(x)",
    "solve x^2 - 5x + 6 = 0",
    "solve x^2 + y^2 = 1, x - y = 0 for x, y",
    "simplify (x^2-1)/(x-1)",
    "factor x^4 - 1",
    "expand (x+1)^5",
    "limit sin(x)/x as x -> 0",
    "series exp(x) at x = 0 to order 6",
    "sum 1/k^2 for k = 1 to oo",
    "sin(pi/6) + sqrt(8)",
    "numeric pi to 50 digits",
]


@pytest.mark.parametrize("text", EXAMPLES)
def test_examples_over_http(client, text):
    res = post(client, text)
    assert res["interpreted_latex"]
    if res["ok"]:
        assert res["latex"]
        assert res["text"]
    else:
        # the only acceptable failure is a coded decline from Alkahest itself
        assert res["error_source"] == "alkahest", res
        assert res["error_code"].startswith("E-")


def test_response_contract(client):
    res = post(client, "d/dx x^3 sin(x)")
    for key in ("ok", "latex", "text", "interpreted_latex", "certificate"):
        assert key in res
    res = post(client, "integrate e^(x^2) dx")
    for key in ("ok", "interpreted_latex", "error_code", "error_message"):
        assert key in res
    assert res["ok"] is False


def test_definitions_round_trip(client):
    d = post(client, "f = x^2 sin(x)")
    res = post(client, "d/dx f", [d["definition"]])
    assert res["ok"]
    assert "cos" in res["text"]


def test_interpret_endpoint_is_parse_only(client):
    res = post(client, "integrate sin(x)^2 dx", path="/math/interpret")
    assert res == {
        "ok": True,
        "interpreted_latex": r"\int \sin^{2}\left(x\right)\,dx",
        "kind": "integrate",
    }
    res = post(client, "2^2^2^2^2^2", path="/math/interpret")
    assert res["error_code"] == ml.E_NUMBER_TOO_LARGE


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('touch /tmp/pwned')",
        "().__class__.__base__.__subclasses__()",
        "x.__dict__",
        "2^2^2^2^2^2",
        "9^9^9^9",
        "y" * 5000,
    ],
)
def test_malicious_over_http(client, text):
    res = post(client, text)
    assert res["ok"] is False
    assert res["error_code"].startswith("E-PLAYGROUND-")


def test_body_size_limit(client):
    r = client.post("/math", content=b"{" + b" " * (math_service.MAX_BODY_BYTES + 10) + b"}")
    assert r.status_code == 413


def test_bad_json(client):
    r = client.post("/math", content=b"not json")
    assert r.status_code == 400
    r = client.post("/math", json=["a list"])
    assert r.json()["error_code"] == ml.E_SYNTAX


def test_token_required_unless_public(client, monkeypatch):
    monkeypatch.setattr(main, "SERVER_TOKEN", "s3cret")
    r = client.post("/math", json={"input": "1+1"})
    assert r.status_code == 401
    r = client.post("/math", json={"input": "1+1"}, headers={"Authorization": "Bearer s3cret"})
    assert r.json()["ok"] is True
    monkeypatch.setattr(main, "MATH_PUBLIC", True)
    r = client.post("/math", json={"input": "1+1"})
    assert r.json()["ok"] is True


def test_health_reports_math(client):
    assert client.get("/health").json()["math"] is True


def test_reuses_the_package_module():
    # One implementation: the server must not carry its own copy any more.
    assert math_service.__name__ == "alkahest.playground.service"
