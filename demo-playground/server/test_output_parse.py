"""Tests for Lean certificate output parsing."""

from output_parse import postprocess_outputs


def test_stdout_marker_extracted():
    text = "hello\n__AK_LEAN_CERT__{\"kind\":\"lean_certificate\",\"source\":\"import Mathlib\\n\"}\n"
    outs = postprocess_outputs([{"type": "text", "stream": "stdout", "text": text}])
    assert any(o.get("type") == "lean" for o in outs)
    lean = next(o for o in outs if o.get("type") == "lean")
    assert "import Mathlib" in lean["source"]
    text_out = next(o for o in outs if o.get("type") == "text")
    assert "hello" in text_out["text"]
    assert "__AK_LEAN_CERT__" not in text_out["text"]
