"""Machine-readable agent contract tests."""

import alkahest


def test_capabilities_reports_installed_build_features():
    caps = alkahest.capabilities()

    assert caps["contract_version"] == 1
    assert {"groebner", "jit", "egraph", "parallel", "features", "primitives", "verification"} <= (
        caps.keys()
    )
    assert {
        "egraph",
        "groebner",
        "jit",
        "cranelift",
        "llvm_jit",
        "cranelift_jit",
        "parallel",
        "numpy",
        "cuda",
        "groebner_cuda",
    } == caps["features"].keys()
    assert caps["groebner"] is caps["features"]["groebner"]
    assert caps["egraph"] is caps["features"]["egraph"]
    assert caps["parallel"] is caps["features"]["parallel"]
    assert caps["features"]["jit"] is caps["features"]["llvm_jit"]
    assert caps["features"]["cranelift"] is caps["features"]["cranelift_jit"]
    assert caps["jit"] is (caps["features"]["llvm_jit"] or caps["features"]["cranelift_jit"])
    assert caps["jit"] is alkahest.jit_is_available()


def test_cranelift_jit_enables_session_jit_flag():
    caps = alkahest.capabilities()
    features = caps["features"]
    if features["cranelift_jit"]:
        assert caps["jit"] is True
        assert alkahest.jit_is_available()
        assert features["cranelift"] is True
        assert not features["llvm_jit"]

    primitives = alkahest.capabilities()["primitives"]

    assert primitives
    assert [row["name"] for row in primitives] == sorted(row["name"] for row in primitives)
    assert {
        "name",
        "simplify",
        "diff_forward",
        "diff_reverse",
        "numeric_f64",
        "numeric_ball",
        "lower_llvm",
        "lean_theorem",
    } == primitives[0].keys()


def test_capabilities_describes_verification_boundary():
    verification = alkahest.capabilities()["verification"]

    assert verification["statuses"] == [
        "lean_checked",
        "certificate_available",
        "exactly_verified",
        "numerically_checked",
        "unverified",
    ]
    assert verification["artifacts"] == {"lean4_source": True}
    assert verification["checkers"] == {"lean4": "external"}


def test_derived_result_labels_emitted_lean_source_as_unchecked_evidence():
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    result = alkahest.simplify(x + pool.integer(0))

    verification = result.verification
    assert verification["status"] == "certificate_available"
    assert verification["evidence"] == "derivation_log"
    assert verification["artifact_format"] == "lean4"
    assert verification["externally_verified"] is False
    assert isinstance(verification["side_conditions"], list)
    assert isinstance(result.certificate, str)
