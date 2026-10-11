"""Shared-password gate: answers + password, signed session cookies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ag.auth import Gate, hash_secret


def gate():
    return Gate({"AG_GATE_HASH": hash_secret("yes,no,no,yes"),
                 "AG_PASSWORD_HASH": hash_secret("test-only-passphrase"),
                 "AG_SESSION_SECRET": "k" * 64})


def test_right_answers_and_password_only():
    g = gate()
    assert g.configured
    assert g.verify(["yes", "no", "no", "yes"], "test-only-passphrase")
    assert g.verify([" Yes", "NO", "no ", "Yes"], "test-only-passphrase")       # case/space tolerant
    assert not g.verify(["yes", "yes", "yes", "yes"], "test-only-passphrase")   # wrong answer
    assert not g.verify(["yes", "no", "no"], "test-only-passphrase")            # missing answer
    assert not g.verify(["yes", "no", "no", "yes"], "test-only-passphrasE")     # wrong password
    assert not g.verify(["yes", "no", "no", "yes"], "")


def test_session_tokens_sign_and_expire():
    g = gate()
    token, max_age = g.issue(now=1_000)
    assert max_age == 30 * 86400
    assert g.session(token, now=2_000)["r"] == "user"
    assert not g.valid(token, now=1_000 + max_age + 1)              # expired
    payload, sig = token.split(".")
    assert not g.valid(f"{payload}x.{sig}", now=2_000)               # tampered payload
    assert not g.valid(f"{payload}.{'0' * len(sig)}", now=2_000)     # tampered signature
    other = Gate({"AG_GATE_HASH": "x", "AG_PASSWORD_HASH": "y", "AG_SESSION_SECRET": "z" * 64})
    assert not other.valid(token, now=2_000)                         # rotated secret logs out


def test_demo_login_rules():
    g = gate()
    assert g.verify_demo(["no", "yes", "yes"], "Maya")
    assert g.verify_demo(["No", "YES", "yes"], "Ana")
    assert not g.verify_demo(["yes", "yes", "yes"], "Ana")           # "robot? yes" fails
    assert not g.verify_demo(["no", "yes", "no"], "Ana")             # not sure
    assert not g.verify_demo(["no", "yes", "yes"], "Alexandria")     # 10 letters: too long
    assert g.verify_demo(["no", "yes", "yes"], "Alexandri")          # 9 letters: ok
    for bad in ("", "Ana1", "Ana Lee", "José", "<b>x", None):
        assert not g.verify_demo(["no", "yes", "yes"], bad), bad


def test_demo_session_lasts_15_minutes_and_carries_name():
    g = gate()
    token, max_age = g.issue("demo", "Ana", now=1_000)
    assert max_age == 15 * 60
    s = g.session(token, now=1_000 + 14 * 60)
    assert s["r"] == "demo" and s["n"] == "Ana" and s["e"] == 1_000 + 900 and len(s["s"]) == 16
    assert g.session(token, now=1_000 + 15 * 60 + 1) is None         # logged out after 15 minutes
    t2, _ = g.issue("demo", "Ana", now=1_000)
    assert g.session(t2, now=1_001)["s"] != s["s"]                   # every demo gets its own space


def test_unconfigured_gate_rejects_everything():
    g = Gate({})
    assert not g.configured and not g.valid("123.abc") and not g.verify(["no"] * 4, "x")
