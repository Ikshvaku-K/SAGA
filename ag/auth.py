"""Login gate for the whole site, with two modes.

User: answer the gate questions, then the shared password. Both are checked against
salted scrypt hashes from the environment; the page never sees the correct answers.
Session: 30 days.

Demo: three fixed questions (robot? no, human? yes, sure? yes) and a first name of
1-9 letters. Session: 15 minutes, and everything it creates is temporary (see
ReportStore.purge_demo). The name lives only in the signed cookie.

The session cookie is "<payload>.<hmac>", payload = base64url JSON {r: role, n: name,
s: session id, e: expiry}, signed with AG_SESSION_SECRET.

Environment (production: /etc/saga/env, local dev: .env.local; see .env.example):
  AG_GATE_QUESTIONS  the user-mode yes/no questions, separated by "|"
  AG_GATE_HASH       scrypt hash of the answers joined with "," (lower-case yes/no)
  AG_PASSWORD_HASH   scrypt hash of the shared password
  AG_SESSION_SECRET  random key that signs session cookies (rotate it to log everyone out)
Generate all four with:  python scripts/set_password.py --questions "..." --answers "..."
The correct answers and the password never appear in code, only as hashes in the env.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time

COOKIE = "saga_session"
SESSION_DAYS = 30
DEFAULT_QUESTIONS = ["Are you a person?", "Are you here to audit sidewalks?",
                     "Is this your first visit?", "Do you agree to the privacy notice?"]
GATE_QUESTIONS = [q.strip() for q in os.environ.get("AG_GATE_QUESTIONS", "").split("|") if q.strip()] \
    or DEFAULT_QUESTIONS
DEMO_QUESTIONS = ["Are you a robot?", "Are you human?", "Sure?"]
DEMO_ANSWERS = ["no", "yes", "yes"]
DEMO_MINUTES = 15
NAME_RE = re.compile(r"[A-Za-z]{1,9}")  # letters only, fewer than 10


def hash_secret(text, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(text.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def check_secret(text, stored):
    try:
        _, salt, _ = stored.split("$")
        return hmac.compare_digest(hash_secret(text, base64.b64decode(salt)), stored)
    except Exception:
        return False


def normalize_answers(answers):
    return ",".join(str(a).strip().lower() for a in answers)


class Gate:
    def __init__(self, env=os.environ):
        self.gate_hash = env.get("AG_GATE_HASH")
        self.password_hash = env.get("AG_PASSWORD_HASH")
        secret = env.get("AG_SESSION_SECRET")
        self.secret = secret.encode() if secret else None
        self.secure_cookie = env.get("AG_SECURE_COOKIE", "0") == "1"
        self.configured = bool(self.gate_hash and self.password_hash and self.secret)

    def verify(self, answers, password):
        # check both every time so timing does not reveal which part was wrong
        ok_gate = check_secret(normalize_answers(answers), self.gate_hash or "")
        ok_pw = check_secret(password or "", self.password_hash or "")
        return ok_gate and ok_pw

    def verify_demo(self, answers, name):
        return (normalize_answers(answers) == ",".join(DEMO_ANSWERS)
                and bool(NAME_RE.fullmatch(str(name or "").strip())))

    def _sign(self, payload):
        return hmac.new(self.secret, b"v2:" + payload.encode(), hashlib.sha256).hexdigest()

    def issue(self, role="user", name=None, now=None):
        """-> (cookie value, max_age seconds)"""
        ttl = DEMO_MINUTES * 60 if role == "demo" else SESSION_DAYS * 86400
        body = {"r": role, "e": int((now or time.time()) + ttl), "s": secrets.token_hex(8)}
        if name:
            body["n"] = name
        payload = base64.urlsafe_b64encode(json.dumps(body, separators=(",", ":")).encode()).decode().rstrip("=")
        return f"{payload}.{self._sign(payload)}", ttl

    def session(self, token, now=None):
        """Decoded session dict if the cookie is authentic and unexpired, else None."""
        if not token or not self.secret or token.count(".") != 1:
            return None
        payload, sig = token.split(".")
        if not hmac.compare_digest(sig, self._sign(payload)):
            return None
        try:
            body = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except Exception:
            return None
        return body if body.get("e", 0) > (now or time.time()) else None

    def valid(self, token, now=None):
        return self.session(token, now) is not None
