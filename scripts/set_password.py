"""Print (or write) the login lines for the env file.

  python scripts/set_password.py --questions "Q1?|Q2?|Q3?" --answers yes,no,yes --out .env.local

Prompts for the shared password (hidden; or set SAGA_PASSWORD). Writes AG_GATE_QUESTIONS,
AG_GATE_HASH, AG_PASSWORD_HASH and a fresh AG_SESSION_SECRET. Only hashes are written,
never the answers or the password. A new session secret logs every device out.
"""
import argparse
import getpass
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ag.auth import hash_secret, normalize_answers

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=os.environ.get("AG_GATE_QUESTIONS"),
                help='yes/no questions separated by "|" (or env AG_GATE_QUESTIONS)')
ap.add_argument("--answers", default=os.environ.get("AG_GATE_ANSWERS"),
                help="comma-separated yes/no, one per question (or env AG_GATE_ANSWERS)")
ap.add_argument("--out", help="write/replace the lines in this env file instead of printing")
a = ap.parse_args()

if not a.questions or not a.answers:
    sys.exit("give --questions and --answers (or AG_GATE_QUESTIONS / AG_GATE_ANSWERS)")
questions = [q.strip() for q in a.questions.split("|") if q.strip()]
answers = a.answers.split(",")
if len(answers) != len(questions) or any(x.strip().lower() not in ("yes", "no") for x in answers):
    sys.exit(f"need {len(questions)} yes/no answers, one for each question")
pw = os.environ.get("SAGA_PASSWORD") or getpass.getpass("Shared password: ")
if len(pw) < 6:
    sys.exit("password too short")

lines = {
    "AG_GATE_QUESTIONS": "|".join(questions),
    "AG_GATE_HASH": hash_secret(normalize_answers(answers)),
    "AG_PASSWORD_HASH": hash_secret(pw),
    "AG_SESSION_SECRET": secrets.token_hex(32),
}
if a.out:
    p = Path(a.out)
    keep = [ln for ln in (p.read_text().splitlines() if p.exists() else []) if ln.split("=", 1)[0] not in lines]
    p.write_text("\n".join(keep + [f"{k}={v}" for k, v in lines.items()]) + "\n")
    p.chmod(0o600)
    print(f"wrote {p}")
else:
    print("\n".join(f"{k}={v}" for k, v in lines.items()))
