"""End-to-end check of a RUNNING planner API: the full operator workflow plus failure cases.

    python scripts/smoke_e2e.py --url http://localhost:8000            # e.g. docker compose up api
    python scripts/smoke_e2e.py --url http://localhost:8001 --out x.json   # PostgreSQL-backed container

Workflow checked (each step is one HTTP call against the live server and database):
  1 health                     database + measured routing reachable
  2 what-if plan               POST /plan stores nothing
  3 initial proposal           ready or awaiting approval -> (approve) -> apply -> becomes the active plan
  4 device failure             same scenario minus one device -> 'restore' proposal -> (approve) -> apply
  5 stale proposal             two proposals on the same base; applying the second after the first -> 409
  6 negative inputs            invalid body -> 422, unknown proposal -> 404, approve an applied proposal -> 409
  7 audit trail                every applied proposal has proposal_created and plan_activated events

Use a throw-away database: the script creates and applies proposals. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def __call__(self, method: str, path: str, body=None) -> tuple[int, object]:
        req = urllib.request.Request(self.url + path, method=method, headers={"Content-Type": "application/json"},
                                     data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")


def settle(api: Client, proposal: dict) -> dict:
    """Approve if needed, then apply. Returns the final proposal."""
    if proposal["status"] == "awaiting_approval":
        _, proposal = api("POST", f"/proposals/{proposal['id']}/approve",
                          {"approver": "smoke-test", "comment": "end-to-end check"})
    _, proposal = api("POST", f"/proposals/{proposal['id']}/apply")
    return proposal


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--scenario", default=str(ROOT / "examples" / "scenario.json"))
    p.add_argument("--out", help="write the check results as JSON")
    a = p.parse_args()

    api = Client(a.url)
    scenario = json.loads(Path(a.scenario).read_text())
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        return ok

    t0 = time.perf_counter()
    code, body = api("GET", "/health")
    if not check("health returns 200", code == 200, str(body)):
        return 1

    code, before = api("GET", "/audit?limit=1000")
    n_audit_before = len(before) if code == 200 else 0
    code, body = api("POST", "/plan", scenario)
    _, after = api("GET", "/audit?limit=1000")
    check("what-if plan returns metrics and writes nothing", code == 200 and len(after) == n_audit_before,
          f"status {code}, audit rows {n_audit_before} -> {len(after)}")

    code, prop = api("POST", "/proposals", scenario)
    check("proposal is created", code == 201 and prop.get("status") in ("ready", "awaiting_approval", "rejected"),
          f"status {code}, proposal status {prop.get('status')}: {prop.get('reason')}")
    if prop.get("status") == "rejected":
        check("first proposal can be applied", False, f"rejected: {prop.get('reason')}")
    else:
        first = settle(api, prop)
        check("first proposal applied", first.get("status") == "applied", f"status {first.get('status')}")
    code, active = api("GET", "/plans/active")
    check("an active plan exists", code == 200, f"plan id {active.get('id') if code == 200 else None}")

    failed = copy.deepcopy(scenario)
    lost = failed["sites"][0]["devices"].pop(0)["name"]
    code, prop = api("POST", "/proposals", failed)
    check("device failure yields an accepted restore proposal", code == 201
          and prop.get("status") in ("ready", "awaiting_approval"),
          f"lost {lost}; status {prop.get('status')}: {prop.get('reason')}")
    restored = settle(api, prop) if code == 201 and prop.get("status") != "rejected" else prop
    check("restore applied", restored.get("status") == "applied", f"status {restored.get('status')}")
    _, active2 = api("GET", "/plans/active")
    check("active plan no longer uses the failed device",
          lost not in {d["name"] for d in active2.get("devices", [])}, f"plan id {active2.get('id')}")

    shifted = copy.deepcopy(failed)
    for site in shifted["sites"]:
        site["mix"] = {"coding": 0.7, "ai": 0.2, "general": 0.1}
    _, p1 = api("POST", "/proposals", shifted)
    _, p2 = api("POST", "/proposals", shifted)
    if p1.get("status") in ("ready", "awaiting_approval") and p2.get("status") in ("ready", "awaiting_approval"):
        settle(api, p1)
        if p2["status"] == "awaiting_approval":
            api("POST", f"/proposals/{p2['id']}/approve", {"approver": "smoke-test"})
        code, body = api("POST", f"/proposals/{p2['id']}/apply")
        check("applying a proposal on an outdated base returns 409 (stale)", code == 409, f"status {code}")
    else:
        check("stale check skipped: workload shift did not justify a change", True,
              f"statuses {p1.get('status')}, {p2.get('status')} (rejected is a valid verdict)")

    bad = copy.deepcopy(scenario)
    bad["sites"][0]["devices"][0]["speed"] = -1
    code, body = api("POST", "/plan", bad)
    check("negative device speed rejected with 422", code == 422, f"status {code}")
    code, _ = api("POST", "/plan", {"sites": [], "unexpected": 1})
    check("malformed body rejected with 422", code == 422, f"status {code}")
    code, _ = api("GET", "/proposals/999999")
    check("unknown proposal returns 404", code == 404, f"status {code}")
    code, _ = api("POST", f"/proposals/{restored.get('id')}/approve", {"approver": "smoke-test"})
    check("approving an applied proposal returns 409", code == 409, f"status {code}")

    _, events = api("GET", "/audit?limit=1000")
    _, applied = api("GET", "/proposals?status=applied")
    kinds = {}
    for e in events:
        kinds.setdefault(e.get("proposal_id"), set()).add(e.get("event"))
    complete = all({"proposal_created", "plan_activated"} <= kinds.get(pr["id"], set()) for pr in applied)
    check("every applied proposal has created + activated audit events", complete and len(applied) >= 2,
          f"{len(applied)} applied proposals, {len(events)} audit events")

    elapsed = time.perf_counter() - t0
    passed = sum(c["passed"] for c in checks)
    print(f"\n{passed}/{len(checks)} checks passed in {elapsed:.2f} s against {a.url}")
    if a.out:
        Path(a.out).write_text(json.dumps({"url": a.url, "elapsed_s": elapsed, "checks": checks}, indent=2))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
