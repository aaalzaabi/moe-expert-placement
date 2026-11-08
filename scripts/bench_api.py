"""Measure response times of a RUNNING planner API (wall clock, sequential requests, client on the same machine).

    .venv/bin/uvicorn service.app:create_app --factory --port 8000     # in another terminal
    .venv/bin/python scripts/bench_api.py --url http://localhost:8000 -n 50

Writes nothing except through the API itself: it creates and applies one proposal, then creates more against it.
Use a throw-away database (MOE_DATABASE_URL) for the server you benchmark.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics as st
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(url: str, method: str, path: str, body=None) -> tuple[float, int, dict]:
    req = urllib.request.Request(url + path, method=method, headers={"Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as r:
            code, out = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        code, out = e.code, json.loads(e.read())
    return (time.perf_counter() - t0) * 1e3, code, out


def bench(label: str, fn, n: int) -> dict:
    for _ in range(3):  # warm-up
        fn()
    ts = [fn()[0] for _ in range(n)]
    row = {"endpoint": label, "n": n, "p50_ms": st.median(ts), "p95_ms": st.quantiles(ts, n=20)[18],
           "req_per_s_sequential": 1000 / st.mean(ts)}
    print(f"{label:34s} n={n}  p50={row['p50_ms']:7.1f} ms  p95={row['p95_ms']:7.1f} ms  "
          f"~{row['req_per_s_sequential']:7.1f} req/s")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--scenario", default=str(ROOT / "examples" / "scenario.json"))
    args = ap.parse_args()
    sc = json.loads(Path(args.scenario).read_text())
    try:
        call(args.url, "GET", "/health")
    except urllib.error.URLError as exc:
        print(f"API not reachable at {args.url}: {exc.reason}", file=sys.stderr)
        return 1
    bad = copy.deepcopy(sc)
    bad["sites"] = []
    rows = [bench("GET /health", lambda: call(args.url, "GET", "/health"), args.n),
            bench("POST /plan", lambda: call(args.url, "POST", "/plan", sc), args.n)]
    _, code, p = call(args.url, "POST", "/proposals", sc)
    if code in (200, 201) and p["status"] == "ready":
        call(args.url, "POST", f"/proposals/{p['id']}/apply")
    rows += [bench("POST /proposals (vs active plan)", lambda: call(args.url, "POST", "/proposals", sc), args.n),
             bench("GET /audit?limit=200", lambda: call(args.url, "GET", "/audit?limit=200"), args.n),
             bench("POST /plan invalid -> 422", lambda: call(args.url, "POST", "/plan", bad), args.n)]
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
