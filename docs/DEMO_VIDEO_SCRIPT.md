# Demonstration video script (8–9 minutes)

Record the screen at 1080p. Before recording: close other apps, start Docker Desktop, and run

```bash
cd ~/Desktop/MOE/moe-expert-placement
docker compose up --build -d api                    # API on :8000 (fresh SQLite volume: docker compose down -v first)
.venv/bin/streamlit run dashboard/app.py            # dashboard on :8501 (host, so the live model page works)
```

| Time | Show | Say (key points) |
|---|---|---|
| 0:00–0:45 | README top + architecture diagram (`docs/figures/architecture_implemented.png`) | Problem: where MoE experts live decides load, traffic and residency. Five layers. Real = routing + sizes; simulated = GPUs and network. |
| 0:45–1:30 | Terminal: `.venv/bin/python -m unittest discover -s tests -t .` (let it finish: 218 OK); GitHub Actions page (green runs) | Testing is the gate for all code, including AI-generated code. |
| 1:30–2:30 | Dashboard → **Try your own prompt** → short question → **Run on the real model**; then **See how the experts handled it** | Only page that loads OLMoE; real routing of your prompt; GPU replay is simulated. |
| 2:30–4:00 | **Guided story** scenes 2–4 (demand, placement animation, migration decision) | −29.4 % held-out peak load, beats all 10,000 random layouts; −29.8 % on 18 prompts the planner never saw. Migration at 12 GB/s hurts p95 (CI −19.9 to −7.5 %), helps at 100 GB/s. |
| 4:00–5:15 | **See the agent decide** → GPU failure → run → **Propose** (per-GPU chart: abudhabi-g3 empties, violet = newly copied experts); then "data must stay local" → second cycle waits → **Approve** | Observe → forecast → propose → check → approve → act. The Authoriser, not the agent, unlocks action. Zero residency violations. |
| 5:15–7:00 | Terminal: `python3 scripts/smoke_e2e.py --url http://localhost:8000`; then browser `localhost:8000/docs` and `GET /audit` | Same rules as a service: proposal → apply → device failure → restore → stale 409 → invalid 422 → audit trail. 14/14. Also verified on PostgreSQL. |
| 7:00–8:00 | `outputs/evidence/EVIDENCE.md` tables | Quantitative evidence: tests by level, replicates with CIs, planner scaling (272 ms at 16,384 experts), API p95 18.5 ms. |
| 8:00–8:45 | `docs/TRACEABILITY.md` "Not implemented" rows | Honest limits: 6 prompts only, no auth, no real GPU migration. Next steps: more prompts, GPU calibration, executor. |

Tips: zoom the terminal font to 16 pt; if the live model is slow, cut the waiting time in editing and say so.
