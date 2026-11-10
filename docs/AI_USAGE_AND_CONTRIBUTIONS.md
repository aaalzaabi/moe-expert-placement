# Contribution and AI usage declaration

## 1. Individual contributions

Individual project. **Ayesha Alzaabi** owns and is accountable for every component.

| Area | Components | Evidence |
|---|---|---|
| Problem framing and design | proposal, architecture document, ADRs, use-case definitions | `docs/proposal/`, architecture PDF, `docs/USE_CASES.md` |
| Real measurement | OLMoE loading on MPS, routing profiler, expert-size calibration, live-prompt page, expert-parallel smoke run | `scripts/profile_*.py`, `scripts/inspect_expert_memory.py`, `dashboard/live_model.py`, `scripts/run_real_expert_parallel.py` |
| Core algorithms | placement, event simulator, migration controller, multi-site planner, forecasting, regret, agent | `moeplace/` |
| Service | planner API, workflow, database schema | `service/`, `docs/SERVICE.md`, `docs/database_schema.sql` |
| Interface | Streamlit dashboard and guided story | `dashboard/` |
| Quality | 218 tests, CI, Docker packaging, evidence pipeline | `tests/`, `.github/workflows/ci.yml`, `Dockerfile`, `scripts/collect_evidence.py` |
| Evaluation and writing | experiments, results interpretation, implementation report | `outputs/`, `docs/IMPLEMENTATION_REPORT.md` |

Every commit in `github.com/aaalzaabi/moe-expert-placement` is authored by Ayesha Alzaabi.

## 2. AI-assisted development tools

| Tool | Used for | Not used for |
|---|---|---|
| Claude Code (Anthropic) — coding agent in the terminal / desktop app | drafting and refactoring Python modules and tests; writing documentation drafts; running the test suite and experiments; Docker and CI configuration; generating the evidence tables and the updated architecture diagram | deciding research questions, thresholds or which results to report; approving its own changes without tests |
| ChatGPT (OpenAI) — chat assistant | general questions, explanations of concepts, and wording suggestions | deciding research questions, thresholds or which results to report; code was not used without being run against the test suite |

The deployed system itself contains **no LLM in the decision path**: the operations agent and the API use a
deterministic Authoriser. The only neural model at run time is OLMoE, which is the object of study.

## 3. How generated code was validated

1. **Tests as the gate.** Every change was run against the full suite (now 218 tests) before it was kept. CI
   repeats this on GitHub for every push.
2. **Bit-for-bit reproduction.** When the original phase scripts were refactored into `moeplace`, reproduction
   tests required every saved CSV to be regenerated exactly. A refactor that changed any number failed.
3. **Invariant and leakage tests** written from the specification, not from the generated code: layer barriers,
   layout pinning, atomic activation, capacity, and "corrupting held-out data must not change the placement".
4. **Manual review of claims.** Generated text and labels were checked against data. Corrections made include:
   bandwidth units (GB/s, not Gbps); a comment calling the 12,582,912-byte expert size "hypothetical" when it is
   derived from the checkpoint; a dashboard that called a load proxy "GPU load"; and a scalability timing that
   had been measured in a Linux VM and must not be described as Mac hardware.
5. **Independent end-to-end checks** against the running Docker containers (`scripts/smoke_e2e.py`) on both SQLite
   and PostgreSQL, so integration is verified outside the test harness.
6. **Example caught in this phase:** after the evidence script added CSVs under `outputs/evidence/`, the
   provenance test failed because the new files had no provenance entry; the folder was then registered explicitly.

## 4. Ownership: what the author can explain and modify

| Topic | Where to look | Key points |
|---|---|---|
| Why demand-aware beats balanced | `moeplace/placement.py` `assign()` | greedy by demand, divide by speed; balanced ignores speed |
| Why migration hurts at 12 GB/s | `moeplace/simulator.py`, EVIDENCE §4 | copies share links with inference; requests arriving during the ~185 ms activation queue behind 9.76 GB of copies |
| Why the controller is conservative | `moeplace/policy.py` `decide()` | 0 % p95 rule on 3 training requests |
| Why regret, not error, triggers refits | `moeplace/forecast.py`, `moeplace/agent.py` | small error near the utilisation cap flips which replicas fit |
| How a stale proposal is prevented | `service/workflow.py` apply | `base_plan_id` must still be active → 409 |
| How one active plan is guaranteed | `service/db.py` | partial unique index on status = 'active' |
| How to add a placement strategy | `placement.assign` + `config` + tests | strategy name is validated in `SimConfig` |
