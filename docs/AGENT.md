# Placement operations agent

Code: `moeplace/agent.py`. Tests: `tests/test_agent_usecases.py`. Outputs per use case:
`agent_decisions.csv` (one row per decision with authoriser reasons and outcome), `agent_tool_trace.csv`
(every tool call with permission and latency), `agent_audit.csv` (events and explanations).

## Why an agent, and what kind

Feedback point 5 asked whether the system could be agentic. Placement operations are a continuous control
problem with goals, tools, changing state and occasional human sign-off, so an agent fits. Choosing the layout,
however, is a numeric optimisation with hard limits (capacity, residency). An LLM making that call would add
non-determinism and a prompt-injection surface without improving the answer. The design therefore separates:

| Role | Implementation | May change serving? |
|---|---|---|
| Agent (plans, uses tools, keeps state, learns) | `PlacementAgent` (rule-based reasoning) | No: it can only *propose* |
| Authoriser (decides) | `Authoriser.check()`: deterministic, testable rules | Issues the only token that unlocks `apply_placement` |
| Human approver | `approver` callable; simulated auto-approval in experiments; denial tested | Required for large or regulated changes |
| Explainer | `template_explainer`; an LLM can be plugged in here | No: it never sees the act tool |

## Loop (every hour = every H = 4 windows)

1. **Observe** events (device failures) and demand up to now.
2. **Learn:** compute the *realised* regret of the previous decision (its plan vs a perfect-foresight plan on the
   demand that actually arrived). If it is above 2%, refit the forecaster on all history.
3. **Forecast** the next hour (site-aware EWMA by default).
4. **Propose** a plan with the multi-site planner. Hysteresis (stickiness 5%) prefers devices that already hold an
   expert, which cuts weight transfers (`test_stickiness_reduces_churn`). For UC6, also propose the smallest
   consolidation (slowest devices off per site) that keeps forecast utilisation ≤ 0.6.
5. **Check** with the authoriser:
   * constraints satisfied, with zero residency violations under the forecast;
   * forecast utilisation of the new plan < 1;
   * normal re-plan: forecast objective gain ≥ 2% **and** transfer time ≤ 5% of the horizon (180 s);
   * consolidation: forecast utilisation ≤ 0.6 and transfer time ≤ 180 s;
   * mandatory change (current plan uses a lost device): allowed regardless of gain.
6. **Escalate** if approved but large: migration > 50 GB, any change to the experts hosted at a residency site,
   or any consolidation → `request_approval`.
7. **Act** through `apply_placement` with the authoriser token, or keep the current plan. If devices are off and
   consolidation no longer holds, or the plan is invalid, restore full capacity. A restore skips the gain test but
   is still checked for constraints and overload; if it fails, it is blocked and logged.
8. **Record** the decision, reasons, explanation and the realised outcome (evaluation only, never used to act).

## In the dashboard

*Multi-site, forecast & agent → Operations agent* runs **one cycle per click** (no background scheduler). Each
click replays the agent deterministically from the chosen start window. When a proposal needs a human, the page
shows **Approve / Deny (simulated)**. The choice is stored only in the browser session and fed to the agent's
`approver`, and the change is not applied while the approval is pending. Every proposal is listed with its
authoriser checks (`PlacementAgent.proposals`), plus the tool calls of the cycle and the audit history.

## Tools and permissions

| Tool | Permission | Purpose |
|---|---|---|
| observe_demand | read | demand tensor up to now |
| forecast_demand | read | site-aware forecast of the next hour |
| propose_placement | propose | run the planner |
| evaluate_plan | read | predicted utilisation, latency, residency of a plan |
| estimate_migration | read | bytes, copies and transfer time between plans (storage reload after device loss) |
| request_approval | escalate | human sign-off |
| apply_placement | act | make a plan active; raises `PermissionDenied` without the authoriser token (tested) |

**State:** current plan and topology, failed and switched-off devices, fitted forecaster, previous decision
(for realised regret), trace, audit log. **Termination:** bounded by the decision horizon and `max_steps`.
**Determinism:** same inputs → same decisions (no randomness in the loop).

## Results (simulated, `usecases-f497898bea`)

See `docs/USE_CASES.md`. In short, the agent beat a never-re-planned plan in UC1, UC2, UC4 and UC5 (for example
442 vs 1,371 after the device failure, and it never exceeded utilisation 0.9 where the static plan reached 1.19).
It obtained 13 approvals for residency-site changes in UC2 and 12 for consolidations in UC6. It made no residency
violations. Its weak point is frequent forecaster refits (12–25 per 47 decisions in the geo use cases): the 2%
regret threshold is strict for the sensitive locality objective, and a rolling-window threshold is the next step.
