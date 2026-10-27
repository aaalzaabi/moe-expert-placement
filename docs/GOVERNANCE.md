# AI governance

This responds to feedback point 8. It is an engineering governance design, **not legal advice**; obligations
must be confirmed with the organisation's legal and compliance teams for the actual deployment and data.

## 1. What is being governed

The planner and agent do not generate text and do not see prompts. They decide **where the experts of a served
model run** and therefore **where prompt-derived data travels**: each token's hidden state (a 2,048-number
vector derived from the user's prompt) is sent to whichever site hosts the selected expert. Because hidden states
are derived from prompts that may contain personal or sensitive data, this design treats them as carrying the
same obligations as the prompt itself. The planner's governance risk is **data movement, availability and
unsupervised operational change**, not content generation.

## 2. Frameworks used

| Framework | Used for | Source |
|---|---|---|
| NIST AI RMF 1.0 (NIST AI 100-1, Jan 2023): GOVERN, MAP, MEASURE, MANAGE | Organising the risk register and controls below | nist.gov/itl/ai-risk-management-framework |
| ISO/IEC 42001:2023, AI management system | Management-system wrapper: roles, change control, internal audit of the agent | iso.org/standard/42001 |
| ISO/IEC 23894:2023, AI risk management guidance | Risk identification and treatment method | iso.org/standard/77304.html |
| UAE Charter for the Development and Use of AI (2024) | Principles: safety, privacy, human oversight, accountability | u.ae (UAE government portal) |

Frameworks are domain-agnostic. **Domain-specific obligations come from the use case and the data** (section 3).

## 3. Compliance by use case

| Use case | Data and actors | Obligations that may apply | How the system supports them |
|---|---|---|---|
| UC1 Multi-region assistant | Enterprise users in UAE, EU, APAC; prompts may contain personal data | **UAE PDPL** (Federal Decree-Law 45/2021): transfers abroad under Art. 22 (adequate jurisdictions) or Art. 23 (contract, consent and other bases). **EU GDPR Chapter V** (Arts 44–49) for EU personal data leaving the EU/EEA. Singapore's PDPA for APAC users (not analysed here). | Cross-site traffic is measured per site pair (`wan_bytes`, `remote_share`); sites whose traffic must not leave can be marked `residency=True`; every plan change is logged. |
| UC2 Sovereign / regulated tenant | Government or health workloads in the UAE | **UAE Federal Law No. 2 of 2019** (ICT in health fields): health data may not be stored, processed or transferred outside the UAE unless approved by a health-authority or Ministry resolution. The PDPL excludes health and government data, which are covered by their own laws. Free zones (DIFC, ADGM) have separate data-protection regimes. | **Hard residency constraint** in the planner: the site hosts a copy of every expert and serves all its own demand (0 violations by construction, tested under unforeseen demand); any change to that site's experts requires human approval. |
| UC3 Single cluster | Internal platform | Internal change management | Release gates, reproducible runs, rollback |
| UC4 Workload shift | Same as UC1 | Model-risk monitoring (NIST MEASURE/MANAGE) | Regret-based drift detection and refits, logged |
| UC5 Device failure | Same as UC1 | Business continuity; incident reporting under the organisation's own SLAs | Forced restore, storage-reload cost estimated, audit trail |
| UC6 Off-peak consolidation | Same as UC1 | Energy/cost reporting accuracy; capacity changes approved by a human | Every consolidation escalated; GPU-hours released recorded per decision |

**EU AI Act** (Regulation (EU) 2024/1689). General-purpose-AI obligations apply from 2 August 2025 and the Act
applies generally from 2 August 2026. The Digital Omnibus (Regulation (EU) 2026/1744) moved high-risk obligations
to 2 December 2027 (Annex III) and 2 August 2028 (Annex I). A serving-infrastructure optimiser is not itself an
Annex III use. Obligations attach to the model provider and to the application that uses the model (for example,
a health or public-services use in UC2). The planner must not undermine the logging, robustness and human-oversight
measures those applications rely on.

## 4. Risk register (NIST AI RMF functions)

| ID | Risk | Function | Likelihood / impact | Control (implemented → proposed) |
|---|---|---|---|---|
| R1 | Prompt-derived activations cross a border without a lawful basis | MAP / MANAGE | Medium / High | Residency constraint; per-site-pair traffic metrics → policy file of allowed site pairs per tenant |
| R2 | Automated migration moves a regulated tenant's experts off-site | MANAGE | Low / High | Planner rejects plans with residency violations; authoriser requires 0 violations; human approval for residency-site changes |
| R3 | Forecast error causes overload (utilisation > 1) | MEASURE | Medium / High | Utilisation cap; authoriser rejects plans with forecast utilisation ≥ 1; regret monitoring |
| R4 | Silent drift: forecast looks accurate but decisions are poor | MEASURE | Medium / Medium | Regret-based (not error-based) drift detection; refit; decision log |
| R5 | Agent exceeds its authority | GOVERN | Low / High | Permission levels per tool; act tool locked behind the authoriser token (tested); approval thresholds; no LLM in the decision path |
| R6 | Unequal service quality across regions (one region gets lower latency) | MAP | Medium / Medium | Per-site latency reported → fairness threshold on the latency gap between sites (proposed) |
| R7 | Tampered or corrupted expert weights during migration | MANAGE | Low / High | Checksums per copied expert; mTLS between planner and executor (proposed, Section K of the architecture) |
| R8 | Wrong cost/energy savings claimed | GOVERN | Medium / Low | Savings computed per decision and labelled simulated until measured |
| R9 | Results presented as real GPU measurements | GOVERN | Medium / Medium | Provenance labels on every output and dashboard page (`registry.py`, `metadata.json`) |

## 5. Governance roles

* **Model/platform owner** (accountable): approves thresholds (`AgentConfig`), utilisation cap and residency policy.
* **Compliance officer:** owns the residency and transfer policy per tenant; reviews the approval log monthly.
* **SRE on call:** receives escalations; can freeze the agent (no-act mode).
* **Internal audit (ISO/IEC 42001):** samples decisions from `agent_decisions.csv` and checks the reasons against the rules.

## Sources

* NIST, *AI Risk Management Framework 1.0*, NIST AI 100-1 (2023): https://www.nist.gov/itl/ai-risk-management-framework
* ISO/IEC 42001:2023: https://www.iso.org/standard/42001
* ISO/IEC 23894:2023: https://www.iso.org/standard/77304.html
* UAE Charter for the Development and Use of AI: https://u.ae/en/about-the-uae/strategies-initiatives-and-awards/policies/Ai/The-UAE-Charter-for-the-Development-and-Use-of-Artificial-Intelligence
* UAE PDPL cross-border rules (Arts 22–23) and exclusions: https://www.trade.gov/market-intelligence/united-arab-emirates-allows-cross-border-data-flows-personal-data
* UAE Federal Law No. 2 of 2019 (health ICT) data localisation: https://tamimi.com/law-update-articles/the-federal-law-regulating-the-use-of-information-and-communication-technology-in-the-uae-healthcare-sector
* EU AI Act timeline and Digital Omnibus (Regulation (EU) 2026/1744): https://brightdefense.com/news/eu-ai-act-delay-keeps-2026-compliance-pressure
* EU AI Act text: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
