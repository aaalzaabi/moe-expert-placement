"""Shared core for the MoE Expert Placement Planner.

Modules
-------
paths      project-root resolution (works from any working directory)
config     centralised, validated simulation / hardware / workload settings
data       CSV loaders with schema validation and provenance labels
placement  static / balanced / demand-aware expert placement + metrics
simulator  discrete-event inference + migration simulator (Phase 4A-4F logic)
policy     cost-aware migration controller and migration state machine
experiment reproducible experiment runner (run ids, metadata, sweeps)
registry   provenance of every saved output file (measured vs simulated)
multisite  multi-site planner: sites, replicas, capacity/memory/residency, utilisation + latency model
workload   site-aware demand tensor Y[t, s, e] from measured routing profiles + synthetic traffic
forecast   demand forecasters, placement regret, decision-aware drift
usecases   six use cases (users, KPIs, governance focus, scenarios)
agent      placement operations agent with tools, permissions, authoriser and approvals
scalability planner timing sweep and optimality gap

Nothing in this package imports torch or loads the OLMoE checkpoint.
"""

__version__ = "0.3.0"
