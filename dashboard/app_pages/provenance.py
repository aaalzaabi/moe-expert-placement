import pandas as pd
import streamlit as st

import ui
from moeplace import experiment as E
from moeplace.paths import MEASUREMENTS
from moeplace.registry import ARTIFACTS, KIND_LABEL

ui.page_header("Where the data comes from",
               "Every saved result file, whether it is real or simulated, and which script produced it.")
ui.explain(
    "A reference list of all saved result files. For each: whether it is a **real measurement**, read from the "
    "**model files**, **calculated** from real data, or a **simulation**, plus the script and settings that made it.",
    ["Use the buttons at the top to show only one kind of file.",
     "**Bottom left:** new simulation runs you created with `scripts/run_experiment.py`.",
     "**Bottom right:** new real measurement runs (none until you run `scripts/profile_routing.py`)."],
    "Every simulated number in this dashboard can be traced back to a file and regenerated exactly.", expanded=False)

kinds = st.pills("Show", list(KIND_LABEL), selection_mode="multi", default=list(KIND_LABEL),
                 format_func=KIND_LABEL.get, key="pv_kinds")
rows = [{"file": f"outputs/{a.path}", "kind": KIND_LABEL[a.kind], "phase": a.phase, "producer": a.producer,
         "parameters": a.parameters, "notes": a.notes, "present": a.exists} for a in ARTIFACTS if a.kind in (kinds or [])]
ui.table(pd.DataFrame(rows), "Provenance of saved output files", height=520,
         column_config={"present": st.column_config.CheckboxColumn("on disk"),
                        "parameters": st.column_config.TextColumn(width="large"),
                        "notes": st.column_config.TextColumn(width="large")})

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        ui.section("New simulation runs", "simulated",
                   caption="Created by scripts/run_experiment.py in outputs/runs/ — each with metadata.json.")
        runs = E.list_runs()
        if runs.empty:
            st.caption("No runs yet. Try: `.venv/bin/python scripts/run_experiment.py --bandwidth 3 12 25 100`")
        else:
            ui.table(runs.drop(columns=["path"]), "Simulation runs on disk")
with c2:
    with st.container(border=True):
        ui.section("New measurement runs", "measured",
                   caption="Created explicitly by scripts/profile_routing.py (loads OLMoE). Never run by the dashboard.")
        found = sorted(p.name for p in MEASUREMENTS.iterdir()) if MEASUREMENTS.exists() else []
        if found:
            ui.table(pd.DataFrame({"run": found}), "Measurement runs on disk")
        else:
            st.caption("None yet. `.venv/bin/python scripts/profile_routing.py --dry-run` validates the setup without loading the model.")
