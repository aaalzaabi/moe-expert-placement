"""MoE Expert Placement Planner — AI Infrastructure Control Center.

Run from the project root:
    .venv/bin/streamlit run dashboard/app.py

The dashboard never loads the OLMoE checkpoint. It reads saved measurements and
runs lightweight, cached simulations. The original single-page dashboard is kept
as dashboard/legacy_app.py.
"""
from __future__ import annotations

import streamlit as st

import ui

st.set_page_config(page_title="MoE Control Center", page_icon=":material/hub:", layout="wide",
                   initial_sidebar_state="expanded")

pages = {
    "": [
        st.Page("app_pages/start.py", title="Start here", icon=":material/home:", default=True),
        st.Page("app_pages/demo.py", title="Presentation mode", icon=":material/slideshow:"),
        st.Page("app_pages/assistant_chat.py", title="Ask the assistant", icon=":material/support_agent:"),
    ],
    "Part 1 · Real data": [
        st.Page("app_pages/model.py", title="1. What is inside the model?", icon=":material/memory:"),
        st.Page("app_pages/routing.py", title="2. Which experts are busy?", icon=":material/route:"),
    ],
    "Part 2 · Placing experts": [
        st.Page("app_pages/placement.py", title="3. Where should experts go?", icon=":material/grid_view:"),
        st.Page("app_pages/inference.py", title="4. How fast would requests run?", icon=":material/timeline:"),
    ],
    "Part 3 · Moving experts": [
        st.Page("app_pages/migration.py", title="5. Should we move experts?", icon=":material/swap_horiz:"),
        st.Page("app_pages/network.py", title="6. Does network speed matter?", icon=":material/network_check:"),
    ],
    "Part 4 · Results": [
        st.Page("app_pages/evaluation.py", title="7. All results side by side", icon=":material/analytics:"),
        st.Page("app_pages/findings.py", title="8. What we learned and the limits", icon=":material/fact_check:"),
    ],
    "Advanced & reference": [
        st.Page("app_pages/multisite.py", title="Multi-site, forecasting & agent", icon=":material/public:"),
        st.Page("app_pages/overview.py", title="Technical summary", icon=":material/dashboard:"),
        st.Page("app_pages/provenance.py", title="Where the data comes from", icon=":material/database:"),
        st.Page("app_pages/glossary.py", title="Glossary", icon=":material/menu_book:"),
    ],
}
page = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.markdown("**MoE Expert Placement Planner**")
    st.caption("Follow the numbered pages from top to bottom. New here? Open **Start here**.")
    with st.popover("Ask a quick question", icon=":material/support_agent:", width="stretch"):
        import assistant
        quick = st.text_input("Your question", placeholder="e.g. Why was the move rejected?", key="quick_q")
        if quick:
            qa = assistant.answer(quick, ui.assistant_facts(ui.cfg_key(ui.scenario())))
            st.markdown(qa.text)
            if qa.page:
                st.page_link(qa.page[0], label=f"Open: {qa.page[1]}", icon=":material/arrow_forward:")
        st.page_link("app_pages/assistant_chat.py", label="Open the full assistant", icon=":material/chat:")
    with st.expander("What the coloured labels mean", icon=":material/label:"):
        for kind in ("measured", "checkpoint", "derived", "simulated", "live"):
            ui.badge(kind)
        st.caption("Green = real. Orange / violet = simulation with assumed hardware. "
                   "This dashboard never loads the model or touches real GPUs.")
    with st.expander("Current simulation settings", icon=":material/tune:"):
        cfg = ui.scenario()
        st.caption(ui.scenario_summary(cfg))
        st.caption("Change them on pages 3–5 under *Change the simulated setup*.")
        if cfg != ui.C.DEFAULT:
            st.button("Reset to default settings", icon=":material/restart_alt:", on_click=ui.reset_scenario,
                      width="stretch")

page.run()
