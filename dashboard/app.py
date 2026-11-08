"""MoE Expert Placement Planner — AI Infrastructure Control Center.

Run from the project root:
    .venv/bin/streamlit run dashboard/app.py

The dashboard loads the OLMoE checkpoint only on 'Try your own prompt', when the user presses its button
(dashboard/live_model.py). Every other page reads saved measurements and
runs lightweight, cached simulations. The original single-page dashboard is kept
as dashboard/legacy_app.py.
"""
from __future__ import annotations

import streamlit as st

import ui

st.set_page_config(page_title="MoE Control Center", page_icon=":material/hub:", layout="wide",
                   initial_sidebar_state="auto")

# The presentation is the front door; research pages remain registered for links and tests.
st.session_state.setdefault("detailed", False)
st.session_state.setdefault("research_tools", False)
with st.sidebar:
    st.markdown("**MoE Placement Planner**")
    st.caption("Start at Welcome. Use the guided story to present.")
    st.toggle("Research tools", key="research_tools", help="Show the complete experiment and reference pages.")
research = st.session_state.research_tools or st.session_state.detailed
adv = "visible" if research else "hidden"

pages = {
    "Presentation": [
        st.Page("app_pages/welcome.py", title="Welcome", icon=":material/home:", default=True),
        st.Page("app_pages/present.py", title="Guided story", icon=":material/slideshow:"),
        st.Page("app_pages/my_prompt.py", title="Try your own prompt", icon=":material/edit_note:"),
        st.Page("app_pages/agent_visual.py", title="See the agent decide", icon=":material/smart_toy:"),
        st.Page("app_pages/assistant_chat.py", title="Ask the assistant", icon=":material/support_agent:"),
    ],
    "Research · model and placement": [
        st.Page("app_pages/start.py", title="Project overview", icon=":material/menu_book:", visibility=adv),
        st.Page("app_pages/model.py", title="Model explorer", icon=":material/memory:", visibility=adv),
        st.Page("app_pages/routing.py", title="Measured routing", icon=":material/route:", visibility=adv),
        st.Page("app_pages/placement.py", title="Placement planner", icon=":material/grid_view:", visibility=adv),
        st.Page("app_pages/gpu_sim.py", title="GPU animations", icon=":material/play_circle:", visibility=adv),
        st.Page("app_pages/inference.py", title="Inference simulator", icon=":material/timeline:", visibility=adv),
    ],
    "Research · decisions and results": [
        st.Page("app_pages/migration.py", title="Migration controller", icon=":material/swap_horiz:", visibility=adv),
        st.Page("app_pages/network.py", title="Network experiments", icon=":material/network_check:", visibility=adv),
        st.Page("app_pages/evaluation.py", title="Evaluation", icon=":material/analytics:", visibility=adv),
        st.Page("app_pages/findings.py", title="Findings and limitations", icon=":material/fact_check:", visibility=adv),
        st.Page("app_pages/multisite.py", title="Multi-site experiments", icon=":material/public:", visibility=adv),
    ],
    "Reference": [
        st.Page("app_pages/demo.py", title="Original ten-step demo", icon=":material/slideshow:", visibility=adv),
        st.Page("app_pages/overview.py", title="Technical summary", icon=":material/dashboard:", visibility=adv),
        st.Page("app_pages/provenance.py", title="Data and runs", icon=":material/database:", visibility=adv),
        st.Page("app_pages/glossary.py", title="Glossary", icon=":material/menu_book:", visibility=adv),
    ],
}
page = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    with st.popover("Ask a quick question", icon=":material/support_agent:", width="stretch"):
        import assistant
        quick = st.text_input("Your question", placeholder="e.g. Why was the move rejected?", key="quick_q")
        if quick:
            qa = assistant.answer(quick, ui.assistant_facts(ui.cfg_key(ui.scenario())))
            st.markdown(qa.text)
            if qa.page:
                st.page_link(qa.page[0], label=f"Open: {qa.page[1]}", icon=":material/arrow_forward:")
        st.page_link("app_pages/assistant_chat.py", label="Open the full assistant", icon=":material/chat:")
    with st.expander("Display and simulation settings", expanded=False):
        st.toggle("Show all details", key="detailed", help="Expand secondary charts and tables on research pages.")
        st.caption(ui.scenario_summary(ui.scenario()))
        if ui.scenario() != ui.C.DEFAULT:
            st.button("Reset simulation settings", icon=":material/restart_alt:", on_click=ui.reset_scenario,
                      width="stretch")
        st.caption("The guided story uses its fixed, labelled research scenario. Research pages use the settings above.")
    st.caption("Real routing · simulated infrastructure")

page.run()
