"""Presentation entrance: one purpose, one starting action, no experiment controls."""
import streamlit as st
import ui

st.caption('MoE Expert Placement Planner · Research demonstration')
st.title('Where should the experts live?')
st.markdown('### An AI model has 1,024 experts. Some are much busier than others.')
st.markdown('We measure which experts are used, plan where they should live on GPUs, and let an operations agent '
            'decide when a change is worth making.')

st.mermaid_chart('''flowchart LR
  Q["A question"] --> R["The model chooses experts"] --> G["We place experts on GPUs"] --> A["The agent checks a change"]
  style Q fill:#13213a,stroke:#60a5fa,color:#e2e8f0
  style R fill:#13213a,stroke:#60a5fa,color:#e2e8f0
  style G fill:#13213a,stroke:#2dd4bf,color:#e2e8f0
  style A fill:#13213a,stroke:#2dd4bf,color:#e2e8f0
''', alt='A question selects experts; the planner assigns them to GPUs; the agent checks placement changes.')

with st.container(border=True):
    st.subheader('Start with the guided story')
    st.markdown('Six short scenes. One picture and one explanation at a time. Use **Next** to present in order.')
    if st.button('Start the presentation', type='primary', icon=':material/play_arrow:', key='welcome_begin', width='stretch'):
        st.session_state.present_step = 0
        st.switch_page('app_pages/present.py')
    st.caption('About 7–10 minutes · uses saved data · no model loading')

left, right = st.columns(2)
with left:
    st.page_link('app_pages/my_prompt.py', label='Try a question yourself', icon=':material/edit_note:')
    st.caption('Run the real model when you press its button, then see your experts on the simulated GPUs.')
with right:
    st.page_link('app_pages/agent_visual.py', label='See the operations agent decide', icon=':material/smart_toy:')
    st.caption('Run a simulated hour. See what the agent observes, proposes and approves or rejects.')

st.info('Real: the OLMoE model, routing measurements and checkpoint sizes. Simulated: GPU placement, network '
        'transfers and infrastructure performance.', icon=':material/info:')
st.caption('Need the full experiments? Turn on “Research tools” in the sidebar. Help is available under “Ask the assistant”.')
