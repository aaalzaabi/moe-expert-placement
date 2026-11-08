"""One picture per scene, based on existing saved evidence and the existing placement visualiser."""
import plotly.graph_objects as go
import streamlit as st
import ui
import gpu_viz as G
import ms_views as V

SCENES = ['Meet the experts', 'See the uneven demand', 'Place them on GPUs',
          'Check the cost of moving', 'Meet the operations agent', 'What did we learn?']
st.session_state.setdefault('present_step', 0)
step = int(st.session_state.present_step)

def move(where):
    st.session_state.present_step = max(0, min(len(SCENES) - 1, where))

st.caption('GUIDED PRESENTATION · fixed four-GPU research scenario')
left, middle, right = st.columns([1, 4, 1], vertical_alignment='center')
left.button('Back', key='present_back', disabled=step == 0, on_click=move, args=(step - 1,), width='stretch')
middle.progress((step + 1) / len(SCENES), text=f'Scene {step + 1} of {len(SCENES)} · {SCENES[step]}')
right.button('Next', key='present_next', type='primary', disabled=step == len(SCENES) - 1,
             on_click=move, args=(step + 1,), width='stretch')
st.title(SCENES[step])
held = ui.csv('heldout_comparison.csv').set_index('strategy')
cut = ui.pct(held.loc['demand_aware', 'peak_test_load'], held.loc['static', 'peak_test_load'])
notes = ''
details = None

if step == 0:
    st.markdown('An **expert** is a small neural network inside the model. A **router** chooses which experts handle each token (a word piece).')
    ui.badge('checkpoint')
    st.mermaid_chart('''flowchart LR
      T["One token"] --> R{"Router"}
      R --> E["8 chosen experts out of 64"] --> S["Combine their answers"] --> N["Next layer"]
    ''', alt='Each token is sent to eight of sixty-four experts; their outputs are combined before the next layer.')
    a, b, c = st.columns(3)
    a.metric('Layers in the model', 16, border=True)
    b.metric('Experts in each layer', 64, border=True)
    c.metric('Experts placed in total', '1,024', border=True)
    takeaway = 'Every expert needs a home. Only some experts work on a particular token.'
    notes = 'OLMoE is the real pretrained model. Explain one layer first, then say this repeats sixteen times. '
    notes += 'Expert (layer 0, expert 3) and expert (layer 1, expert 3) are different networks.'
    details = ('app_pages/model.py', 'Inspect the model')
elif step == 1:
    st.markdown('We ran **six real questions** through OLMoE. The experts were not used equally.')
    ui.badge('measured')
    d = ui.D.demand(ui.routing()).unstack().loc[0]
    fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=ui.TEAL))
    fig.update_layout(xaxis_title='Expert number in layer 0', yaxis_title='Times selected across six questions')
    ui.chart(fig, 'Measured expert selections in layer zero across six prompts', height=320)
    takeaway = 'Giving every GPU the same number of experts does not necessarily give it the same amount of work.'
    notes = 'These are recorded router choices, not invented traffic. Some experts are busy, others are quiet. '
    notes += 'We use three questions to plan and hold back three to check the plan. The sample remains small.'
    details = ('app_pages/routing.py', 'Explore the measured data')
elif step == 2:
    st.markdown('Each small square is **one expert**. Each large group is a **simulated GPU**. Press **Play** to watch all 1,024 experts being placed.')
    ui.badge('simulated')
    method = st.segmented_control('Compare two methods', ['static', 'demand_aware'], default='demand_aware',
                                  format_func=lambda x: 'Simple placement' if x == 'static' else 'Demand-aware placement', key='present_method') or 'demand_aware'
    pl = ui.plan(ui.cfg_key(ui.C.DEFAULT))
    tr = G.placement_trace(pl, method, ui.C.DEFAULT)
    ui.chart(G.fig_placement_animation(tr, ui.C.DEFAULT, ui.STRATEGY_LABEL[method]),
             'All 1024 experts assigned to four simulated GPUs, with estimated work per GPU', height=450,
             key=f'present_gpu_{method}')
    takeaway = f'On the three held-out questions, demand-aware placement reduces the busiest GPU’s load proxy by {cut:.1f}%.'
    notes = 'The picture uses training demand to construct the plan; the headline improvement is checked on held-out questions. '
    notes += 'Fast GPUs can handle more work. Speeds 1.0, 1.3, 0.8 and 1.6 are assumptions. These bars show a load proxy, not measured utilisation or milliseconds.'
    details = ('app_pages/gpu_sim.py', 'Explore the GPUs and question animation')
elif step == 3:
    st.markdown('A better layout is not free: experts must be copied while requests are still running.')
    ui.badge('simulated')
    ex = ui.explain_decision(ui.cfg_key(ui.C.PRESETS['phase4e_legacy']))
    a, b = st.columns([1, 2])
    with a:
        st.metric('Experts that must move', '776', border=True)
        st.caption('Saved controller experiment: 8 MB per expert, 12 GB/s. The later checkpoint-calibrated experiment uses 12.583 MB per expert.')
    with b:
        st.error(ex.headline, icon=':material/cancel:')
        for check in ex.checks:
            mark = 'PASS' if check['passed'] is True else 'FAIL' if check['passed'] is False else 'NOT CHECKED'
            st.markdown(f"**{mark} · {check['check']}**  \n{check['meaning']}")
    takeaway = 'Rejecting a move can be the right decision: protect the running requests, even if the final layout is better.'
    notes = 'The controller uses only training requests. In this saved scenario it forecasts about a 4% total-time gain, '
    notes += 'but the p95 gets about 33% worse during copying. The limit is 0%. Do not mix this 8 MB run with checkpoint-size results.'
    details = ('app_pages/migration.py', 'Inspect the decision and its causes')
elif step == 4:
    st.markdown('The **operations agent** repeats this loop as demand changes. A person approves certain large or regulated changes.')
    ui.badge('simulated')
    st.mermaid_chart('''flowchart LR
      O["Observe"] --> F["Forecast"] --> P["Propose"] --> C{"Check rules"}
      C -->|pass| H{"Human needed?"}
      H -->|no| A["Apply in simulation"]
      H -->|yes| W["Wait for approval"] --> A
      C -->|fail| K["Keep current plan + reason"]
      A --> L["Measure outcome and learn"] --> O
    ''', alt='Operations agent observes, forecasts, proposes, checks, seeks approval when needed and learns from outcomes.')
    run = V.latest_run('usecases')
    dec, missing = V.load_csv(run, 'uc5_device_failure/agent_decisions.csv', 'run_usecases.py')
    if dec is not None and not dec.empty and (dec.action == 'restore').any():
        row = dec[dec.action == 'restore'].iloc[0]
        with st.container(border=True):
            st.markdown('**Recorded example: a GPU fails in Abu Dhabi**')
            st.markdown(f'The agent detected the failed device and restored a placement using **{int(row.new_copies)} new copies** '
                        f'({row.migration_gb:.2f} GB of modelled transfers).')
            st.caption('Source: saved device-failure simulation. This is a recorded outcome, not a live hardware incident.')
    st.page_link('app_pages/agent_visual.py', label='Open the agent demonstration', icon=':material/smart_toy:')
    takeaway = 'The agent proposes and learns; fixed checks control its actions. The assistant chatbot is a separate help tool.'
    notes = 'This is a rule-based agent, not an LLM controlling GPUs. Demand across sites is synthetic over measured routing profiles. '
    notes += 'The next page lets you run its existing simulation one hour at a time.'
    details = ('app_pages/multisite.py', 'Explore the full multi-site experiment')
else:
    st.markdown('The project shows how measured routing can guide placement—and why moving experts needs a separate decision.')
    fig = go.Figure(go.Bar(x=['Simple', 'Load-balanced', 'Demand-aware'],
                          y=[held.loc[s, 'peak_test_load'] for s in ui.C.STRATEGIES],
                          marker_color=[ui.SLATE, ui.BLUE, ui.TEAL], texttemplate='%{y:.0f}', textposition='outside'))
    fig.update_layout(yaxis_title='Busiest GPU’s held-out load proxy (lower is better)', yaxis_range=[0, 1050])
    ui.chart(fig, 'Peak simulated GPU load for three placement strategies on held-out requests', height=300)
    left, right = st.columns(2)
    with left:
        st.markdown('**What we measured**  \nReal expert routing, single-device forward-pass timings and checkpoint sizes.')
    with right:
        st.markdown('**What still needs validation**  \nMore real prompts and execution on physical multi-GPU infrastructure.')
    takeaway = f'{cut:.1f}% lower peak load in this scenario. It is a research result under assumed hardware—not a measured GPU speedup.'
    notes = 'Only six short prompts, three held out, support the original results. No confidence intervals or real multi-GPU speedup claim. '
    notes += 'The professor agreed a planner and simulator scope; real deployment is future validation.'
    st.page_link('app_pages/my_prompt.py', label='Optional live demonstration: try your own prompt', icon=':material/edit_note:')
    details = ('app_pages/findings.py', 'Read the findings and limitations')

st.success(takeaway, icon=':material/lightbulb:')
with st.expander('Presenter notes — what to say', icon=':material/record_voice_over:'):
    st.markdown(notes)
    if details:
        st.page_link(details[0], label=details[1], icon=':material/open_in_new:')
with st.expander('Jump to a scene', icon=':material/list:'):
    target = st.selectbox('Scene', range(len(SCENES)), index=step,
                          format_func=lambda n: f'{n + 1}. {SCENES[n]}', key=f'present_jump_{step}')
    if st.button('Go to scene', key='present_jump_go'):
        move(target)
        st.rerun()
