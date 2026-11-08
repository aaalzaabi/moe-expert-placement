"""A focused, one-cycle-at-a-time view of the existing operations agent."""
import plotly.graph_objects as go
import streamlit as st
import ui
import ms_views as V
from moeplace import usecases as U

CASES = {
    'uc5_device_failure': ('A GPU fails', 40, 'A device in Abu Dhabi fails at the start of this demonstration.'),
    'uc4_workload_shift': ('Demand changes', 32, 'Frankfurt has recently received a surge in coding questions.'),
    'uc2_sovereign_residency': ('Data must stay local', 8, 'Abu Dhabi traffic must be served locally; changes may need a person’s approval.'),
}
STAGES = ['Observe', 'Forecast', 'Propose', 'Check', 'Act & learn']

st.caption('OPERATIONS AGENT · multi-site simulation')
st.title('See the agent decide')
st.markdown('Choose a situation, then run **one simulated hour**. Follow the highlighted steps to see why the agent acts—or keeps the current plan.')
st.caption('Site traffic is synthetic, based on measured OLMoE routing. Actions change only this session’s simulation.')
key = st.segmented_control('Situation', list(CASES), default='uc5_device_failure',
                           format_func=lambda k: CASES[k][0], key='av_case') or 'uc5_device_failure'
st.info(CASES[key][2], icon=':material/info:')

@st.cache_data(show_spinner=False)
def demand(case):
    return U.USE_CASES[case].demand(ui.routing())[0]

states = st.session_state.setdefault('av_states', {})
state = states.setdefault(key, {'cycles': 0, 'decisions': {}, 'result': None})

# Use the same agent and approval mechanism as the research page. No new decision policy is introduced here.
def replay():
    return V.run_agent_cycles(key, demand(key), state['cycles'], CASES[key][1], state['decisions'])

run, reset = st.columns([3, 1])
if run.button('Run the agent' if state['cycles'] == 0 else 'Run next simulated hour', key='av_run',
              type='primary', icon=':material/play_arrow:', width='stretch',
              disabled=bool(state['result'] and state['result']['pending']) or state['cycles'] >= 6):
    with st.spinner('Observing demand, forecasting and checking the proposal…'):
        next_cycles = state['cycles'] + 1
        # Commit the new cycle count only after a successful run.
        result = V.run_agent_cycles(key, demand(key), next_cycles, CASES[key][1], state['decisions'])
        state.update(cycles=next_cycles, result=result)
    st.rerun()  # Refresh the run button immediately if this cycle requires approval.
if reset.button('Start again', key='av_reset', width='stretch'):
    states[key] = {'cycles': 0, 'decisions': {}, 'result': None}
    st.rerun()

result = state['result']
if result is None:
    st.mermaid_chart('''flowchart LR
      O["1 · Observe demand"] --> F["2 · Forecast the next hour"] --> P["3 · Propose a placement"]
      P --> C["4 · Check the rules"] --> A["5 · Act and learn"]
    ''', alt='Five agent stages: observe, forecast, propose, check, act and learn.')
    st.markdown('**What to watch for:** a proposed move can be approved, rejected, or held for human approval. '
                'The explanation will appear immediately after the run.')
    st.page_link('app_pages/present.py', label='Back to the guided presentation', icon=':material/arrow_back:')
    st.stop()

last = result['decisions'].iloc[-1]
pending = result['pending']
if pending:
    st.warning('WAITING FOR A PERSON · the automatic checks passed, but this change needs human approval.', icon=':material/pan_tool:')
elif last.action == 'keep':
    st.info('KEEP CURRENT PLAN · no approved proposal was applied in this hour.', icon=':material/pause:')
else:
    labels = {'restore': 'RESTORE CAPACITY', 'rebalance': 'REBALANCE EXPERTS', 'consolidate': 'RELEASE DEVICES'}
    st.success(f'{labels.get(last.action, str(last.action).upper())} · applied in the simulation.', icon=':material/check_circle:')
st.caption(f"Cycle {state['cycles']} · window {result['last_t']} · next-hour forecast · deterministic authorisation")

stage = st.segmented_control('Follow the agent', STAGES, default='Check', key=f'av_stage_{key}') or 'Check'
st.progress((STAGES.index(stage) + 1) / len(STAGES), text=f'Step {STAGES.index(stage) + 1} of 5 · {stage}')
with st.container(border=True):
    if stage == 'Observe':
        st.subheader('What changed around the agent?')
        obs = V.observation(demand(key), key, result['last_t'])
        fig = go.Figure(go.Bar(x=obs['site'], y=obs['assignments in window t'], marker_color=ui.TEAL))
        fig.update_layout(xaxis_title='Site', yaxis_title='Synthetic expert assignments in this 15-minute window')
        ui.chart(fig, 'Observed synthetic expert demand by site at the current decision window', height=280)
        if key == 'uc5_device_failure':
            st.markdown('The failed device is **abudhabi-g3**. The agent must find a feasible plan for the remaining devices.')
    elif stage == 'Forecast':
        st.subheader('What does it expect next?')
        fc = result['agent'].forecaster.predict(result['last_t'], U.spec().horizon).mean(0).sum(axis=1)
        sites = [s.name for s in U.USE_CASES[key].topology.sites]
        fig = go.Figure(go.Bar(x=sites, y=fc, marker_color=ui.BLUE))
        fig.update_layout(xaxis_title='Site', yaxis_title='Forecast assignments per 15-minute window (next-hour mean)')
        ui.chart(fig, 'Site-aware forecast of mean expert assignments across the next hour', height=280)
        st.markdown('The agent forecasts each site separately. It does not assume the same experts are equally popular everywhere.')
    elif stage == 'Propose':
        st.subheader('What change does it suggest?')
        if result['proposals']:
            p = result['proposals'][-1]['proposal']
            a, b, c = st.columns(3)
            a.metric('New expert copies', int(p.migration['new_copies']), border=True)
            b.metric('Estimated transfer', f"{p.migration['bytes'] / 1e9:.2f} GB", border=True)
            c.metric('Estimated copy duration', f"{p.migration['est_time_s']:.2f} s", border=True)
            st.mermaid_chart('''flowchart LR
              A["Current expert layout"] --> B["Copy required experts"] --> C["Proposed layout"]
            ''', alt='The proposed change copies only required experts from the current to the proposed layout.')
            st.caption('Copy duration is an analytical estimate; it is not a measured transfer or the single-cluster event simulation.')
        else:
            st.markdown('No feasible new placement was proposed. The audit below contains the planner’s reason.')
    elif stage == 'Check':
        st.subheader('Why is this allowed—or rejected?')
        if not result['proposals']:
            st.warning('No candidate passed to the authoriser. The current plan stays in place; inspect the audit below for the cause.')
        for i, prop in enumerate(result['proposals']):
            if len(result['proposals']) > 1:
                st.markdown(f"**Proposal {i + 1}: {prop['proposal'].reason}**")
            for reason, passed in prop['reasons']:
                icon = ':material/check_circle:' if passed else ':material/cancel:'
                st.markdown(f"{icon} **{'PASS' if passed else 'FAIL'}** · {reason}")
                if not passed:
                    meaning, change = V.explain_check(reason)
                    st.caption(f'{meaning} What would change it: {change}')
            why = V.explain_proposal(prop)
            if why:
                st.warning(why)
        st.caption('Approval requires the existing authoriser’s rules. A required restore is still checked for overload and residency.')
    else:
        st.subheader('What actually happened?')
        st.markdown('**' + {'keep': 'The agent kept the current placement.', 'restore': 'The agent restored a feasible layout after the failure.',
                            'rebalance': 'The agent applied a new layout.', 'consolidate': 'The agent released devices while keeping its forecast below the utilisation target.'}.get(last.action, str(last.action)) + '**')
        st.markdown('**Learning:** ' + ('The previous hour’s placement regret triggered a forecaster refit.' if bool(last.refit)
                                       else 'No forecaster refit was triggered in this cycle.'))
        st.caption('Regret is checked after demand arrives. Future outcomes are not used to approve the current action.')

if pending:
    p = pending[0]
    st.markdown('**Human approval** · the proposal remains pending until you choose.')
    yes, no = st.columns(2)
    choice = None
    if yes.button('Approve this simulated change', key='av_approve', type='primary', width='stretch'):
        choice = 'approve'
    if no.button('Deny and keep the current plan', key='av_deny', width='stretch'):
        choice = 'deny'
    if choice:
        state['decisions'][f"{p['t']}:{p['proposal'].reason}"] = choice
        state['result'] = replay()
        st.rerun()

with st.expander('Evidence: tools and decision history'):
    trace = result['trace']
    ui.table(trace[trace.t == result['last_t']], 'Tools called during the latest simulated agent cycle')
    ui.table(result['audit'], 'Audit trail for these simulated cycles', height=240)

st.page_link('app_pages/present.py', label='Back to the guided presentation', icon=':material/arrow_back:')
