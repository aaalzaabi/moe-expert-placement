"""Meaningful presentation interactions, including the agent's approval gate and answer-first flow."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tests.helpers import ROOT, routing
from tests.test_dashboard import open_page
sys.path.insert(0, str(ROOT / 'dashboard'))
import live_model as LM
import streamlit as st


class TestPresentationFlow(unittest.TestCase):
    def test_start_button_opens_story_and_all_scenes_work(self):
        at = open_page('welcome')
        at.button(key='welcome_begin').click().run()
        self.assertEqual(at.session_state.present_step, 0)
        # AppTest needs an explicit page selection after a script invokes switch_page.
        at.switch_page('app_pages/present.py').run()
        for step in range(6):
            self.assertFalse(at.exception, [e.value for e in at.exception])
            self.assertEqual(at.session_state.present_step, step)
            if step < 5:
                at.button(key='present_next').click().run()
        self.assertTrue(at.button(key='present_next').disabled)
        at.button(key='present_back').click().run()
        self.assertEqual(at.session_state.present_step, 4)

    def test_failure_cycle_and_all_stages_work(self):
        at = open_page('agent_visual')
        self.assertEqual(at.session_state['av_states']['uc5_device_failure']['cycles'], 0)
        at.button(key='av_run').click().run()
        state = at.session_state['av_states']['uc5_device_failure']
        self.assertEqual(state['result']['decisions'].iloc[-1].action, 'restore')
        for stage in ['Observe', 'Forecast', 'Propose', 'Check', 'Act & learn']:
            at.segmented_control(key='av_stage_uc5_device_failure').set_value(stage).run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
        at.button(key='av_reset').click().run()
        self.assertEqual(at.session_state['av_states']['uc5_device_failure']['cycles'], 0)

    def test_human_approval_waits_and_respects_approve_or_deny(self):
        for button in ['av_approve', 'av_deny']:
            with self.subTest(button=button):
                at = open_page('agent_visual')
                at.segmented_control(key='av_case').set_value('uc2_sovereign_residency').run()
                at.button(key='av_run').click().run()
                at.button(key='av_run').click().run()
                result = at.session_state['av_states']['uc2_sovereign_residency']['result']
                self.assertTrue(result['pending'])
                self.assertEqual(result['decisions'].iloc[-1].action, 'keep')
                self.assertTrue(at.button(key='av_run').disabled)
                at.button(key=button).click().run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                result = at.session_state['av_states']['uc2_sovereign_residency']['result']
                self.assertFalse(result['pending'])
                if button == 'av_deny':
                    self.assertEqual(result['decisions'].iloc[-1].action, 'keep')
                else:
                    self.assertEqual(result['decisions'].iloc[-1].action, 'rebalance')

    def test_real_model_button_displays_answer_and_reruns_do_not_generate_again(self):
        r = routing(); r = r[r.request_id == 1].assign(request_id=LM.PROMPT_ID, category='your prompt')
        n = int(r.input_tokens.iloc[0])
        routed = LM.Routed('What is the capital of France?', ['token'] * n, r,
                           np.zeros((16, n, 8), dtype=int), 0.5, 'mps', 'bfloat16')
        answer = LM.Generated(routed.prompt, 'The capital of France is Paris.', 8, 1.0, False)
        st.cache_resource.clear()
        try:
            with patch.object(LM, 'checkpoint_present', return_value=True), \
                 patch.object(LM, 'load', return_value=SimpleNamespace(device='mps')), \
                 patch.object(LM, 'generate', return_value=answer) as gen, \
                 patch.object(LM, 'route', return_value=routed) as route:
                at = open_page('my_prompt')
                self.assertFalse(gen.called)
                at.text_area(key='mp_text').set_value(routed.prompt)
                next(b for b in at.button if b.label == 'Run on the real model').click().run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertTrue(any('capital of France is Paris' in m.value for m in at.markdown))
                self.assertFalse(at.toggle(key='mp_explore').value)
                self.assertEqual(len(at.metric), 0)
                at.toggle(key='mp_explore').set_value(True).run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                gen.assert_called_once()
                route.assert_called_once()
                self.assertGreater(len(at.metric), 0)
        finally:
            st.cache_resource.clear()

    def test_generation_failure_is_visible_and_does_not_reuse_old_answer(self):
        st.cache_resource.clear()
        try:
            with patch.object(LM, 'checkpoint_present', return_value=True), \
                 patch.object(LM, 'load', return_value=SimpleNamespace(device='mps')), \
                 patch.object(LM, 'generate', side_effect=RuntimeError('test generation failure')):
                at = open_page('my_prompt')
                at.session_state['mp_answer'] = LM.Generated('old question', 'old answer', 2, 1.0, False)
                at.text_area(key='mp_text').set_value('new question')
                next(b for b in at.button if b.label == 'Run on the real model').click().run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertTrue(any('test generation failure' in e.value for e in at.error))
                self.assertNotIn('mp_answer', at.session_state)
        finally:
            st.cache_resource.clear()


if __name__ == '__main__':
    unittest.main()
