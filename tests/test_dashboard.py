"""Dashboard import, smoke and interaction tests (headless, via streamlit.testing.AppTest)."""
import logging
import sys
import unittest

from tests.helpers import ROOT

from streamlit.testing.v1 import AppTest

logging.getLogger("streamlit").setLevel(logging.ERROR)
APP = str(ROOT / "dashboard" / "app.py")
PAGES = ["start", "glossary", "assistant_chat", "overview", "demo", "model", "routing", "placement", "inference", "migration", "evaluation", "network",
         "findings", "provenance", "multisite"]


def open_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    return at.switch_page(f"app_pages/{page}.py").run()


class TestDashboardSmoke(unittest.TestCase):
    def test_entrypoint_renders_default_page(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.title[0].value, "Start here")

    def test_every_page_renders_without_exception(self):
        for page in PAGES:
            with self.subTest(page=page):
                at = open_page(page)
                self.assertFalse(at.exception, [str(e.value)[:400] for e in at.exception])
                self.assertGreater(len(at.title), 0)

    def test_dashboard_never_imports_torch(self):
        for page in PAGES:
            open_page(page)
        self.assertNotIn("torch", sys.modules, "dashboard must not import torch / load the model")

    def test_no_deprecated_use_container_width(self):
        for path in (ROOT / "dashboard").rglob("*.py"):
            self.assertNotIn("use_container_width", path.read_text(), str(path))


class TestDashboardInteractions(unittest.TestCase):
    def test_demo_walks_all_ten_steps(self):
        at = open_page("demo")
        for i in range(9):
            at.button[1].click().run()
            self.assertFalse(at.exception, f"step {i + 2}: {[e.value for e in at.exception]}")
        self.assertEqual(at.session_state.demo_step, 9)
        self.assertTrue(at.button[1].disabled)

    def test_migration_sandbox_follow_controller_rejects(self):
        at = open_page("migration")
        at.button[0].click().run()  # follow controller (default scenario recommends reject)
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.mig_sm.state, "rejected")

    def test_migration_sandbox_override_executes_and_activates(self):
        at = open_page("migration")
        at.button[1].click().run()  # approve / override
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.session_state.mig_sm.state, "active")
        self.assertTrue(any("Simulated execution" in s.value for s in at.subheader))

    def test_migration_fault_injection_fails_and_rolls_back(self):
        at = open_page("migration")
        at.toggle(key="mg_fail_on").set_value(True).run()
        at.button[1].click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.session_state.mig_sm.state, "failed")
        self.assertTrue(any("rollback" in e.value for e in at.error))

    def test_placement_planner_reconfigures_cluster(self):
        at = open_page("placement")
        at.number_input(key="pl_n").set_value(6).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(len(at.session_state.scenario["gpu_speeds"]), 6)

    def test_evaluation_live_mode(self):
        at = open_page("evaluation")
        at.segmented_control(key="ev_view").set_value("live").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])

    def test_network_fine_sweep(self):
        at = open_page("network")
        at.toggle(key="nw_live").set_value(True).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])


class TestMultisitePage(unittest.TestCase):
    def test_live_planner_switches_use_case(self):
        at = open_page("multisite")
        at.selectbox(key="ms_uc").set_value("uc2_sovereign_residency").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])



class TestExplanationsAndAssistant(unittest.TestCase):
    def test_migration_page_explains_rejection(self):
        at = open_page("migration")
        self.assertFalse(at.exception, [e.value for e in at.exception])
        md = " ".join(m.value for m in at.markdown)
        self.assertIn("Why was it rejected?", md)
        self.assertIn("What would change the decision", md)
        self.assertTrue(any("REJECTED because" in e.value for e in at.error))

    def test_placement_page_shows_random_check(self):
        at = open_page("placement")
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any("Random placements beaten" in m.label for m in at.metric))

    def test_chat_answers_a_question(self):
        at = open_page("assistant_chat")
        at.chat_input[0].set_value("Why was the move rejected?").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.session_state.chat_history[-1]["role"], "assistant")
        self.assertIn("REJECTED", at.session_state.chat_history[-1]["text"])

    def test_sidebar_quick_question(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.text_input(key="quick_q").set_value("how are the experts distributed").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])


if __name__ == "__main__":
    unittest.main()
