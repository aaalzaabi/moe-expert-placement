"""The offline assistant routes questions to the right topic and every answer builds from live data."""
import logging
import sys
import unittest

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "dashboard"))
logging.getLogger("streamlit").setLevel(logging.ERROR)

import assistant as A  # noqa: E402
import ui  # noqa: E402

CASES = {
    "what is this project about": "project", "how are the experts distributed?": "method",
    "what is the methodology of distributing the experts": "method",
    "how can I know if it is correct or just random guessing": "random", "is the result reliable": "random",
    "why was it rejected": "why_rejected", "why rejected?": "why_rejected", "can you explain the decision": "why_rejected",
    "why is the controller saying no": "why_rejected", "what would make it approve": "what_change",
    "how does the controller decide": "controller", "does network speed matter": "network",
    "is this real or simulated": "real", "what data did you use": "data", "limitations": "limits",
    "how much does moving cost": "migration", "how do i present this to my professor": "present",
    "how do I run the dashboard": "run", "what are the results": "results",
    "why did the agent reject the proposal": "agent_reject", "what is p95": "glossary",
    "what is a router": "glossary", "what is an expert": "glossary", "pizza recipe": "unknown",
    "can I see an animation of the gpus": "visual", "show me a visual simulation of the experts": "visual",
}


class TestAssistant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = ui.assistant_facts(ui.cfg_key(ui.C.DEFAULT))

    def test_routing(self):
        for q, topic in CASES.items():
            with self.subTest(q=q):
                self.assertEqual(A.answer(q, self.facts).topic, topic)

    def test_every_topic_answers(self):
        for t in A.TOPICS:
            with self.subTest(topic=t.name):
                a = t.build(self.facts)
                self.assertGreater(len(a.text), 40)
                if a.page:
                    self.assertTrue((ROOT / "dashboard" / a.page[0]).exists(), a.page)

    def test_unknown_gets_suggestions_not_a_guess(self):
        a = A.answer("pizza recipe", self.facts)
        self.assertEqual(a.topic, "unknown")
        self.assertTrue(a.suggestions)

    def test_numbers_match_saved_results(self):
        self.assertIn("29.4%", A.answer("what are the results", self.facts).text)
        self.assertIn("REJECTED", A.answer("why was it rejected", self.facts).text)


if __name__ == "__main__":
    unittest.main()
