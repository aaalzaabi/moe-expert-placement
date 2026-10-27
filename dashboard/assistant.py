"""Offline project assistant: answers questions about this project from its own data.

It is NOT a language model. It matches the question to one of a fixed set of topics (keyword + fuzzy matching)
and builds the answer from live numbers in the saved results, so answers always agree with the pages.
Unknown questions get suggestions instead of a guess.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable

from knowledge import TERMS


@dataclass
class Answer:
    text: str
    page: tuple[str, str] | None = None  # (file, label) to open for details
    topic: str = ""
    suggestions: list[str] = field(default_factory=list)


@dataclass
class Topic:
    name: str
    keywords: tuple[str, ...]        # any match scores; multi-word phrases score more
    examples: tuple[str, ...]        # example questions for fuzzy matching and suggestions
    build: Callable[[dict], Answer]


# ------------------------------------------------------------------ answers (facts dict -> Answer)
def _project(f):
    return Answer(
        "This project asks **where the experts of a large AI model should live on GPUs, and when it is worth moving "
        "them.** It runs the real OLMoE model to measure which of its 1,024 experts are used most, then — in a "
        "simulation — places the experts on GPUs so no GPU is overloaded, and uses a controller to decide whether "
        f"rearranging them is worth the copying cost. Main result: the busiest GPU's load drops by about "
        f"**{f['cut']:.0f}%** (simulation).", ("app_pages/start.py", "Start here"))


def _moe(f):
    return Answer(
        "OLMoE is a **Mixture-of-Experts** model: it has **16 layers**, each with **64 small networks (experts)**. "
        "For every word, a *router* picks **8 of the 64** experts in each layer. That is **1,024 experts** in total, "
        "each about **12.6 MB**, together about 93% of the model's size.", ("app_pages/model.py", "What is inside the model?"))


def _method(f):
    return Answer(
        "The **demand-aware** method is a greedy 'biggest job first' rule:\n"
        "1. Count how often each expert was chosen on the **training questions (0, 2, 4)** — its *demand*.\n"
        "2. Sort the 1,024 experts from most to least used.\n"
        "3. Put each expert on the GPU where **(work already there + this expert's demand) ÷ GPU speed** is smallest, "
        "as long as the GPU has a free slot (max 256 per GPU).\n\n"
        "So faster GPUs receive more work and all GPUs finish at about the same time. It is compared with **static** "
        "(dealt out in order) and **load-balanced** (same rule but ignoring speed). Page 3 has a worked example.",
        ("app_pages/placement.py", "Where should experts go?"))


def _visual(f):
    return Answer(
        "Open **3b. Watch the GPUs at work (visual)**. It draws each simulated GPU as a grid of 256 slots and has two "
        "animations (press **▶ Play**):\n"
        "1. **Placing the experts:** the rule fills the GPUs one expert at a time, busiest first, while bars show each "
        "GPU's busy time growing. You can also pick any step to see the numbers behind that decision.\n"
        "2. **Running a question:** a real question goes through the 16 layers; the experts it uses light up on their "
        "GPU and the slowest GPU of each layer turns red, because the others must wait for it.\n\n"
        "Switch between static, load-balanced and demand-aware to compare. GPU speeds are assumed (simulation).",
        ("app_pages/gpu_sim.py", "Watch the GPUs at work"))


def _random(f):
    v = f["validation"]
    da = v["table"].set_index("strategy").loc["demand_aware"]
    return Answer(
        "It is **not random guessing** — three pieces of evidence:\n"
        f"- **Beats random:** we tried {v['n']:,} random placements. Ours beats **{da.beats_random_pct:.1f}%** of them "
        f"(best random: {v['random_best']:,.0f}; ours: {da.peak_load:,.0f}; lower is better).\n"
        f"- **Close to the best possible:** within **{da.gap_to_best_pct:.1f}%** of the theoretical best "
        f"({v['lower_bound']:,.0f}), which no placement can beat.\n"
        "- **Tested on unseen questions:** it is planned on questions 0, 2, 4 and checked on 1, 3, 5. A test changes "
        "the unseen questions and confirms the placement does not change (no peeking).\n\n"
        "Caveat: only 3 unseen questions and assumed GPU speeds, so the exact % may change with more data.",
        ("app_pages/placement.py", "Where should experts go?"))


def _results(f):
    return Answer(
        f"- **Placement:** the busiest GPU's load drops by **{f['cut']:.1f}%** vs static on unseen questions "
        f"(load-balanced alone: {f['bal_cut']:.1f}%).\n"
        f"- **Request time (batch simulation):** slowest requests about **{f['p95_cut']:.0f}%** faster.\n"
        f"- **Moving experts** costs copying 776 experts (≈ 9.8 GB); it only pays off on fast networks "
        f"(above ≈ {f['crossover']:.0f} GB/s).\n"
        f"- **Controller:** said **{f['saved_decision']}** in the saved scenario, and the check on unseen questions agreed.\n\n"
        "All of these are simulation results with assumed hardware.", ("app_pages/evaluation.py", "All results side by side"))


def _why_rejected(f):
    ex = f["explanation"]
    lines = [ex.headline, ""]
    failed = [c for c in ex.checks if c["passed"] is False]
    lines += [f"- {c['meaning']}" for c in failed]
    if ex.cause:
        lines += ["", "**Cause:** " + " ".join(ex.cause)]
    if ex.approved:
        lines = [ex.headline]
    return Answer("\n".join(lines) + f"\n\n*(For your current settings: {f['scenario']}.)*",
                  ("app_pages/migration.py", "Should we move experts?"))


def _what_change(f):
    ex = f["explanation"]
    if ex.approved:
        return Answer("With your current settings the controller already **approves** the move.",
                      ("app_pages/migration.py", "Should we move experts?"))
    return Answer("Each of these was found by re-running the controller with one setting changed:\n"
                  + "\n".join(f"- {c}" for c in ex.what_would_change),
                  ("app_pages/migration.py", "Should we move experts?"))


def _controller(f):
    return Answer(
        "The **controller** decides whether to move experts to the better layout. Using only the past (training) "
        "questions, it simulates *move* vs *don't move* and approves only if **all three** checks pass:\n"
        "1. every GPU has room for old + new copies during the copy;\n"
        "2. overall time improves by at least 2%;\n"
        "3. the slowest request is not slowed down (limit 0%).\n\n"
        "Afterwards the unseen questions are used to check whether it was right. Ask *'why was it rejected?'* for "
        "the current reasons.", ("app_pages/migration.py", "Should we move experts?"))


def _network(f):
    return Answer(
        "Moving experts means copying ≈ 9.8 GB between GPUs. On a **slow network** the copy takes long and delays "
        "running requests, so moving **hurts**; on a **fast network** it **helps**. In the simulation the break-even "
        f"is about **{f['crossover']:.0f} GB/s** (12.6 MB experts; ≈ 15 GB/s with the older 8 MB guess). "
        "GB/s = gigabytes per second; 100 Gbps (gigabits) ≈ 12.5 GB/s. This break-even applies to this simulated "
        "scenario only.", ("app_pages/network.py", "Does network speed matter?"))


def _real(f):
    return Answer(
        "**Real:** the OLMoE model and its experts, which experts were chosen for 6 real questions, the model's run "
        "time on your Mac, and the expert sizes (from the model files).\n\n"
        "**Simulated:** the GPUs, their speeds, the network, all latency/time numbers and moving experts. Nothing is "
        "run on real GPUs. Green labels on the pages = real; orange/violet = simulation.",
        ("app_pages/start.py", "Start here"))


def _data(f):
    return Answer(
        f"The real model was run on **{f['n_requests']} questions** (2 AI, 2 coding, 2 general; {f['tokens']} tokens "
        "in total). Questions **0, 2, 4** are used to plan and decide (*training*); questions **1, 3, 5** are kept "
        "aside to check results fairly (*held-out*). For each question we recorded which 8 of 64 experts each word "
        "used in each of the 16 layers.", ("app_pages/routing.py", "Which experts are busy?"))


def _limits(f):
    return Answer(
        "Main limitations:\n- Only **6 questions** (3 for checking), so small differences are not reliable.\n"
        "- **GPU speeds and network are assumed**, not measured.\n- Only the reading of the prompt (prefill) is "
        "simulated, not text generation.\n- The 'slowest request' with 3 questions is basically one request.\n\n"
        "Next steps: more questions, repeated runs with confidence intervals, and calibration on real NVIDIA GPUs.",
        ("app_pages/findings.py", "What we learned and the limits"))


def _latency(f):
    return Answer(
        "- **Latency:** how long one request takes, arrival to finish.\n- **p50:** the typical (median) request.\n"
        "- **p95:** the slow requests — with only 3 questions this is basically the slowest one.\n"
        "- **Makespan:** when the last request finishes.\nAll are simulated milliseconds; lower is better.",
        ("app_pages/inference.py", "How fast would requests run?"))


def _migration_cost(f):
    return Answer(
        "Switching from the simple layout to the demand-aware one moves **776 of the 1,024 experts**. At 12.6 MB each "
        "that is **≈ 9.8 GB** of copying (6.2 GB with the older 8 MB guess). Copies are staged next to the old ones "
        "and the new layout switches on only when **all** copies have arrived; requests already running keep the old "
        "layout. If a copy fails, the system rolls back to the old layout.", ("app_pages/migration.py", "Should we move experts?"))


def _use(f):
    return Answer(
        "Start with **Start here**, then follow the numbered pages 1–8 in the sidebar using the *Next* links at the "
        "bottom of each page. Every page opens with an **About this page** box. Settings are optional (in *Change "
        "the simulated setup*). For a talk, use **Presentation mode**. Unknown words: **Glossary**.",
        ("app_pages/start.py", "Start here"))


def _present(f):
    return Answer(
        "Use **Presentation mode** (second item in the sidebar): 10 steps, about 9 minutes, with talking points for "
        "each. Optional live moment: on page 5 click *Approve (override)* and watch the move happen, then try "
        "*Inject copy failure* to show the rollback. See docs/DEMO_GUIDE.md.", ("app_pages/demo.py", "Presentation mode"))


def _run(f):
    return Answer(
        "From the project folder:\n```\n.venv/bin/streamlit run dashboard/app.py          # this dashboard\n"
        ".venv/bin/python -m unittest discover -s tests -t .   # all tests\n"
        ".venv/bin/python scripts/run_experiment.py --bandwidth 3 12 25 100   # new simulation runs\n"
        ".venv/bin/python scripts/profile_routing.py --dry-run  # check before collecting new real data\n```\n"
        "New results go to `outputs/runs/`; the original files are never overwritten.")


def _multisite(f):
    return Answer(
        "The **multi-site** extension spreads experts over several data-centre sites. Demand changes during the day, "
        "so it **forecasts** demand, re-plans placement, and an **operations agent** proposes changes; big or "
        "regulated changes need human approval. Site traffic there is synthetic.",
        ("app_pages/multisite.py", "Multi-site, forecasting & agent"))


def _agent_reject(f):
    return Answer(
        "The multi-site agent's proposals are checked by fixed rules. A proposal is **rejected** if it would break a "
        "capacity/memory/residency rule, overload a device, bring too small a gain (< 2%), or take too long to copy "
        "(> 5% of the hour). If all rules pass but the change is large (> 50 GB), touches a regulated site or switches "
        "devices off, it **waits for a human**. Each rejected proposal on that page now shows what failed, what it "
        "means and what would change it.", ("app_pages/multisite.py", "Multi-site, forecasting & agent"))


TOPICS = [
    Topic("project", ("project", "about", "purpose", "goal", "summary", "overview", "what does this", "research question"),
          ("What is this project about?", "What is the goal?"), _project),
    Topic("moe", ("olmoe", "mixture", "expert", "layer", "router", "model", "top-8", "top 8"),
          ("What is a mixture of experts?", "What is inside the model?"), _moe),
    Topic("method", ("method", "methodology", "distribute", "distribution", "algorithm", "how are experts",
                     "placement", "place", "assign", "greedy", "demand-aware", "demand aware"),
          ("How are the experts distributed?", "What is the placement method?"), _method),
    Topic("visual", ("visual", "visualise", "visualize", "animation", "animate", "watch", "picture", "see the gpus",
                     "show me the gpus", "gpu simulation", "simulator", "step by step"),
          ("Can I see the experts being placed on the GPUs?", "Is there a visual simulation?"), _visual),
    Topic("random", ("reliable", "trustworthy", "accurate", "random", "guess", "guessing", "luck", "correct", "proof", "prove", "valid", "trust", "sure",
                     "real result", "chance"),
          ("Is it better than random guessing?", "How do I know the placement is correct?", "Are the results reliable?"), _random),
    Topic("results", ("result", "results", "finding", "findings", "improvement", "better", "percent", "%", "how much"),
          ("What are the main results?", "How much better is it?"), _results),
    Topic("why_rejected", ("explain the decision", "saying no", "says no", "said no", "why rejected", "rejected", "reject", "why not", "refuse", "denied", "no move", "why was"),
          ("Why was the move rejected?",), _why_rejected),
    Topic("what_change", ("what would change", "approve", "approved", "make it pass", "would pass", "change the decision",
                          "how to approve"),
          ("What would make the controller approve?",), _what_change),
    Topic("controller", ("controller", "decision", "decide", "cost-aware", "cost aware", "policy", "rule", "checks"),
          ("How does the controller decide?",), _controller),
    Topic("network", ("network", "bandwidth", "gb/s", "gbps", "speed", "crossover", "break-even", "break even", "link"),
          ("Does network speed matter?", "What is the break-even bandwidth?"), _network),
    Topic("real", ("real", "simulated", "simulation", "measured", "fake", "assumed", "actual gpu"),
          ("What is real and what is simulated?",), _real),
    Topic("data", ("data", "questions", "prompts", "requests", "training", "held-out", "held out", "test set", "split"),
          ("What data was used?", "What are training and held-out questions?"), _data),
    Topic("limits", ("limitation", "limitations", "weakness", "problem", "threat", "next step", "future", "small sample"),
          ("What are the limitations?", "What are the next steps?"), _limits),
    Topic("latency", ("latency", "p95", "p50", "makespan", "percentile", "slowest", "time"),
          ("What do p50, p95 and makespan mean?",), _latency),
    Topic("migration", ("migration", "migrate", "move", "moving", "copy", "copies", "776", "transfer", "rollback"),
          ("How much does moving experts cost?",), _migration_cost),
    Topic("use", ("how to use", "navigate", "start", "where do i", "dashboard", "confusing", "page", "help"),
          ("How do I use this dashboard?",), _use),
    Topic("present", ("present", "presentation", "demo", "professor", "talk", "slides"),
          ("How should I present this?",), _present),
    Topic("run", ("run the dashboard", "start the dashboard", "open the dashboard", "run", "command", "install", "launch", "terminal", "test", "tests", "script"),
          ("How do I run it?",), _run),
    Topic("multisite", ("multi-site", "multisite", "site", "sites", "region", "forecast", "forecasting", "regret"),
          ("What is the multi-site extension?",), _multisite),
    Topic("agent_reject", ("agent", "authoriser", "authorizer", "proposal", "human approval", "pending"),
          ("Why did the agent reject a proposal?",), _agent_reject),
]

SUGGESTED = ["What is this project about?", "How are the experts distributed?", "Is it better than random guessing?",
             "Why was the move rejected?", "What would make the controller approve?", "What is real and what is simulated?",
             "What are the main results?", "What are the limitations?"]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%/+\-\. ]", " ", text.lower()).strip()


def _score(q: str, topic: Topic) -> float:
    words = set(q.split())
    s = 0.0
    for k in topic.keywords:
        if " " in k or "-" in k or "/" in k:
            s += 2.0 if k in q else 0.0
        elif k in words or (len(k) > 4 and any(w.startswith(k) for w in words)):
            s += 1.0
    s += 3.0 * max(difflib.SequenceMatcher(None, q, _norm(e)).ratio() for e in topic.examples) ** 2
    return s


def _glossary(q: str) -> Answer | None:
    if re.match(r"^(why|how)\b", q):  # reasons and procedures are topics, not definitions
        return None
    for _, term, meaning in TERMS:
        names = [t.strip().lower() for t in re.split(r"[/(),]", term) if t.strip()]
        if any(n and re.search(rf"\b{re.escape(n)}\b", q) for n in names):
            if re.match(r"^(what is|what's|what are|define|meaning of|what does)\b", q) or len(q.split()) <= 3:
                return Answer(f"**{term}** — {meaning}", ("app_pages/glossary.py", "Glossary"), topic="glossary")
    return None


def answer(question: str, facts: dict) -> Answer:
    q = _norm(question)
    if not q:
        return Answer("Ask me anything about the project, for example:", suggestions=SUGGESTED[:4])
    ranked = sorted(((_score(q, t), i, t) for i, t in enumerate(TOPICS)), key=lambda x: (-x[0], x[1]))
    best_score, _, best = ranked[0]
    g = _glossary(q)
    # definitions win for "what is X" questions; otherwise a clear topic match wins over a word lookup
    if g and (re.match(r"^(what is|what's|what are|define|meaning of|what does)\b", q) or best_score < 2.0):
        return g
    if best_score < 1.0:
        return Answer("I'm not sure I understood. I can answer questions about this project — try one of these, or "
                      "check the **Glossary** for a specific term:", ("app_pages/glossary.py", "Glossary"),
                      topic="unknown", suggestions=SUGGESTED[:5])
    ans = best.build(facts)
    ans.topic = best.name
    ans.suggestions = [t.examples[0] for _, _, t in ranked[1:4]]
    return ans
