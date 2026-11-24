import json

import pytest

from ragtune.indexing.autoindex.loop import AutoIndex, AutoIndexConfig
from ragtune.indexing.autoindex.program import IDENTITY_PROGRAM_CODE

from .conftest import (
    BOTH_PROGRAM,
    EMPTY_PROGRAM,
    SPLIT_PROGRAM,
    TITLE_PROGRAM,
    ScriptedLLM,
    make_analysis_llm,
    make_code_llm,
)


def _cfg(**kw):
    base = dict(n_iterations=1, n_candidates=2, bm25_backend="python")
    base.update(kw)
    return AutoIndexConfig(**base)


def test_adopts_improving_candidate(toy_data):
    corpus, q, qrels = toy_data
    ai = AutoIndex(_cfg(), make_analysis_llm(), make_code_llm({"H1": TITLE_PROGRAM, "H2": EMPTY_PROGRAM}))
    res = ai.fit(corpus, q, qrels, n_eval_queries=20)
    assert res.initial_val.recall_at_100 == pytest.approx(0.2)
    assert res.best_val.recall_at_100 == pytest.approx(0.6)
    assert res.best_code.strip() == TITLE_PROGRAM.strip()
    assert res.history.iterations[0].adopted_id == "it0-H1"
    h2 = [c for c in res.history.candidates if c.candidate_id == "it0-H2"][0]
    assert not h2.proven and h2.regressed_query_ids == ["q8", "q9"]
    assert res.analyses[0].startswith("Titles are not indexed")


def test_no_adoption_when_nothing_improves(toy_data):
    corpus, q, qrels = toy_data
    ai = AutoIndex(_cfg(), make_analysis_llm(), make_code_llm({"H1": EMPTY_PROGRAM, "H2": IDENTITY_PROGRAM_CODE}))
    res = ai.fit(corpus, q, qrels)
    assert res.best_code.strip() == IDENTITY_PROGRAM_CODE.strip()
    assert res.history.iterations[0].adopted_id is None


def test_erroring_candidates_are_recorded_not_fatal(toy_data):
    corpus, q, qrels = toy_data
    bad = "import os\n" + TITLE_PROGRAM
    ai = AutoIndex(_cfg(), make_analysis_llm(), make_code_llm({"H1": bad, "H2": TITLE_PROGRAM}))
    res = ai.fit(corpus, q, qrels)
    errs = {c.candidate_id: c.error for c in res.history.candidates}
    assert "not allowed" in errs["it0-H1"]
    assert res.best_code.strip() == TITLE_PROGRAM.strip()


def test_synthesis_adopted_when_it_beats_best_single(toy_data):
    corpus, q, qrels = toy_data
    llm = make_code_llm({"H1": TITLE_PROGRAM, "H2": SPLIT_PROGRAM}, synthesis_code=BOTH_PROGRAM)
    res = AutoIndex(_cfg(), make_analysis_llm(), llm).fit(corpus, q, qrels)
    assert res.best_val.recall_at_100 == pytest.approx(1.0)
    assert res.best_code.strip() == BOTH_PROGRAM.strip()
    assert res.history.iterations[0].synthesized is True


def test_synthesis_rejected_when_not_better(toy_data):
    corpus, q, qrels = toy_data
    llm = make_code_llm({"H1": TITLE_PROGRAM, "H2": SPLIT_PROGRAM}, synthesis_code=SPLIT_PROGRAM)
    res = AutoIndex(_cfg(), make_analysis_llm(), llm).fit(corpus, q, qrels)
    assert res.best_code.strip() == TITLE_PROGRAM.strip()  # tie on R@100 (0.6) broken by nDCG, never by synthesis
    assert res.history.iterations[0].synthesized is False


def test_incumbent_carries_across_iterations(toy_data):
    corpus, q, qrels = toy_data
    state = {"n": 0}
    base = make_code_llm({"H1": TITLE_PROGRAM})
    later = make_code_llm({"H1": BOTH_PROGRAM})

    def respond(messages, tools):
        user = messages[-1]["content"]
        if "hypothesis IDEAS" in user:
            state["n"] += 1
        return (base if state["n"] == 1 else later).responder(messages, tools)

    res = AutoIndex(_cfg(n_iterations=2, n_candidates=1), make_analysis_llm(), ScriptedLLM(respond)).fit(corpus, q, qrels)
    assert [r.recall_at_100 for r in res.history.iterations] == pytest.approx([0.2, 0.6])
    assert res.best_code.strip() == BOTH_PROGRAM.strip()


def test_no_analysis_ablation_uses_metrics_only(toy_data):
    corpus, q, qrels = toy_data
    code = make_code_llm({"H1": TITLE_PROGRAM})
    res = AutoIndex(_cfg(use_analysis=False, n_candidates=1), None, code).fit(corpus, q, qrels)
    assert "Current validation metrics" in code.calls[0][1]["content"]
    assert res.best_code.strip() == TITLE_PROGRAM.strip()


def test_no_history_ablation_omits_history(toy_data):
    corpus, q, qrels = toy_data
    code = make_code_llm({"H1": TITLE_PROGRAM})
    AutoIndex(_cfg(use_history=False, n_candidates=1), make_analysis_llm(), code).fit(corpus, q, qrels)
    assert "Search history" not in code.calls[0][1]["content"]


def test_zero_iteration_budget_returns_initial_program_without_llm_calls(toy_data):
    corpus, q, qrels = toy_data
    analysis, code = make_analysis_llm(), make_code_llm({"H1": TITLE_PROGRAM})
    res = AutoIndex(_cfg(n_iterations=0), analysis, code).fit(corpus, q, qrels)
    assert res.best_code.strip() == IDENTITY_PROGRAM_CODE.strip()
    assert res.best_val.recall_at_100 == res.initial_val.recall_at_100
    assert analysis.calls == [] and code.calls == []


def test_requires_validation_queries_with_qrels(toy_data):
    corpus, q, _ = toy_data
    with pytest.raises(ValueError):
        AutoIndex(_cfg(), make_analysis_llm(), make_code_llm({})).fit(corpus, q, {})


def test_requires_analysis_llm_when_analysis_enabled():
    with pytest.raises(ValueError):
        AutoIndex(_cfg(), None, make_code_llm({}))


def test_log_dir_artifacts_and_heldout_eval(toy_data, tmp_path):
    corpus, q, qrels = toy_data
    ai = AutoIndex(_cfg(log_dir=str(tmp_path)), make_analysis_llm(), make_code_llm({"H1": TITLE_PROGRAM}))
    res = ai.fit(corpus, q, qrels)
    assert (tmp_path / "best_program.py").read_text().strip() == TITLE_PROGRAM.strip()
    assert json.loads((tmp_path / "iteration_0.json").read_text())["adopted"] == "it0-H1"
    assert json.loads((tmp_path / "result.json").read_text())["best_val"]["recall_at_100"] == pytest.approx(0.6)
    held = ai.evaluate_code(res.best_code, corpus, {"q0": "title0"}, {"q0": {"d0": 1}})
    assert held.recall_at_100 == 1.0
