import json

from ragtune.indexing.autoindex.history import CandidateRecord, IterationRecord, SearchHistory


def test_empty_render():
    assert "(none yet)" in SearchHistory().render()


def test_render_scores_candidates_and_overfit_warning(tmp_path):
    h = SearchHistory()
    h.record_iteration(IterationRecord(0, 0.50, 0.30, 0.20))
    h.record_candidate(CandidateRecord(0, "H1", "add title chunk", delta_recall_100=0.05, proven=True,
                                       improved_query_ids=["a"], regressed_query_ids=["b", "c"]))
    h.record_candidate(CandidateRecord(0, "H2", "strip code", error="SyntaxError"))
    h.set_adoption(0, "H1")
    h.record_iteration(IterationRecord(1, 0.55, 0.31, 0.21))

    text = h.render()
    assert "iter 0: R@100=0.5000" in text and "→ adopted H1" in text
    assert "iter 1: R@100=0.5500 (+0.0500)" in text
    assert "adopted; ΔR@100=+0.0500" in text
    assert "error: SyntaxError" in text
    assert "broke more validation queries" in text

    h.save(tmp_path / "h.json")
    data = json.loads((tmp_path / "h.json").read_text())
    assert data["candidates"][0]["adopted"] is True
