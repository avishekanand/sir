"""
Query- and document-level evidence for LLM-driven tuning arms.

Turns a trial's ranked lists plus the ground-truth qrels into text a proposer
can diagnose: where each relevant document sits in the first-stage ranking,
whether it entered the candidate pool, whether it was reranked, and where it
ended up. Shared by the `reflect` arm (proposer reads this directly) and the
`agenthpo3` arm (a separate Analyzer agent digests it first).

First-stage ranks are computed ONCE per cell. The retriever is fixed across
trials and original_query_depth only truncates its ordering, so a relevant
document's rank in that ordering does not change between trials.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Depth to probe when locating relevant documents in the first-stage ranking.
# Beyond this they are reported as not retrieved, which is itself actionable:
# no setting of the tunable knobs can recover them.
FIRST_STAGE_PROBE_DEPTH = 300


def build_relevance_map(
    fixed_retriever: Any, eval_dataset: Any, n_eval_queries: int
) -> Dict[str, Dict[str, Any]]:
    """query_id -> {"query": text, "relevant": [(doc_id, rel, first_stage_rank)]}."""
    import pandas as pd

    transformer = getattr(fixed_retriever, "pt_transformer", None)
    out: Dict[str, Dict[str, Any]] = {}
    for eq in eval_dataset.iter_queries(limit=n_eval_queries):
        rel = {d: r for d, r in eq.qrels.items() if r > 0}
        ranks: Dict[str, Optional[int]] = {d: None for d in rel}
        if transformer is not None and rel:
            try:
                res = transformer.transform(
                    pd.DataFrame([{"qid": "q1", "query": eq.query}])
                )
                ordered = list(
                    res.sort_values("score", ascending=False)["docno"].astype(str)
                )[:FIRST_STAGE_PROBE_DEPTH]
                pos = {d: i + 1 for i, d in enumerate(ordered)}
                ranks = {d: pos.get(d) for d in rel}
            except Exception:  # noqa: BLE001 — evidence is best-effort
                pass
        out[eq.query_id] = {
            "query": eq.query,
            "relevant": sorted(
                ((d, rel[d], ranks.get(d)) for d in rel),
                key=lambda t: (t[2] is None, t[2] or 0),
            ),
        }
    return out


def choose_evidence_qids(
    query_ndcg: Dict[str, float],
    relevance_map: Dict[str, Dict[str, Any]],
    n_evidence: int,
    n_contrast: int,
) -> List[str]:
    """Worst queries carry the diagnosis; a few good ones give contrast."""
    if not query_ndcg:
        return list(relevance_map)[:n_evidence]
    ordered = sorted(query_ndcg, key=lambda q: query_ndcg[q])
    n_bad = max(1, n_evidence - n_contrast)
    return ordered[:n_bad] + (ordered[-n_contrast:] if n_contrast else [])


def render_evidence(
    qids: Sequence[str],
    per_query: Dict[str, List[Dict[str, Any]]],
    query_ndcg: Dict[str, float],
    relevance_map: Dict[str, Dict[str, Any]],
    depth: Optional[int],
) -> str:
    out: List[str] = ["## Per-query outcomes (ranked list vs ground truth)"]
    for qid in qids:
        info = relevance_map.get(qid, {})
        out.append(
            f"\n### query {qid}  NDCG@10={query_ndcg.get(qid, 0.0):.3f}"
            f"\n  text: {str(info.get('query',''))[:160]}"
        )
        rel = info.get("relevant", [])
        if rel:
            out.append(f"  relevant documents ({len(rel)}):")
            for doc_id, r, fs in rel[:6]:
                where = (f"first-stage rank {fs}" if fs
                         else f"NOT in first-stage top {FIRST_STAGE_PROBE_DEPTH}")
                flag = (f"  <-- BEYOND original_query_depth={depth}"
                        if fs and depth and fs > depth else "")
                out.append(f"    rel={r} doc={doc_id}  {where}{flag}")
        returned = per_query.get(qid, [])
        if returned:
            out.append("  returned top-10 (rank, rel, first-stage rank, reranked):")
            for d in returned:
                hit = "HIT " if d["rel"] > 0 else "    "
                out.append(
                    f"    {hit}{d['rank']:2}. rel={d['rel']} "
                    f"first_stage={d['initial_rank']:3} "
                    f"reranked={'y' if d['reranked'] else 'n'}  {d['doc_id']}"
                )
            found = {d["doc_id"] for d in returned if d["rel"] > 0}
            missed = [(i, r, fs) for i, r, fs in rel if i not in found]
            if missed:
                out.append("  MISSED relevant documents:")
                for doc_id, r, fs in missed[:5]:
                    if fs is None:
                        why = "never retrieved by the first stage"
                    elif depth and fs > depth:
                        why = (f"first-stage rank {fs} > depth {depth}: "
                               "never entered the pool")
                    else:
                        why = (f"first-stage rank {fs}: entered the pool but "
                               "lost downstream")
                    out.append(f"    rel={r} doc={doc_id} — {why}")
    return "\n".join(out)
