"""
AutoIndex on a CRUMB task: learn a representation program, then score it on held-out queries.

Follows the paper's protocol: full-document corpus + full-document qrels, a 1:2
validation:evaluation query split, BM25 with MaxP, and held-out queries used once at the end.

    python examples/autoindex_crumb.py --task theorem_retrieval \
        --model openai/qwen/qwen3-coder --api-base https://openrouter.ai/api/v1

Pass ``--splits-file`` with the reference repo's ``query_splits.json`` to reuse the
paper's exact query partition. Otherwise the split is a seeded shuffle.
Needs: ``pip install -e '.[benchmarks,autoindex]'`` plus an API key litellm can find.

WARNING: AutoIndex executes LLM-generated Python in this process. The AST screen
in ``program.py`` is a guardrail, not a sandbox; run inside a container.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Tuple

from ragtune.indexing.autoindex import (
    IDENTITY_PROGRAM_CODE,
    AutoIndex,
    AutoIndexConfig,
    LiteLLMClient,
    UsageTracker,
)

CORPUS_DESCRIPTIONS = {
    "theorem_retrieval": "Mathematical theorems, lemmas and proofs. Queries describe a result in natural language.",
    "tip_of_the_tongue": "Wikipedia-style articles (e.g. movies, books). Queries are vague recollections.",
    "clinical_trial": "Clinical trial records. Queries are patient descriptions.",
    "paper_retrieval": "Scientific papers. Queries describe a paper's contribution.",
    "stack_exchange": "Web documents linked from StackExchange answers. Queries are user questions.",
    "code_retrieval": "Source code and documentation. Queries are programming problems.",
    "legal_qa": "Legal documents and statutes. Queries are legal questions.",
    "set_operation_entity_retrieval": "Wikipedia entity articles. Queries combine constraints via set operations.",
}


def load_crumb_documents(task: str, max_docs: int | None) -> Tuple[Dict, Dict, Dict]:
    """Full-document CRUMB (the paper's setting). Gold docs are always kept."""
    from datasets import load_dataset

    queries, qrels = {}, {}
    for row in load_dataset("jfkback/crumb", "evaluation_queries", split=task):
        rels = row.get("full_document_qrels") or row.get("passage_qrels") or []
        rel = {str(r["id"]): int(r["label"]) for r in rels if int(r.get("label", 0)) > 0}
        if rel:
            queries[str(row["query_id"])] = str(row["query_content"])
            qrels[str(row["query_id"])] = rel
    gold = {d for r in qrels.values() for d in r}
    corpus, n_other = {}, 0
    for row in load_dataset("jfkback/crumb", "full_document_corpus", split=task, streaming=True):
        did = str(row["document_id"])
        if did not in gold:
            if max_docs is not None and n_other >= max_docs:
                if gold.issubset(corpus):
                    break
                continue
            n_other += 1
        corpus[did] = {"text": row["document_content"], "title": ""}
    return corpus, queries, qrels


def split_queries(task, queries, splits_file, seed):
    if splits_file:
        s = json.loads(Path(splits_file).read_text())[task]
        val, ev = [q for q in s["validation"] if q in queries], [q for q in s["evaluation"] if q in queries]
    else:
        qids = sorted(queries)
        random.Random(seed).shuffle(qids)
        cut = len(qids) // 3
        val, ev = qids[:cut], qids[cut:]
    return {q: queries[q] for q in val}, {q: queries[q] for q in ev}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="theorem_retrieval", choices=sorted(CORPUS_DESCRIPTIONS))
    p.add_argument("--model", required=True, help="litellm model id, used for both agents")
    p.add_argument("--analysis-model", help="override the Analysis Agent model")
    p.add_argument("--api-base", default=None)
    p.add_argument("--max-tokens", type=int, default=None, help="cap tokens per LLM reply (local models)")
    p.add_argument("--llm-timeout", type=float, default=None, help="seconds per LLM call")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--no-analysis", action="store_true", help="ablation: Code Agent sees metrics only")
    p.add_argument("--no-history", action="store_true", help="ablation: no search history")
    p.add_argument("--max-corpus-docs", type=int, default=None, help="cap non-gold docs (debugging)")
    p.add_argument("--passages", action="store_true", help="use RAGtune's passage-level CRUMBLoader instead")
    p.add_argument("--splits-file", default=None, help="reference repo's query_splits.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/autoindex")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.passages:
        from ragtune.data.loaders.CRUMBLoader import CRUMBLoader
        corpus, queries, qrels = CRUMBLoader(args.task, max_corpus_docs=args.max_corpus_docs).load()
    else:
        corpus, queries, qrels = load_crumb_documents(args.task, args.max_corpus_docs)
    val_q, eval_q = split_queries(args.task, queries, args.splits_file, args.seed)
    print(f"[autoindex] {args.task}: {len(corpus)} docs, {len(val_q)} val / {len(eval_q)} eval queries")

    out = Path(args.out) / args.task
    usage = UsageTracker()
    api_key = (os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")) if args.api_base else None
    mk = lambda model, agent: LiteLLMClient(model, temperature=1.0, api_base=args.api_base, api_key=api_key,
                                             timeout=args.llm_timeout, max_tokens=args.max_tokens,
                                             usage=usage, agent=agent)
    cfg = AutoIndexConfig(n_iterations=args.iterations, n_candidates=args.candidates,
                          use_analysis=not args.no_analysis, use_history=not args.no_history,
                          corpus_description=CORPUS_DESCRIPTIONS[args.task], log_dir=str(out))
    ai = AutoIndex(cfg, analysis_llm=mk(args.analysis_model or args.model, "analysis"),
                   code_llm=mk(args.model, "code"), usage=usage)

    result = ai.fit(corpus, val_q, qrels, n_eval_queries=len(eval_q))

    # Held-out evaluation, once, at the end.
    base = ai.evaluate_code(IDENTITY_PROGRAM_CODE, corpus, eval_q, qrels)
    learned = ai.evaluate_code(result.best_code, corpus, eval_q, qrels)
    report = {
        "task": args.task, "model": args.model,
        "val": {"baseline": result.initial_val.as_dict(), "learned": result.best_val.as_dict()},
        "eval": {"baseline": base.as_dict(), "learned": learned.as_dict()},
        "usage": usage.stats,
    }
    (out / "heldout.json").write_text(json.dumps(report, indent=2))
    for split in ("val", "eval"):
        b, l = report[split]["baseline"], report[split]["learned"]
        print(f"[autoindex] {split:4s} R@100 {b['recall_at_100']:.4f} → {l['recall_at_100']:.4f} | "
              f"nDCG@10 {b['ndcg_at_10']:.4f} → {l['ndcg_at_10']:.4f}")
    print(f"[autoindex] program + logs → {out}  (tokens used: {usage.total_tokens():,})")


if __name__ == "__main__":
    main()
