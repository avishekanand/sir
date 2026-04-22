"""
Onboarding a QA Dataset (Proxy Qrels Path)
===========================================
When a dataset has answer strings but no explicit relevance judgments, RAGtune
can derive binary qrels via answer-presence heuristics. This script shows:

  1. PROXY_EXACT    — answer string appears verbatim in doc text
  2. PROXY_TOKEN_F1 — >50% of answer tokens appear in doc text

It also surfaces cases where the two strategies disagree, making the noise
visible rather than hiding it.

Run:
    python examples/onboard_qa_dataset.py

This example is self-contained (synthetic data, no ir_datasets download).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from ragtune.core.types import ScoredDocument
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.components.rerankers import NoOpReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import BaselineEstimator
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget

# ── 1. Synthetic corpus ───────────────────────────────────────────────────────
#
# Intended relevant docs contain the answer correctly in context.
# Noise docs contain the answer string but in an unrelated or wrong context.

CORPUS = [
    # query: "What is the capital of France?"  answer: "Paris"
    ScoredDocument(id="d1", content="Paris is the capital and largest city of France, known for the Eiffel Tower.", score=0.9, token_count=15),
    ScoredDocument(id="d2", content="France is a country in Western Europe. Its capital is Paris.", score=0.8, token_count=12),
    ScoredDocument(id="d3", content="The Paris Agreement is an international treaty on climate change.", score=0.6, token_count=10),  # noise: "Paris" present but wrong context
    ScoredDocument(id="d4", content="London is the capital of England and the United Kingdom.", score=0.5, token_count=12),
    ScoredDocument(id="d5", content="European capitals include Brussels, Amsterdam, and Berlin.", score=0.4, token_count=10),

    # query: "Who wrote Romeo and Juliet?"  answer: "Shakespeare"
    ScoredDocument(id="d6", content="Romeo and Juliet is a tragedy written by William Shakespeare.", score=0.85, token_count=12),
    ScoredDocument(id="d7", content="Shakespeare wrote many famous plays including Hamlet and Macbeth.", score=0.75, token_count=12),
    ScoredDocument(id="d8", content="The Shakespeare pub in London serves traditional British food.", score=0.55, token_count=11),  # noise
    ScoredDocument(id="d9", content="Elizabethan theatre flourished during the reign of Queen Elizabeth I.", score=0.45, token_count=11),
    ScoredDocument(id="d10", content="Globe Theatre was built by Shakespeare's playing company.", score=0.4, token_count=10),
]

QUERIES = [
    {"id": "q1", "text": "What is the capital of France?"},
    {"id": "q2", "text": "Who wrote Romeo and Juliet?"},
]

# Gold answers — what the QA dataset ships instead of qrels
ANSWERS = {
    "q1": ["Paris"],
    "q2": ["Shakespeare", "William Shakespeare"],
}

# ── 2. Proxy qrel derivation ──────────────────────────────────────────────────

def _token_f1(answer: str, text: str) -> float:
    a_toks = set(answer.lower().split())
    t_toks = set(text.lower().split())
    return len(a_toks & t_toks) / len(a_toks) if a_toks else 0.0

def derive_proxy_qrels(docs, query_id, answers, method="exact"):
    qrels = {}
    for doc in docs:
        text = doc.content.lower()
        if method == "exact":
            rel = int(any(ans.lower() in text for ans in answers))
        else:  # token_f1
            rel = int(any(_token_f1(ans, text) > 0.5 for ans in answers))
        qrels[(query_id, doc.id)] = rel
    return qrels

# ── 3. Run retrieval and compare proxy strategies ─────────────────────────────

class FullRetriever(InMemoryRetriever):
    def retrieve(self, query, top_k):
        return self.docs

retriever = FullRetriever(CORPUS)
controller = RAGtuneController(
    retriever=retriever,
    reformulator=IdentityReformulator(),
    reranker=NoOpReranker(),
    assembler=GreedyAssembler(),
    scheduler=ActiveLearningScheduler(batch_size=5),
    estimator=BaselineEstimator(),
    budget=CostBudget(limits={"rerank_docs": 0, "latency_ms": 5_000}),
)

print("=" * 60)
print("Proxy Qrel Comparison: EXACT vs TOKEN_F1")
print("=" * 60)
print("⚠  Proxy qrels measure answer PRESENCE, not document relevance.")
print("   Scores are directional only — not comparable to gold qrels.\n")

for q in QUERIES:
    out  = controller.run(q["text"])
    docs = out.documents[:5]

    exact_qrels = derive_proxy_qrels(docs, q["id"], ANSWERS[q["id"]], "exact")
    f1_qrels    = derive_proxy_qrels(docs, q["id"], ANSWERS[q["id"]], "token_f1")

    print(f"Query: {q['text']}")
    print(f"  Answers: {ANSWERS[q['id']]}\n")
    print(f"  {'Doc':<6}  {'Content (truncated)':<55}  {'EXACT':>6}  {'F1':>4}  {'Agree?':>7}")
    print("  " + "-" * 80)

    disagreements = 0
    for doc in docs:
        e = exact_qrels.get((q["id"], doc.id), 0)
        f = f1_qrels.get((q["id"], doc.id), 0)
        agree = "✓" if e == f else "✗ NOISE"
        if e != f:
            disagreements += 1
        snippet = doc.content[:52] + "…" if len(doc.content) > 52 else doc.content
        print(f"  {doc.id:<6}  {snippet:<55}  {e:>6}  {f:>4}  {agree:>7}")

    print(f"\n  Disagreements (noise): {disagreements}/{len(docs)}")
    exact_recall = sum(1 for d in docs if exact_qrels.get((q["id"], d.id), 0)) / len(docs)
    f1_recall    = sum(1 for d in docs if f1_qrels.get((q["id"], d.id), 0))    / len(docs)
    print(f"  Proxy Recall@5 (EXACT):    {exact_recall:.3f}")
    print(f"  Proxy Recall@5 (TOKEN_F1): {f1_recall:.3f}")
    print()

print("Takeaway: TOKEN_F1 filters the 'Shakespeare pub' false positive.")
print("Neither strategy is ground truth — use gold qrels when available.")
