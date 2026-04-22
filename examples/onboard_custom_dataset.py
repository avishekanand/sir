"""
Onboarding a Custom Dataset (CUSTOM Qrel Source Path)
======================================================
Use this template when your dataset is not in ir_datasets and you need to
supply both the corpus loader and the qrel derivation logic yourself.

The three-component contract RAGtune requires:
    doc_iter_fn  →  generator of {"docno": str, "text": str}
    queries      →  list of {"id": str, "text": str}
    qrels        →  dict of {(query_id, doc_id): int}

This example uses synthetic in-memory data so it runs without any files.
In real usage, replace the inline data with reads from your JSONL/TSV files.

Run:
    python examples/onboard_custom_dataset.py
"""

import os
import sys
import io
import csv

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

# ── 1. Simulated "files" (replace with open() calls in real usage) ────────────

DOCS_JSONL = """\
{"doc_id": "d1", "title": "Photosynthesis", "text": "Plants convert sunlight into glucose via photosynthesis."}
{"doc_id": "d2", "title": "Respiration", "text": "Cells release energy from glucose through cellular respiration."}
{"doc_id": "d3", "title": "Mitosis", "text": "Mitosis is cell division that produces two identical daughter cells."}
{"doc_id": "d4", "title": "Osmosis", "text": "Osmosis is the movement of water through a semi-permeable membrane."}
{"doc_id": "d5", "title": "DNA", "text": "DNA carries genetic information and is found in the cell nucleus."}
"""

QUERIES_TSV = """\
q1\tHow do plants make food?
q2\tWhat is the function of DNA?
"""

QRELS_TSV = """\
q1\td1\t2
q1\td2\t1
q2\td5\t2
"""

# ── 2. Custom loader functions ─────────────────────────────────────────────────

import json

def load_corpus():
    """Yields {"docno": str, "text": str} for each document."""
    for line in DOCS_JSONL.strip().splitlines():
        doc = json.loads(line)
        yield {"docno": doc["doc_id"], "text": f"{doc['title']} {doc['text']}".strip()}


def load_queries(n=10):
    """Returns list of {"id": str, "text": str}."""
    reader = csv.reader(io.StringIO(QUERIES_TSV), delimiter="\t")
    return [{"id": row[0], "text": row[1]} for row in reader][:n]


def derive_qrels(corpus_data, queries):
    """
    Returns {(query_id, doc_id): relevance_int}.

    This is the function you pass as DatasetConfig.derive_qrels.
    It receives whatever ir_datasets returns as `ds` (or None for custom
    datasets), plus the loaded queries list.
    """
    qrels = {}
    reader = csv.reader(io.StringIO(QRELS_TSV), delimiter="\t")
    for qid, did, rel in reader:
        qrels[(qid, did)] = int(rel)
    return qrels


# ── 3. Build the three components ─────────────────────────────────────────────

print("Loading custom dataset …")
corpus_docs = list(load_corpus())
queries     = load_queries()
qrels       = derive_qrels(None, queries)

print(f"  {len(corpus_docs)} documents  |  {len(queries)} queries  |  {len(qrels)} qrel pairs")

# ── 4. Build an in-memory retriever (substitute PyTerrierRetriever for scale) ──

scored_docs = [
    ScoredDocument(id=d["docno"], content=d["text"], score=1.0, token_count=len(d["text"].split()))
    for d in corpus_docs
]

class FullRetriever(InMemoryRetriever):
    def retrieve(self, query, top_k):
        return self.docs

retriever  = FullRetriever(scored_docs)
controller = RAGtuneController(
    retriever=retriever,
    reformulator=IdentityReformulator(),
    reranker=NoOpReranker(),
    assembler=GreedyAssembler(),
    scheduler=ActiveLearningScheduler(batch_size=5),
    estimator=BaselineEstimator(),
    budget=CostBudget(limits={"rerank_docs": 0, "latency_ms": 5_000}),
)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────

print("\nResults (CUSTOM qrel_source, RETRIEVAL_GRADED task):\n")
print(f"  {'Query':<35}  {'Recall@3':>9}  {'MRR':>6}")
print("  " + "-" * 55)

for q in queries:
    out     = controller.run(q["text"])
    docs    = out.documents
    rel_set = {did for (qid, did), r in qrels.items() if qid == q["id"] and r > 0}

    recall  = sum(1 for d in docs[:3] if d.id in rel_set) / max(len(rel_set), 1)
    mrr_val = next((1.0 / (i + 1) for i, d in enumerate(docs) if d.id in rel_set), 0.0)
    print(f"  {q['text']:<35}  {recall:>9.3f}  {mrr_val:>6.3f}")

print()
print("─" * 55)
print("To use this dataset in the experiment grid, add to DATASETS:")
print("""
  DatasetConfig(
      name="my-bench",
      ir_id="",                            # not in ir_datasets
      doc_cap=0,
      n_queries=100,
      task_type=RAGTaskType.RETRIEVAL_GRADED,
      qrel_source=QrelSource.CUSTOM,
      derive_qrels=derive_qrels,           # your function above
  ),
""")
print("And add a custom branch to load_dataset() for the corpus + queries.")
