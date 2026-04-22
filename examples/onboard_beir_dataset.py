"""
Onboarding a BEIR Dataset (Gold Qrels Path)
============================================
Shows the minimal steps to integrate any BEIR benchmark that ships with
explicit relevance judgments. Uses beir/fiqa as the example.

Run:
    python examples/onboard_beir_dataset.py

Expected output:
    - Index build (cached after first run)
    - NDCG@5 / Recall@5 / MRR for 5 queries
    - A confirmation that task_type and qrel_source appear in the CSV
"""

import os
import sys
import tempfile

import ir_datasets
import numpy as np
import pyterrier as pt

if not pt.started():
    pt.init()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.components.rerankers import NoOpReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import BaselineEstimator

# ── 1. Define what this dataset provides ─────────────────────────────────────
#
# beir/fiqa:  financial domain QA
#   - has_collection: yes
#   - has_qrels:      yes, binary (0/1)
#   - has_answers:    no
#   → RAGTaskType.RETRIEVAL_BINARY, QrelSource.GOLD

IR_ID      = "beir/fiqa"
N_QUERIES  = 5      # keep small for a quick demo
TASK_TYPE  = "retrieval_binary"
QREL_SRC   = "gold"

# ── 2. Load qrels, queries, and corpus ───────────────────────────────────────

print(f"Loading {IR_ID} …")
ds = ir_datasets.load(IR_ID)

qrels = {}
for qr in ds.qrels_iter():
    qrels[(qr.query_id, qr.doc_id)] = qr.relevance

queries = []
for q in ds.queries_iter():
    if any(rel > 0 for (qid, _), rel in qrels.items() if qid == q.query_id):
        queries.append({"id": q.query_id, "text": q.text})
    if len(queries) >= N_QUERIES:
        break

print(f"  {len(queries)} queries with relevant documents loaded.")

# ── 3. Index the corpus ───────────────────────────────────────────────────────

index_path = os.path.join(tempfile.gettempdir(), "ragtune_fiqa_index")

if os.path.exists(os.path.join(index_path, "data.properties")):
    print("  Loading cached index …")
    index_ref = pt.IndexFactory.of(index_path)
else:
    print("  Building index (first run only) …")
    os.makedirs(index_path, exist_ok=True)

    def doc_iter():
        for doc in ds.docs_iter():
            yield {"docno": doc.doc_id, "text": f"{doc.title} {doc.text}".strip()}

    indexer  = pt.IterDictIndexer(index_path, overwrite=True, meta={"docno": 64, "text": 4096})
    index_ref = indexer.index(doc_iter())

bm25      = pt.BatchRetrieve(index_ref, wmodel="BM25", metadata=["docno", "text"], num_results=50)
retriever = PyTerrierRetriever(bm25)

# ── 4. Run BM25 baseline ──────────────────────────────────────────────────────

controller = RAGtuneController(
    retriever=retriever,
    reformulator=IdentityReformulator(),
    reranker=NoOpReranker(),
    assembler=GreedyAssembler(),
    scheduler=ActiveLearningScheduler(batch_size=5),
    estimator=BaselineEstimator(),
    budget=CostBudget(limits={"rerank_docs": 0, "latency_ms": 10_000}),
)

# ── 5. Evaluate and print metrics ─────────────────────────────────────────────

def recall_at_k(docs, qrels, qid, k=5):
    relevant = {did for (q, did), r in qrels.items() if q == qid and r > 0}
    return sum(1 for d in docs[:k] if d.id in relevant) / max(len(relevant), 1)

def mrr(docs, qrels, qid):
    relevant = {did for (q, did), r in qrels.items() if q == qid and r > 0}
    for i, d in enumerate(docs):
        if d.id in relevant:
            return 1.0 / (i + 1)
    return 0.0

rows = []
for q in queries:
    out   = controller.run(q["text"])
    docs  = out.documents
    rel   = {did for (qid, did), r in qrels.items() if qid == q["id"] and r > 0}
    p_at5 = sum(1 for d in docs[:5] if d.id in rel) / 5
    rows.append({
        "qid":      q["id"],
        "p@5":      p_at5,
        "recall@5": recall_at_k(docs, qrels, q["id"]),
        "mrr":      mrr(docs, qrels, q["id"]),
        "task_type":  TASK_TYPE,
        "qrel_source": QREL_SRC,
    })
    print(f"  Q {q['id'][:8]}…  P@5={p_at5:.3f}  Recall@5={rows[-1]['recall@5']:.3f}  MRR={rows[-1]['mrr']:.3f}")

print(f"\nAvg P@5={np.mean([r['p@5'] for r in rows]):.4f}  "
      f"Recall@5={np.mean([r['recall@5'] for r in rows]):.4f}  "
      f"MRR={np.mean([r['mrr'] for r in rows]):.4f}")
print(f"\ntask_type={TASK_TYPE}  qrel_source={QREL_SRC}  ← always tag your results")
print("Done. Add DatasetConfig('fiqa', 'beir/fiqa', 0, 50, "
      "RAGTaskType.RETRIEVAL_BINARY, QrelSource.GOLD) to DATASETS to include in grid runs.")
