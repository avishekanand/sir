import os
import time
import pandas as pd
from typing import Dict, List, Set, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.langchain import LangChainRetriever
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import SimilarityEstimator, BaselineEstimator
from ragtune.data.loaders.BRIGHTLoader import BRIGHTLoader
from ragtune.utils.config import config
from ragtune.utils.console import print_header, print_step

# Configuration
DOMAINS: List[str] = os.environ.get("BRIGHT_DOMAINS", "biology,coding,mathematics").split(",")
QUERIES_PER_DOMAIN: int = int(os.environ.get("BRIGHT_QUERIES", "5"))
CANDIDATES_TOP_K = 50
RERANK_BUDGET = 10


class _OracleReranker(SimulatedReranker):
    """Gold-aware oracle reranker for smoke testing. No model download required."""
    def __init__(self):
        self._gold: set = set()

    def set_gold(self, gold_ids: set):
        self._gold = gold_ids

    def rerank(self, documents, context, strategy=None):
        return {doc.doc_id: (0.95 if doc.doc_id in self._gold else 0.3) for doc in documents}


_reranker = _OracleReranker()


def load_bright_data(domain: str) -> Tuple[
    List[Dict],                  # queries: [{"query": str, "gold_ids": List[str]}]
    List[Dict],                  # corpus:  [{"id": str, "content": str}]
]:
    """Loads queries and corpus for a BRIGHT domain via BRIGHTLoader."""
    print_step(f"Loading BRIGHT [{domain}]...")
    loader = BRIGHTLoader(task=domain, split="test")
    corpus_dict = loader.get_corpus()
    queries_dict = loader.get_queries()
    qrels = loader.get_qrels()

    # Build query list with gold_ids from qrels (relevance >= 1)
    queries: List[Dict] = []
    for qid, qtext in list(queries_dict.items())[:QUERIES_PER_DOMAIN]:
        gold_ids = [did for did, rel in qrels.get(qid, {}).items() if rel >= 1]
        queries.append({"query": qtext, "gold_ids": gold_ids})

    # Collect gold doc IDs so they are never dropped
    all_gold: Set[str] = {did for q in queries for did in q["gold_ids"]}

    # Cap corpus to 1000 docs but always include gold docs
    corpus: List[Dict] = []
    print_step("Sampling corpus for relevance...")
    for doc_id, doc_data in corpus_dict.items():
        if len(corpus) >= 1000 and doc_id not in all_gold:
            continue
        corpus.append({"id": doc_id, "content": doc_data.get("text", "")})

    return queries, corpus


def evaluate(controller: RAGtuneController, queries: List[Dict]) -> Dict:
    """Runs evaluation and returns metrics."""
    results = []
    for q in queries:
        query_str = q["query"]
        gold_ids = set(q["gold_ids"])

        _reranker.set_gold(gold_ids)
        start = time.time()
        output = controller.run(query_str)
        elapsed = time.time() - start

        found = any(doc.id in gold_ids for doc in output.documents)
        docs_reranked = output.final_budget_state.get("rerank_docs", 0)

        results.append({
            "found": found,
            "docs_reranked": docs_reranked,
            "latency": elapsed
        })

    df = pd.DataFrame(results)
    return {
        "accuracy": df["found"].mean(),
        "avg_docs_reranked": df["docs_reranked"].mean(),
        "avg_latency": df["latency"].mean()
    }


def run_benchmark():
    config.set("retrieval.original_query_depth", CANDIDATES_TOP_K)
    print_header("RAGtune Advanced Benchmarking: The BRIGHT Test")

    all_metrics = []

    for domain in DOMAINS:
        queries, corpus = load_bright_data(domain)

        print_step(f"Indexing {len(corpus)} documents...")
        docs = [Document(page_content=c["content"], metadata={"id": c["id"]}) for c in corpus]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = LangChainRetriever(vectorstore.as_retriever(search_kwargs={"k": CANDIDATES_TOP_K}))

        baseline_controller = RAGtuneController(
            retriever=retriever,
            reformulator=IdentityReformulator(),
            reranker=_reranker,
            assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
            scheduler=ActiveLearningScheduler(batch_size=CANDIDATES_TOP_K),
            estimator=BaselineEstimator(),
            budget=CostBudget.simple(docs=CANDIDATES_TOP_K, tokens=100_000, latency=600_000),
        )

        ragtune_controller = RAGtuneController(
            retriever=retriever,
            reformulator=IdentityReformulator(),
            reranker=_reranker,
            assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
            scheduler=ActiveLearningScheduler(batch_size=5),
            estimator=SimilarityEstimator(),
            budget=CostBudget.simple(docs=RERANK_BUDGET, tokens=100_000, latency=600_000),
        )

        print_step("Evaluating Baseline...")
        m_baseline = evaluate(baseline_controller, queries)
        m_baseline["domain"] = domain
        m_baseline["method"] = "Baseline (Static-Rerank-All)"

        print_step("Evaluating RAGtune...")
        m_ragtune = evaluate(ragtune_controller, queries)
        m_ragtune["domain"] = domain
        m_ragtune["method"] = "RAGtune (Iterative-Budget-10)"

        all_metrics.extend([m_baseline, m_ragtune])

    final_df = pd.DataFrame(all_metrics)
    print("\n" + "=" * 50)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 50)
    print(final_df[["domain", "method", "accuracy", "avg_docs_reranked", "avg_latency"]].to_string(index=False))
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()
