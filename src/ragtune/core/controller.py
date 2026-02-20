from typing import List, Optional, Dict, Any
from ragtune.core.types import ControllerOutput, ControllerTrace, ScoredDocument, BatchProposal, RAGtuneContext, ItemState
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.interfaces import BaseRetriever, BaseReformulator, BaseReranker, BaseAssembler, BaseScheduler, BaseEstimator, BaseFeedback
from ragtune.core.pool import CandidatePool, PoolItem
from ragtune.utils.config import config

class RAGtuneController:
    def __init__(
        self,
        retriever: BaseRetriever,
        reformulator: BaseReformulator,
        reranker: BaseReranker,
        assembler: BaseAssembler,
        scheduler: BaseScheduler,
        estimator: BaseEstimator,
        budget: Optional[CostBudget] = None,
        feedback: Optional[BaseFeedback] = None
    ):
        self.retriever = retriever
        self.reformulator = reformulator
        self.reranker = reranker
        self.assembler = assembler
        self.scheduler = scheduler
        self.estimator = estimator
        self.budget = budget or CostBudget()
        self.feedback = feedback
        self._reformulation_cache: Dict[str, List[str]] = {}

    def run(self, query: str, override_budget: Optional[CostBudget] = None) -> ControllerOutput:
        budget = override_budget or self.budget
        trace = ControllerTrace()
        tracker = CostTracker(budget, trace)
        context = RAGtuneContext(query=query, tracker=tracker)
        
        # 1. Original Retrieval
        d_orig = config.get("retrieval.original_query_depth", 10)
        d_ref = config.get("retrieval.depth_per_reformulation", 5)
        max_pool = config.get("retrieval.max_pool_size", 50)
        
        pool = CandidatePool()
        if tracker.try_consume_retrieval():
            docs_orig = self.retriever.retrieve(context, top_k=d_orig)
            pool.add_items(docs_orig, source="original")
        
        # 2. Reformulation Decision (Gated by Estimator)
        queries = []
        if self.estimator.needs_reformulation(context, pool):
            if query in self._reformulation_cache:
                queries = self._reformulation_cache[query]
                trace.add("controller", "reformulation_cache_hit", query=query)
            else:
                queries = self.reformulator.generate(context)
                self._reformulation_cache[query] = queries
                
        if not queries:
            queries = []
        
        # 3. Supplemental Retrieval (with budget check per round)
        seen_queries = {query.lower().strip()}
        for q_idx, q in enumerate(queries):
            q_norm = q.lower().strip()
            if q_norm in seen_queries:
                continue
            seen_queries.add(q_norm)
            
            if not tracker.try_consume_retrieval():
                trace.add("controller", "retrieval_skipped", query=q, reason="budget_exhausted")
                continue

            q_context = context.model_copy(update={"query": q})
            docs_ref = self.retriever.retrieve(q_context, top_k=d_ref)
            pool.add_items(docs_ref, source=f"rewrite_{q_idx}")
        
        # Enforce pool cap
        pool.enforce_cap(max_pool)
        metrics = pool.get_metrics()
        trace.add("controller", "pool_init", count=len(pool), reformulations=queries, metrics=metrics)

        # 3. Iterative Loop
        while not tracker.is_exhausted():
            # A. Valorization (Estimator determines priorities)
            est_outputs = self.estimator.value(pool, context)
            priorities = {k: v.priority for k, v in est_outputs.items()}
            pool.apply_priorities(priorities)

            # Aggregate estimator metadata so feedback can read it (e.g. ReformIR learned weights)
            estimates: Dict[str, Any] = {}
            for out in est_outputs.values():
                estimates.update(out.metadata)

            # B. Feedback/Stop Check (runs after estimator so feedback sees current-iteration data)
            if self.feedback:
                should_stop, reason = self.feedback.should_stop(pool.get_metrics(), tracker.remaining_view(), estimates)
                if should_stop:
                    trace.add("controller", "feedback_stop", reason=reason)
                    break
            
            # C. Scheduling (Policy selects batch)
            proposal = self.scheduler.select_batch(pool, tracker.remaining_view())
            if not proposal:
                break
                
            # D. Transition to IN_FLIGHT
            pool.transition(proposal.doc_ids, ItemState.IN_FLIGHT)
            
            # E. Execution (Reranker processes batch)
            try:
                batch_items = pool.get_items(proposal.doc_ids)
                results = self.reranker.rerank(batch_items, context, strategy=proposal.strategy)
                # F. Update scores and move to RERANKED
                pool.update_scores(results, strategy=proposal.strategy, expected_ids=proposal.doc_ids)
                tracker.consume(proposal.expected_cost)
                
                trace.add(
                    "controller", "rerank_batch", 
                    count=len(proposal.doc_ids), 
                    strategy=proposal.strategy,
                    doc_ids=proposal.doc_ids
                )
            except Exception as e:
                trace.add("controller", "rerank_error", error=str(e), doc_ids=proposal.doc_ids)
                pool.transition(proposal.doc_ids, ItemState.DROPPED)
            
        # 4. Assembly
        active_items = pool.get_active_items()
        final_docs = self.assembler.assemble(active_items, context)
        
        return ControllerOutput(
            query=query,
            documents=final_docs,
            trace=trace,
            final_budget_state=tracker.snapshot()
        )
