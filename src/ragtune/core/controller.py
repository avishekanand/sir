from typing import List, Optional, Dict, Any
from ragtune.core.types import ControllerOutput, ControllerTrace, ScoredDocument, ReformulationResult, BatchProposal, RAGtuneContext, ItemState
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.interfaces import BaseRetriever, BaseReformulator, BaseReranker, BaseAssembler, BaseScheduler, BaseEstimator
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
        budget: Optional[CostBudget] = None
    ):
        self.retriever = retriever
        self.reformulator = reformulator
        self.reranker = reranker
        self.assembler = assembler
        self.scheduler = scheduler
        self.estimator = estimator
        self.budget = budget or CostBudget()

    def run(self, query: str, override_budget: Optional[CostBudget] = None) -> ControllerOutput:
        budget = override_budget or self.budget
        trace = ControllerTrace()
        tracker = CostTracker(budget, trace)
        context = RAGtuneContext(query=query, tracker=tracker)
        
        # 1. Reformulation
        queries = self.reformulator.generate(context)
        if not queries:
            queries = [query]
        
        # 2. Retrieval & Pool Initialization
        multiplier = config.get("retrieval.pool_multiplier", 2.0)
        default_top_k = config.get("retrieval.default_top_k", 10)
        max_docs = budget.limits.get("rerank_docs", default_top_k)
        top_k = int(max_docs * multiplier)
        
        raw_items = []
        seen_ids = set()
        for q_idx, q in enumerate(queries):
            q_context = context.model_copy(update={"query": q})
            docs = self.retriever.retrieve(q_context, top_k=top_k)
            for d_idx, doc in enumerate(docs):
                if doc.id not in seen_ids:
                    item = PoolItem(
                        doc_id=doc.id,
                        content=doc.content,
                        metadata=doc.metadata,
                        sources={f"query_{q_idx}": doc.score},
                        initial_rank=d_idx
                    )
                    raw_items.append(item)
                    seen_ids.add(doc.id)
                else:
                    # Update sources for duplicates
                    for item in raw_items:
                        if item.doc_id == doc.id:
                            item.sources[f"query_{q_idx}"] = doc.score
                            break
                            
        pool = CandidatePool(raw_items)
        trace.add("controller", "pool_init", count=len(pool))

        # 3. Iterative Loop (RAGtune v0.54 logic)
        while not tracker.is_exhausted():
            # A. Valorization (Estimator determines priorities)
            priorities = self.estimator.value(pool, context)
            pool.apply_priorities(priorities)
            
            # B. Scheduling (Policy selects batch)
            proposal = self.scheduler.select_batch(pool, tracker.remaining_view())
            if not proposal:
                break
                
            # C. Transition to IN_FLIGHT
            pool.transition(proposal.doc_ids, ItemState.IN_FLIGHT)
            
            # D. Execution (Reranker processes batch)
            try:
                batch_items = pool.get_items(proposal.doc_ids)
                results = self.reranker.rerank(batch_items, context, strategy=proposal.strategy)
                # E. Update scores and move to RERANKED
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
                # v0.54 Failure Rule: Drop failed docs
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

    async def arun(self, query: str, override_budget: Optional[CostBudget] = None) -> ControllerOutput:
        budget = override_budget or self.budget
        trace = ControllerTrace()
        tracker = CostTracker(budget, trace)
        context = RAGtuneContext(query=query, tracker=tracker)
        
        # 1. Reformulation
        queries = await self.reformulator.agenerate(context)
        if not queries:
            queries = [query]
        
        # 2. Retrieval & Pool Initialization
        multiplier = config.get("retrieval.pool_multiplier", 2.0)
        default_top_k = config.get("retrieval.default_top_k", 10)
        max_docs = budget.limits.get("rerank_docs", default_top_k)
        top_k = int(max_docs * multiplier)
        
        raw_items = []
        seen_ids = set()
        for q_idx, q in enumerate(queries):
            q_context = context.model_copy(update={"query": q})
            docs = await self.retriever.aretrieve(q_context, top_k=top_k)
            for d_idx, doc in enumerate(docs):
                if doc.id not in seen_ids:
                    item = PoolItem(
                        doc_id=doc.id,
                        content=doc.content,
                        metadata=doc.metadata,
                        sources={f"query_{q_idx}": doc.score},
                        initial_rank=d_idx
                    )
                    raw_items.append(item)
                    seen_ids.add(doc.id)
                else:
                    for item in raw_items:
                        if item.doc_id == doc.id:
                            item.sources[f"query_{q_idx}"] = doc.score
                            break
                            
        pool = CandidatePool(raw_items)
        trace.add("controller", "pool_init", count=len(pool))

        # 3. Iterative Loop
        while not tracker.is_exhausted():
            # A. Valorization
            priorities = self.estimator.value(pool, context)
            pool.apply_priorities(priorities)
            
            # B. Scheduling
            proposal = await self.scheduler.aselect_batch(pool, tracker.remaining_view())
            if not proposal:
                break
                
            # C. Transition to IN_FLIGHT
            pool.transition(proposal.doc_ids, ItemState.IN_FLIGHT)
            
            # D. Execution
            try:
                batch_items = pool.get_items(proposal.doc_ids)
                results = await self.reranker.arerank(batch_items, context, strategy=proposal.strategy)
                # E. Update scores and move to RERANKED
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
        final_docs = await self.assembler.aassemble(active_items, context)
        
        return ControllerOutput(
            query=query,
            documents=final_docs,
            trace=trace,
            final_budget_state=tracker.snapshot()
        )
