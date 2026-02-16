from typing import List, Optional
from ragtune.core.types import ControllerOutput, ControllerTrace, ScoredDocument, ReformulationResult, BatchProposal, RAGtuneContext
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.interfaces import BaseRetriever, BaseReformulator, BaseReranker, BaseAssembler, BaseScheduler
from ragtune.utils.config import config

class RAGtuneController:
    def __init__(
        self,
        retriever: BaseRetriever,
        reformulator: BaseReformulator,
        reranker: BaseReranker,
        assembler: BaseAssembler,
        scheduler: BaseScheduler,
        budget: Optional[CostBudget] = None
    ):
        self.retriever = retriever
        self.reformulator = reformulator
        self.reranker = reranker
        self.assembler = assembler
        self.scheduler = scheduler
        self.default_budget = budget or CostBudget()

    def run(self, query: str, override_budget: Optional[CostBudget] = None) -> ControllerOutput:
        budget = override_budget or self.default_budget
        trace = ControllerTrace()
        tracker = CostTracker(budget, trace)
        context = RAGtuneContext(query=query, tracker=tracker)
        
        # 1. Reformulation
        queries = self.reformulator.generate(context)
        if not queries:
            queries = [query]
        
        # 2. Retrieval
        reformulation_results = []
        # Dynamically determine top_k based on budget and configured multiplier
        multiplier = config.get("retrieval.pool_multiplier", 2.0)
        default_top_k = config.get("retrieval.default_top_k", 10)
        
        max_docs = budget.limits.get("rerank_docs", default_top_k)
        top_k = int(max_docs * multiplier)
        
        for q in queries:
            docs = self.retriever.retrieve(context, top_k=top_k)
            reformulation_results.append(
                ReformulationResult(original_query=query, reformulated_query=q, candidates=docs)
            )
            
        # 3. Fusion
        pool = []
        seen = set()
        for res in reformulation_results:
            for doc in res.candidates:
                if doc.id not in seen:
                    pool.append(doc)
                    seen.add(doc.id)
                    
        # 4. Iterative Loop
        processed_indices = []
        while True:
            proposal = self.scheduler.propose_next_batch(
                pool, processed_indices, context
            )
            
            if proposal is None:
                break
                
            batch_docs = [pool[i] for i in proposal.document_indices]
            
            if tracker.try_consume_rerank(len(batch_docs)):
                reranked_batch = self.reranker.rerank(batch_docs, context, strategy=proposal.strategy)
                
                for idx, new_doc in zip(proposal.document_indices, reranked_batch):
                    pool[idx] = new_doc
                    processed_indices.append(idx)
                    
                trace.add(
                    "controller", "rerank_batch", 
                    count=len(batch_docs), 
                    strategy=proposal.strategy,
                    doc_ids=[pool[i].id for i in proposal.document_indices],
                    utility=proposal.estimated_utility
                )
            else:
                trace.add("controller", "skip_batch", reason="budget_denied")
                break
            
        processed_docs = pool
            
        # 5. Assembly
        final_docs = self.assembler.assemble(processed_docs, context)
        
        return ControllerOutput(
            query=query,
            documents=final_docs,
            trace=trace,
            final_budget_state=tracker.snapshot()
        )

    async def arun(self, query: str, override_budget: Optional[CostBudget] = None) -> ControllerOutput:
        budget = override_budget or self.default_budget
        trace = ControllerTrace()
        tracker = CostTracker(budget, trace)
        context = RAGtuneContext(query=query, tracker=tracker)
        
        # 1. Reformulation
        queries = await self.reformulator.agenerate(context)
        if not queries:
            queries = [query]
        
        # 2. Retrieval
        reformulation_results = []
        multiplier = config.get("retrieval.pool_multiplier", 2.0)
        default_top_k = config.get("retrieval.default_top_k", 10)
        
        max_docs = budget.limits.get("rerank_docs", default_top_k)
        top_k = int(max_docs * multiplier)
        
        for q in queries:
            docs = await self.retriever.aretrieve(context, top_k=top_k)
            reformulation_results.append(
                ReformulationResult(original_query=query, reformulated_query=q, candidates=docs)
            )
            
        # 3. Fusion
        pool = []
        seen = set()
        for res in reformulation_results:
            for doc in res.candidates:
                if doc.id not in seen:
                    pool.append(doc)
                    seen.add(doc.id)
                    
        # 4. Iterative Loop
        processed_indices = []
        while True:
            proposal = await self.scheduler.apropose_next_batch(
                pool, processed_indices, context
            )
            
            if proposal is None:
                break
                
            batch_docs = [pool[i] for i in proposal.document_indices]
            
            if tracker.try_consume_rerank(len(batch_docs)):
                reranked_batch = await self.reranker.arerank(batch_docs, context, strategy=proposal.strategy)
                
                for idx, new_doc in zip(proposal.document_indices, reranked_batch):
                    pool[idx] = new_doc
                    processed_indices.append(idx)
                    
                trace.add(
                    "controller", "rerank_batch", 
                    count=len(batch_docs), 
                    strategy=proposal.strategy,
                    doc_ids=[pool[i].id for i in proposal.document_indices],
                    utility=proposal.estimated_utility
                )
            else:
                trace.add("controller", "skip_batch", reason="budget_denied")
                break
            
        processed_docs = pool
            
        # 5. Assembly
        final_docs = await self.assembler.aassemble(processed_docs, context)
        
        return ControllerOutput(
            query=query,
            documents=final_docs,
            trace=trace,
            final_budget_state=tracker.snapshot()
        )
