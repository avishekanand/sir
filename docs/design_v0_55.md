# RAGtune Core 0.55 Design Specification: LLM-based Query Rewriting

This document formalizes the LLM-based query reformulation and multi-round retrieval architecture for RAGtune v0.55.

## 1. Goal
The primary objective of v0.55 is to improve retrieval recall by generating multiple "views" of a user's query using an LLM. By retrieving documents for both the original query and its reformulations, RAGtune can build a more comprehensive `CandidatePool` for the iterative reranking loop.

## 2. Architecture Overview

### Query Rewriting Pipeline
1.  **Original Retrieval**: Retrieve $D_{orig}$ documents for the user's original query $Q_{orig}$.
2.  **LLM Reformulation**: Use an LLM to generate $m$ distinct reformulations $Q_1, Q_2, ..., Q_m$.
3.  **Reformulation Retrieval**: For each $Q_i$, retrieve $D_{ref}$ documents.
4.  **Pool Aggregation**: Union all retrieved documents (preserving provenance and using `doc_id` for deduplication) into a single `CandidatePool`.
5.  **Iterative Loop**: Proceed with the v0.54 Estimator-Scheduler-Reranker loop on the aggregated pool.

## 3. Configuration (YAML)

New variables in `defaults.yaml`:

```yaml
retrieval:
  num_reformulations: 2              # m
  depth_per_reformulation: 5         # D_ref
  original_query_depth: 10           # D_orig
```

New prompt in `prompts.yaml`:

```yaml
reformulation:
  llm_rewrite:
    system: "You are a search expert. Your goal is to rewrite the user's query into different variations to improve retrieval."
    user: "Original Query: {query}\nGenerate {m} different search queries that cover different aspects or synonyms of the original query. Output as a JSON list of strings only."
```

## 4. Component Interface

### LLMReformulator
A new component registered as `llm_rewrite`.

-   **Input**: `RAGtuneContext` (containing the original query).
-   **Output**: List of $m$ rewritten strings (excluding the original query).
-   **Logic**:
    -   Fetches `num_reformulations` from config.
    -   Fetches `llm_rewrite` prompt template.
    -   Calls LLM.
    -   Parses result into a list.

## 5. Controller Execution (Pseudo-code)

```python
def run(self, query: str):
    tracker = self.budget.create_tracker()
    context = RAGtuneContext(query=query, tracker=tracker)
    
    # 1. Original Retrieval
    d_orig = config.get("retrieval.original_query_depth", 10)
    docs_orig = self.retriever.retrieve(context, top_k=d_orig)
    
    # 2. Reformulation
    reformulations = self.reformulator.generate(context) # returns [Q1, Q2, ...]
    
    # 3. Aggregation
    raw_items = []
    # Add docs from original query
    self._add_to_raw_items(raw_items, docs_orig, source="original")
    
    # 4. Supplemental Retrieval
    d_ref = config.get("retrieval.depth_per_reformulation", 5)
    for i, q_rewritten in enumerate(reformulations):
        q_context = context.model_copy(update={"query": q_rewritten})
        docs_ref = self.retriever.retrieve(q_context, top_k=d_ref)
        self._add_to_raw_items(raw_items, docs_ref, source=f"rewrite_{i}")
        
    pool = CandidatePool(raw_items)
    
    # ... Continue with Iterative Loop (v0.54)
```

## 6. Invariants & Rules
1.  **Originality Bias**: The original query remains the primary source, typically with a higher depth ($D_{orig} \geq D_{ref}$).
2.  **Deduplication**: `doc_id` remains the stable identifier. Documents appearing in multiple retrieval rounds have their `sources` provenance updated.
3.  **Budget Awareness**: Reformulation calls and supplemental retrieval rounds must be tracked against the budget.
4.  **Graceful Degradation**: If the LLM reformulation fails, the system proceeds with only the original retrieval results.
