# RAGtune Core 0.54 Design Specification (Final)

This document formalizes the iterative, state-based reranking architecture for RAGtune v0.54.

## 1. Data Model

### Pool States & State Machine
The `CandidatePool` enforces a strict state machine. Illegal transitions raise `IllegalTransitionError`.

| Current State | Target State | Trigger |
| :--- | :--- | :--- |
| `CANDIDATE` | `IN_FLIGHT` | `pool.transition(ids, IN_FLIGHT)` |
| `CANDIDATE` | `DROPPED` | Explicit pruning or budget exhaustion |
| `IN_FLIGHT` | `RERANKED` | `pool.update_scores(scores)` |
| `IN_FLIGHT` | `DROPPED` | Execution failure (No retries in v0.54) |
| `RERANKED` | `DROPPED` | Post-rerank pruning |

```python
from enum import Enum

class ItemState(Enum):
    CANDIDATE = "candidate"  # Eligible for scheduling
    IN_FLIGHT = "in_flight"  # Currently moving through a reranker
    RERANKED = "reranked"    # Final reranker_score available
    DROPPED = "dropped"      # Excluded from final results
```

### PoolItem
The stable unit of work. Identity is always `doc_id`.
```python
class PoolItem(BaseModel):
    doc_id: str                      # Primary Key
    content: str
    metadata: Dict[str, Any]      
    
    state: ItemState = ItemState.CANDIDATE
    
    # Provenance
    sources: Dict[str, float]        # {strategy: score}
    initial_rank: int                # From first retriever
    
    # Iterative State
    priority_value: float = 0.0      # Set by Estimator
    reranker_score: Optional[float] = None
    reranker_strategy: Optional[str] = None

    def final_score(self) -> float:
        """Precedence: Reranker > Estimator > Retrieval Baseline."""
        if self.reranker_score is not None:
            return self.reranker_score
        if self.priority_value > 0:
            return self.priority_value
        return max(self.sources.values()) if self.sources else 0.0
```

### CandidatePool
Internal registry for all items.
```python
class CandidatePool:
    def transition(self, doc_ids: List[str], target: ItemState):
        """Validates current state before moving to target."""
        # CANDIDATE -> (IN_FLIGHT, DROPPED)
        # IN_FLIGHT -> (RERANKED, DROPPED)
        # Else: raise IllegalTransitionError

    def update_scores(self, scores: Dict[str, float], strategy: str):
        """Updates scores only for docs in IN_FLIGHT, then moves to RERANKED."""
        for did, score in scores.items():
            item = self._items[did]
            if item.state != ItemState.IN_FLIGHT:
                raise IllegalTransitionError(did)
            item.reranker_score = score
            item.reranker_strategy = strategy
            item.state = ItemState.RERANKED

    def get_active_items(self) -> List[PoolItem]:
        """Returns docs that are CANDIDATE or RERANKED (excludes IN_FLIGHT/DROPPED)."""
        return [it for it in self._items.values() 
                if it.state in (ItemState.CANDIDATE, ItemState.RERANKED)]
```

---

## 2. Global Interfaces

### Cost & Budget
```python
class CostObject(BaseModel):
    tokens: int = 0
    docs: int = 0
    calls: int = 0

class RemainingBudgetView(BaseModel):
    remaining_tokens: int
    remaining_rerank_docs: int
    remaining_rerank_calls: int
    assembly_token_buffer: int
```

### Proposels & Components
```python
class BatchProposal(BaseModel):
    doc_ids: List[str]
    strategy: str
    expected_cost: CostObject

class BaseEstimator(ABC):
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, float]:
        """Returns {doc_id: priority_value}. Controller writes them to items."""

class BaseScheduler(ABC):
    def select_batch(self, pool: CandidatePool, budget: RemainingBudgetView) -> Optional[BatchProposal]:
        """Must be stable under ties (tie-break by initial_rank then doc_id). Let's use it for tie-breaking."""
```

---

## 3. Controller Iterative Loop

```python
def run(self, query: str):
    pool = self.retriever.retrieve_pool(query)
    context = RAGtuneContext(query=query, tracker=self.budget.create_tracker())

    while not context.tracker.is_exhausted():
        # 1. Estimation (Controller updates pool from estimator output)
        priorities = self.estimator.value(pool, context)
        pool.apply_priorities(priorities)

        # 2. Scheduling
        proposal = self.scheduler.select_batch(pool, context.tracker.remaining_view())
        if not proposal:
            break

        # 3. State Change & Execution
        pool.transition(proposal.doc_ids, ItemState.IN_FLIGHT)
        try:
            results = self.reranker.rerank(pool.get_items(proposal.doc_ids), 
                                          strategy=proposal.strategy)
            # results is Dict[doc_id, score]
            pool.update_scores(results, strategy=proposal.strategy)
            context.tracker.consume(proposal.expected_cost)
        except Exception:
            # v0.54: Failure -> DROPPED (No retries)
            pool.transition(proposal.doc_ids, ItemState.DROPPED)

    # 4. Assembly
    active_docs = pool.get_active_items()
    # Assembler sorts by item.final_score() then enforces token budget
    return self.assembler.assemble(active_docs, context)
```

---

## 4. Invariants

1.  **Stable Identity**: `doc_id` mapping is immutable for the request lifecycle.
2.  **Transition Integrity**: Any attempt to move a `RERANKED` doc to `IN_FLIGHT` or a `DROPPED` doc to `CANDIDATE` raises an exception.
3.  **Monotonicity (Eligibility)**: Once a doc leaves `CANDIDATE`, it never re-enters. (Exception Handling -> `DROPPED`).
4.  **Budget View Fairness**: Scheduler only sees a snapshot; Controller is the source of truth for consumption.
5.  **Assembly Source**: Assembler only sees docs in `CANDIDATE` or `RERANKED` states.
6.  **Tie-Breaking**: All selectors must be deterministic (stable sort by `initial_rank`, `doc_id`).
7.  **No Double Counting**: `update_scores` only processes docs that were actually `IN_FLIGHT`.
8.  **Final Score Precedence**: `reranker_score` > `priority_value` > `initial_score`.





# RAGtune Core: Iterative Pool-Valued Reranking (v0.54 but is more text and will be replaced by the ad-hoc rationales above)

## Goal

RAGtune’s core problem is **budgeted ranking under uncertainty**: we start with a large candidate pool from cheap retrieval, but we can only afford expensive reranking on a small subset. The system must decide—iteratively and cost-aware—**which documents are worth reranking next**, and stop gracefully when budget is exhausted.

This core design formalizes RAGtune as a **pool-valued decision process**:

1. Build a candidate pool (possibly unioned across reformulations/retrievers).
2. An **Estimator** assigns a *priority value* to every eligible candidate given current evidence.
3. A **Scheduler** selects the next rerank batch from eligible items.
4. The **Controller** moves items through states and consumes budget.
5. Repeat until no budget or no useful work remains.
6. Assemble final context from active items using a consistent final scoring rule.

---

## Key Abstractions and Rationale

### CandidatePool (mutable state + invariants)

**Why it exists:**  
A static list + “processed indices” is not a pool. In real pipelines we reorder, deduplicate, union results, and attach provenance. If identity is tied to indices, correctness collapses.

**What it guarantees:**
- Stable identity by `doc_id`.
- Explicit lifecycle states:
  - `CANDIDATE` → eligible for scheduling
  - `IN_FLIGHT` → selected for reranking, temporarily removed from eligibility
  - `RERANKED` → reranker_score available
  - `DROPPED` → excluded
- Strict state machine transitions (no silent illegal transitions).
- O(1) item access by `doc_id`.

**Why state matters:**  
State ensures you can never “double rerank” a doc in the same request, and it makes iterative selection well-defined.

---

### Estimator (values the pool)

**Design intent:** The estimator is the “brain” that **scores the remaining pool** at each iteration.

- Input: pool snapshot (eligible + reranked items), request context, and optional diagnostics.
- Output: `priority_value` for eligible docs (and optionally diagnostics used for tracing).
- Crucially: estimator **does not schedule** and **does not consume budget**.

**Why separate estimator from scheduler?**
- Estimator answers: “What is the current value of each candidate?”
- Scheduler answers: “Which ones do we process next given budget and policy?”
- This separation allows:
  - swapping estimators without rewriting scheduling policy,
  - using the same estimator with multiple scheduling strategies (greedy vs uncertainty),
  - clean testing (value correctness vs selection correctness).

**Estimator families:**
- **Baseline:** `priority_value = retrieval_score` (cheap and stable)
- **Dynamic:** re-estimate remaining candidates after each rerank using evidence from winners (e.g., metadata overlap, embedding similarity, calibrated mapping from retrieval→rerank scores)
- **Uncertainty-driven:** prioritize docs where reranking is likely to change the top ordering (entropy/variance, small score margins, duplicates, disagreement)

---

### Scheduler (pops the next batch)

**Design intent:** The scheduler is a policy that converts values into an action under constraints.

- Input: pool + immutable remaining budget view.
- Output: `BatchProposal(doc_ids, strategy, expected_cost, diagnostics)`
- It **must only select** from `CANDIDATE`.
- It **does not mutate budget** and **does not perform transitions**.

**Why scheduler chooses strategy per batch?**
RAGtune may support multiple rerankers (e.g., Cross-Encoder, LLM). The scheduler is where we decide:
- “When do we escalate to LLM?”
- “How big is the next batch given remaining rerank_docs/calls/latency?”
- “Do we stop early?”

This makes staged degradation and adaptive escalation clean and testable.

---

### Controller (single owner of budget and state transitions)

**Design intent:** The controller is the only component that:
- consumes budget,
- executes reranking,
- transitions pool states,
- enforces invariants.

This prevents budget double-counting and state corruption.

---

## Architecture Diagram (Mermaid)

```mermaid
flowchart TD
  U[User query] --> R[Retrievers and optional reformulations]
  R --> P[CandidatePool keyed by doc_id]
  P --> E[Estimator assigns priority_value]
  E --> S[Scheduler selects next batch using remaining budget view]
  S --> C[Controller owns budget and state transitions]

  C --> IF[Transition selected docs to IN_FLIGHT]
  IF --> RR[Reranker runs chosen strategy]
  RR --> UP[Update reranker_score and transition to RERANKED]
  UP --> P

  C --> A[Assembler selects final context under token budget]
  A --> O[Final context and trace]
  ```