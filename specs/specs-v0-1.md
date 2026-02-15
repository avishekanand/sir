This is the executable specification for **RAGtune v0.1+ (Active Loop Architecture)**. It identifies the core data contracts and orchestration logic for the iterative scheduler.

### **1. Directory Structure**

```text
ragtune/
├── pyproject.toml                 # Dependencies: pydantic>=2.0, tiktoken
├── README.md
├── src/
│   └── ragtune/
│       ├── core/                  # ORCHESTRATION & STATE
│       │   ├── types.py           # Data Contracts (BatchProposal, ScoredDocument)
│       │   ├── budget.py          # CostTracker & Budget Logic
│       │   ├── interfaces.py      # BaseScheduler, BaseReranker, etc.
│       │   └── controller.py      # Iterative "While" Loop
│       │
│       ├── components/            # ALGORITHMS
│       │   ├── schedulers.py      # ActiveLearningScheduler
│       │   ├── estimators.py      # UtilityEstimator (Brain)
│       │   ├── retrievers.py      # InMemoryRetriever
│       │   ├── rerankers.py       # SimulatedReranker (for testing)
│       │   └── assemblers.py      # GreedyAssembler
│       │
│       └── utils/
│           └── tokenizer.py       # Token counting
│
└── tests/
    ├── conftest.py                # Fixtures: FakeScheduler, FakeReranker
    ├── unit/
    └── integration/
        └── test_controller.py     # Loop mechanics & feedback tests
```

---

### **2. Core Specifications (`src/ragtune/core/`)**

#### **A. `types.py` (The Data Contract)**

```python
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class RerankStrategy(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"

class BatchProposal(BaseModel):
    """The Scheduler's next move."""
    document_indices: List[int]
    strategy: RerankStrategy
    estimated_utility: float = 0.0

class ScoredDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    reranker_score: Optional[float] = None
    token_count: int = 0
```

#### **B. `interfaces.py` (The Plugins)**

```python
class BaseScheduler(ABC):
    @abstractmethod
    def propose_next_batch(
        self, 
        pool: List[ScoredDocument], 
        processed_indices: List[int], 
        tracker: CostTracker
    ) -> Optional[BatchProposal]:
        pass

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        pass
```

---

### **3. Orchestration Logic (`core/controller.py`)**

The controller implements the **Iterative Feedback Loop**:

```python
class RAGtuneController:
    def run(self, query: str, budget: CostBudget) -> ControllerOutput:
        pool = self.retriever.retrieve(query)
        processed_indices = []
        
        while True:
            # 1. Ask Scheduler for next move
            proposal = self.scheduler.propose_next_batch(pool, processed_indices, tracker)
            if proposal is None: break
            
            # 2. Rerank the proposed batch
            batch_docs = [pool[i] for i in proposal.document_indices]
            if tracker.try_consume_rerank(len(batch_docs)):
                reranked_batch = self.reranker.rerank(batch_docs, query)
                
                # 3. Update pool (Feedback loop)
                for idx, new_doc in zip(proposal.document_indices, reranked_batch):
                    pool[idx] = new_doc
                    processed_indices.append(idx)
            else:
                break
        
        return self.assembler.assemble(pool, tracker)
```

---

### **4. Component Logic (`src/ragtune/components/`)**

#### **A. `estimators.py` (The Feedback Mechanism)**
Predicts unranked document utility by boosting candidates that share metadata (e.g., `section`, `source`) with high-scoring reranked documents.

#### **B. `schedulers.py` (Adaptive Sampling)**
Uses the `UtilityEstimator` to select the top-K unranked candidates for the next reranking round.

---

### **5. Verification Checklist**

1. **Loop Integrity**: Verify the loop stops exactly when budget is hit (e.g., `max_reranker_docs`).
2. **Feedback Propagation**: Verify that a "winner" in Round 1 boosts its neighbors for Round 2.
3. **Traceability**: Ensure `rerank_batch` events include `doc_ids` and `utility` scores.