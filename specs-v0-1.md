This is the executable specification for **RAGtune v0.1**. It is designed to be implemented file-by-file.

### **1. Directory Structure**

I am plannign folder structure that is.

```text
ragtune/
├── pyproject.toml                 # Dependencies: pydantic>=2.0, tiktoken
├── README.md
├── src/
│   └── ragtune/
│       ├── __init__.py
│       ├── core/                  # ORCHESTRATION & STATE
│       │   ├── __init__.py
│       │   ├── types.py           # Data Contracts (Pydantic)
│       │   ├── budget.py          # CostTracker & Budget Logic
│       │   ├── interfaces.py      # Abstract Base Classes
│       │   ├── errors.py          # Custom Exceptions
│       │   └── controller.py      # Main Loop
│       │
│       ├── components/            # ALGORITHMS
│       │   ├── __init__.py
│       │   ├── retrievers.py      # InMemoryRetriever
│       │   ├── rerankers.py       # NoOpReranker (Identity)
│       │   ├── reformulators.py   # IdentityReformulator
│       │   ├── fusion.py          # SimpleConcatFusion (v0.1 placeholder)
│       │   └── assemblers.py      # GreedyAssembler
│       │
│       └── utils/
│           ├── __init__.py
│           └── tokenizer.py       # Token counting wrapper
│
└── tests/
    ├── conftest.py                # Fixtures: FakeRetriever, StingyBudget
    ├── unit/
    │   └── test_budget.py         # Test math & logic
    └── integration/
        └── test_controller.py     # Test degradation scenarios

```

---

### **2. Core Specifications (`src/ragtune/core/`)**

#### **A. `types.py` (The Data Contract)**

*Dependencies:* `pydantic`, `typing`, `enum`, `uuid`, `time`

```python
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class ScoredDocument(BaseModel):
    """Atomic unit of content."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0           # Current active score
    original_score: float = 0.0  # From retriever
    reranker_score: Optional[float] = None
    fusion_score: Optional[float] = None
    token_count: int = 0
    
    model_config = ConfigDict(frozen=True) # Hashable for sets

class ReformulationResult(BaseModel):
    """Output of one retrieval path."""
    original_query: str
    reformulated_query: str
    strategy: str = "identity"
    candidates: List[ScoredDocument] = Field(default_factory=list)

class TraceEvent(BaseModel):
    """Log entry for decision debugging."""
    timestamp: float = Field(default_factory=time.time)
    component: str
    action: str      # e.g. "consume", "skip", "error"
    details: Dict[str, Any]

class ControllerTrace(BaseModel):
    """Full execution history."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    events: List[TraceEvent] = Field(default_factory=list)

    def add(self, component: str, action: str, **kwargs):
        self.events.append(TraceEvent(component=component, action=action, details=kwargs))

class ControllerOutput(BaseModel):
    """Final artifact returned to user."""
    query: str
    documents: List[ScoredDocument]
    trace: ControllerTrace
    final_budget_state: Dict[str, Any]

```

#### **B. `budget.py` (The Bank)**

*Logic:* Never raise exceptions on budget exhaustion. Return `False`.

```python
from typing import Optional
import time
from ragtune.core.types import ControllerTrace
from pydantic import BaseModel

class CostBudget(BaseModel):
    max_tokens: int = 4000
    max_reranker_docs: int = 50
    max_reformulations: int = 1
    max_latency_ms: float = 2000.0

class CostTracker:
    def __init__(self, budget: CostBudget, trace: ControllerTrace):
        self.budget = budget
        self.trace = trace
        self._tokens_used = 0
        self._rerank_docs_used = 0
        self._reformulations_used = 0
        self._start_time = time.time()

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self._start_time) * 1000

    def try_consume_reformulation(self, n=1) -> bool:
        if self._reformulations_used + n <= self.budget.max_reformulations:
            self._reformulations_used += n
            self.trace.add("budget", "consume_reformulation", count=n)
            return True
        self.trace.add("budget", "deny_reformulation", reason="limit_reached")
        return False

    def try_consume_rerank(self, n_docs: int) -> bool:
        # Check Latency
        if self.elapsed_ms > self.budget.max_latency_ms:
            self.trace.add("budget", "deny_rerank", reason="latency_exceeded", elapsed=self.elapsed_ms)
            return False
        
        # Check Capacity
        remaining = self.budget.max_reranker_docs - self._rerank_docs_used
        if n_docs > remaining:
            self.trace.add("budget", "deny_rerank", reason="doc_limit_exceeded", requested=n_docs, remaining=remaining)
            return False
            
        self._rerank_docs_used += n_docs
        self.trace.add("budget", "consume_rerank", count=n_docs)
        return True

    def try_consume_tokens(self, n_tokens: int) -> bool:
        if self._tokens_used + n_tokens <= self.budget.max_tokens:
            self._tokens_used += n_tokens
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "tokens": self._tokens_used,
            "latency": self.elapsed_ms,
            "rerank_docs": self._rerank_docs_used
        }

```

#### **C. `interfaces.py` (The Plugins)**

```python
from abc import ABC, abstractmethod
from typing import List
from ragtune.core.types import ScoredDocument, ReformulationResult
from ragtune.core.budget import CostTracker

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        pass

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        pass

class BaseReformulator(ABC):
    @abstractmethod
    def generate(self, query: str, tracker: CostTracker) -> List[str]:
        pass

class BaseAssembler(ABC):
    @abstractmethod
    def assemble(self, candidates: List[ScoredDocument], tracker: CostTracker) -> List[ScoredDocument]:
        pass

```

---

### **3. Component Specifications (`src/ragtune/components/`)**

#### **A. `retrievers.py` (InMemory for v0.1)**

*Requirement:* Must be deterministic for tests.

```python
from typing import List
from ragtune.core.interfaces import BaseRetriever
from ragtune.core.types import ScoredDocument

class InMemoryRetriever(BaseRetriever):
    def __init__(self, documents: List[ScoredDocument]):
        self.docs = documents

    def retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        # Simple keyword match or just return top_k for testing
        results = [d for d in self.docs if query.lower() in d.content.lower()]
        if not results:
            results = self.docs # Fallback for "fake" tests
        return results[:top_k]

```

#### **B. `controller.py` (The Logic)**

*Requirement:* Connects components. Implements the "If budget, then X" logic.

```python
from typing import List, Optional
from ragtune.core.types import ControllerOutput, ControllerTrace, ScoredDocument, ReformulationResult
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.interfaces import *

class RAGtuneController:
    def __init__(
        self,
        retriever: BaseRetriever,
        reformulator: BaseReformulator,
        reranker: BaseReranker,
        assembler: BaseAssembler,
        budget: CostBudget
    ):
        self.retriever = retriever
        self.reformulator = reformulator
        self.reranker = reranker
        self.assembler = assembler
        self.default_budget = budget

    def run(self, query: str, override_budget: Optional[CostBudget] = None) -> ControllerOutput:
        budget = override_budget or self.default_budget
        trace = ControllerTrace()
        tracker = CostTracker(budget, trace)
        
        # 1. Reformulation
        queries = self.reformulator.generate(query, tracker)
        
        # 2. Retrieval
        reformulation_results = []
        for q in queries:
            docs = self.retriever.retrieve(q, top_k=10) # hardcoded top_k for v0.1
            reformulation_results.append(
                ReformulationResult(original_query=query, reformulated_query=q, candidates=docs)
            )
            
        # 3. Fusion (Simple flatten for v0.1)
        all_docs = []
        seen = set()
        for res in reformulation_results:
            for doc in res.candidates:
                if doc.id not in seen:
                    all_docs.append(doc)
                    seen.add(doc.id)
                    
        # 4. Reranking Gate
        if tracker.try_consume_rerank(len(all_docs)):
            processed_docs = self.reranker.rerank(all_docs, query)
        else:
            trace.add("controller", "skip_rerank", reason="budget_denied")
            processed_docs = all_docs
            
        # 5. Assembly
        final_docs = self.assembler.assemble(processed_docs, tracker)
        
        return ControllerOutput(
            query=query,
            documents=final_docs,
            trace=trace,
            final_budget_state=tracker.snapshot()
        )

```

---

### **4. Test Specifications (`tests/`)**

#### **A. `conftest.py` (The Fake World)**

This is crucial. It creates deterministic components.

```python
import pytest
from ragtune.core.types import ScoredDocument, CostBudget
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.core.interfaces import BaseReranker, BaseReformulator, BaseAssembler

class FakeReranker(BaseReranker):
    def rerank(self, documents, query):
        # Reverse list to prove it ran
        return list(reversed(documents))

class FakeReformulator(BaseReformulator):
    def generate(self, query, tracker):
        return [query]

class FakeAssembler(BaseAssembler):
    def assemble(self, candidates, tracker):
        # Just take first N that fit budget
        result = []
        for doc in candidates:
            if tracker.try_consume_tokens(doc.token_count):
                result.append(doc)
        return result

@pytest.fixture
def doc_pool():
    return [
        ScoredDocument(id=f"doc_{i}", content=f"text {i}", token_count=10)
        for i in range(100)
    ]

@pytest.fixture
def fake_retriever(doc_pool):
    return InMemoryRetriever(doc_pool)

@pytest.fixture
def fake_reranker():
    return FakeReranker()

```

#### **B. `integration/test_controller.py` (The Scenarios)**

1. **test_happy_path**: Give infinite budget. Assert output has docs and reranker ran (order reversed).
2. **test_reranker_skip**: Give `max_reranker_docs=0`. Assert reranker did NOT run (order not reversed) and Trace has "skip_rerank".
3. **test_token_limit**: Give `max_tokens=25`. Input docs size 10. Assert exactly 2 docs returned.

---

### **5. Implementation Checklist**

1. **Initialize:** `mkdir ragtune`, `uv init` (or `poetry init`), create folder structure.
2. **Core Types:** Copy `types.py`. Run `mypy` or python to check imports.
3. **Budget:** Copy `budget.py`. Write a small unit test (`test_budget.py`) to verify `try_consume` returns False correctly.
4. **Interfaces:** Copy `interfaces.py`.
5. **Components:** Implement `retrievers.py`, `reformulators.py` (identity), `assemblers.py`.
6. **Controller:** Implement `controller.py`.
7. **Tests:** Copy `conftest.py` and write `test_controller.py`. Run `pytest`.

This spec is sufficient to reach **v0.1 Exit Criteria**: A working pipeline that makes cost decisions.