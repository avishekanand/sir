This is the executable specification for **RAGtune v0.2 (Declarative Pipeline & CLI Lifecycle)**. It defines the standardized configuration schema and the CLI-managed automation introduced in v0.56.

### **1. The Declarative Schema (v0.2)**

The core shift in v0.2 is moving from "component-only" configuration to "full-lifecycle" configuration.

#### **A. Configuration Structure (`src/ragtune/config/models.py`)**

```python
class RAGtuneConfig(BaseModel):
    pipeline: PipelineConfig

class PipelineConfig(BaseModel):
    name: str
    data: Optional[DataConfig]   # Source data for indexing
    index: Optional[IndexConfig] # Target index parameters
    components: PipelineComponents
    feedback: Optional[FeedbackConfig]
    budget: BudgetConfig
```

---

### **2. The CLI-Managed Lifecycle**

v0.56 operationalizes the pipeline through three new core commands that interact with the v0.2 schema.

#### **A. Initialization & Ingestion**
- **`init --wizard`**: Interactive constructor for the v0.2 YAML.
- **`index`**: Invokes `BaseIndexer` to build storage based on `data` and `index` configs.

#### **B. Static Analysis**
- **`validate`**: 
    1. Parses config using Pydantic.
    2. Resolves all `type` keys against the `RagTuneRegistry`.
    3. Verifies path existence for `data` and `index`.

---

### **3. Orchestration Updates**

The controller and interfaces are updated to support the more complex configurations defined in v0.2.

#### **A. Structured Feedback (`interfaces.py`)**
Allows declarative "stop conditions" to be injected into the run loop.

```python
class BaseFeedback(ABC):
    @abstractmethod
    def should_stop(self, state: Dict[str, Any], budget: Any, estimates: Dict[str, float]) -> Tuple[bool, str]:
        pass
```

#### **B. Composite Components**
The `ConfigLoader` supports recursive instantiation. If a configuration block is a list, it is treated as a composite or multi-strategy component (e.g., `CompositeEstimator`).

---

### **4. Verification Checklist (v0.2)**

1.  **Schema Integrity**: Does `ragtune validate` correctly identify missing required fields?
2.  **Registry Resolution**: Does the CLI correctly load all adapters and components before validating `type` strings?
3.  **Indexing Automation**: Does `ragtune index` successfully create the directory structure defined in the YAML?
4.  **Feedback Gating**: Does the `RAGtuneController` respect the stop signal from a `BaseFeedback` component?
