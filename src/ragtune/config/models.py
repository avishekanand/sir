from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator

class ComponentConfig(BaseModel):
    type: str = "noop"
    params: Dict[str, Any] = Field(default_factory=dict)

class DataConfig(BaseModel):
    collection_path: str
    collection_format: str
    id_field: str = "doc_id"
    text_field: str = "content"
    metadata_fields: List[str] = Field(default_factory=lambda: ["source"])

class ModelConfig(BaseModel):
    # HuggingFace model ID or pyterrier_dr shorthand
    # (e.g. "sentence-transformers/all-MiniLM-L6-v2", "qwen3", "bge-m3").
    # This is identity only — how it runs belongs in IndexConfig.params.
    name: Optional[str] = None

class IndexConfig(BaseModel):
    # "sparse" → always PyTerrier BM25 (no encoder needed)
    # "dense"  → embedding-based; requires backend + model
    type: str = "sparse"

    # Where the built index is written to (and loaded from).
    index_path: str = "indexes/default"

    # ── Dense-only fields ──────────────────────────────────────────────────
    # Ignored when type = "sparse".

    # Registry key for the dense backend: "faiss" | "flex"
    backend: Optional[str] = None

    # Which model to use (identity only).
    model: Optional[ModelConfig] = None

    # Runtime/behavior knobs forwarded as-is to the resolved encoder's
    # constructor — valid keys differ per backend/family, e.g.:
    #   common:        device, batch_size, max_length
    #   Qwen3Encoder:   use_fp16, task_description, add_instruction_to_query
    #   GenericHFEncoder: pooling, query_prefix, doc_prefix, fp16, normalize
    params: Dict[str, Any] = Field(default_factory=dict)

class FeedbackConfig(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class PipelineComponents(BaseModel):
    retriever: ComponentConfig = Field(default_factory=lambda: ComponentConfig(type="bm25"))
    reformulator: ComponentConfig = Field(default_factory=lambda: ComponentConfig(type="identity"))
    reranker: ComponentConfig = Field(default_factory=lambda: ComponentConfig(type="noop"))
    assembler: ComponentConfig = Field(default_factory=lambda: ComponentConfig(type="greedy"))
    scheduler: ComponentConfig = Field(default_factory=lambda: ComponentConfig(type="graceful-degradation"))
    estimator: Union[ComponentConfig, List[ComponentConfig]] = Field(default_factory=lambda: ComponentConfig(type="baseline"))

class BudgetConfig(BaseModel):
    limits: Dict[str, float] = Field(default_factory=lambda: {
        "tokens": 4000,
        "rerank_docs": 50,
        "retrieval_calls": 5,
        "latency_ms": 2000.0
    })

class PipelineConfig(BaseModel):
    name: str = "My RAGtune Pipeline"
    data: Optional[DataConfig] = None
    index: Optional[IndexConfig] = None
    components: PipelineComponents = Field(default_factory=PipelineComponents)
    feedback: Optional[FeedbackConfig] = None
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

class RAGtuneConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
