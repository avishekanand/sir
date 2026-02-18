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

class IndexConfig(BaseModel):
    framework: str
    type: str = "sparse"
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
