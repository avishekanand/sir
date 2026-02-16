from typing import List, Optional, Dict
from ragtune.core.interfaces import BaseReranker
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.registry import registry
from ragtune.utils.config import config

@registry.reranker("noop")
class NoOpReranker(BaseReranker):
    """Identity reranker that returns documents as is."""
    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        return documents

@registry.reranker("cross-encoder")
class CrossEncoderReranker(BaseReranker):
    """Local reranking using SentenceTransformers CrossEncoder models."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        if not documents:
            return []
        
        pairs = [[context.query, doc.content] for doc in documents]
        scores = self.model.predict(pairs)
        
        results = []
        for doc, score in zip(documents, scores):
            results.append(doc.model_copy(update={
                "score": float(score),
                "reranker_score": float(score)
            }))
        return results

@registry.reranker("llm")
class LLMReranker(BaseReranker):
    """API-based reranking using LiteLLM for broad model support."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        import litellm
        self.model_name = model_name

    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        if not documents:
            return []
            
        import litellm
        import json
        
        # Load prompt from config
        sys_prompt = config.get_prompt("reranking.pointwise_scoring.system", "You are a helpful assistant.")
        user_template = config.get_prompt("reranking.pointwise_scoring.user", "Query: {query}\n\nDocuments: {documents}")
        
        doc_list = ""
        for i, doc in enumerate(documents):
            doc_list += f"[{i}] {doc.content}\n"
            
        user_prompt = user_template.format(query=context.query, documents=doc_list)
        
        response = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parsing logic (simplified for v0.2)
        content = response.choices[0].message.content
        try:
            raw_scores = json.loads(content).get("scores", [])
        except:
            raw_scores = [0.0] * len(documents)
            
        results = []
        for doc, score in zip(documents, raw_scores):
            results.append(doc.model_copy(update={
                "score": float(score),
                "reranker_score": float(score)
            }))
        return results

@registry.reranker("simulated")
class SimulatedReranker(BaseReranker):
    """Placeholder for testing."""
    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        results = []
        for doc in documents:
            is_match = context.query.lower() in doc.content.lower()
            reranker_score = 0.95 if is_match else 0.3
            results.append(doc.model_copy(update={
                "score": reranker_score,
                "reranker_score": reranker_score
            }))
        return results

@registry.reranker("ollama-listwise")
class OllamaListwiseReranker(BaseReranker):
    """Listwise reranking using Ollama for local LLM inference."""
    def __init__(self, model_name: str = "deepseek-r1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = f"ollama/{model_name}"
        self.api_base = base_url

    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        if not documents:
            return []
            
        import litellm
        import json
        
        # Load prompt from config
        sys_prompt = config.get_prompt("reranking.listwise_ranking.system", "You are a helpful assistant.")
        user_template = config.get_prompt("reranking.listwise_ranking.user", "Query: {query}\n\nDocuments: {documents}")
        
        doc_list = ""
        for doc in documents:
            doc_list += f"Document ID: {doc.id}\nContent: {doc.content[:500]}\n---\n"
            
        user_prompt = user_template.format(query=context.query, documents=doc_list)
        
        try:
            response = litellm.completion(
                model=self.model_name,
                api_base=self.api_base,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            # Handle potential markdown wrappers if the model returns them
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            # Find the list in the JSON (might be under a key like 'rankings' or 'scores')
            rankings = data if isinstance(data, list) else (data.get("rankings") or data.get("scores") or [])
            
            # Map scores back to documents
            score_map = {str(item.get("id")): float(item.get("relevance_score", 0.0)) for item in rankings}
            
            results = []
            for doc in documents:
                score = score_map.get(str(doc.id), 0.0)
                results.append(doc.model_copy(update={
                    "score": score,
                    "reranker_score": score
                }))
            return results
            
        except Exception as e:
            # Fallback to 0.0 scores if LLM fails
            print(f"Ollama Rerank Error: {e}")
            return [doc.model_copy(update={"score": 0.0, "reranker_score": 0.0}) for doc in documents]

@registry.reranker("multi-strategy")
class MultiStrategyReranker(BaseReranker):
    """Router for multiple reranking strategies."""
    def __init__(self, strategies: Dict[str, BaseReranker], default_strategy: Optional[str] = None):
        self.strategies = strategies
        self.default_strategy = default_strategy or "identity"

    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        target = strategy or self.default_strategy
        reranker = self.strategies.get(target)
        if not reranker:
            # Fallback to identity or first available?
            # Let's use identity if registered, else skip
            return documents
        
        return reranker.rerank(documents, context, strategy=target)

    async def arerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        target = strategy or self.default_strategy
        reranker = self.strategies.get(target)
        if not reranker:
            return documents
        
        return await reranker.arerank(documents, context, strategy=target)
