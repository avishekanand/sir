from typing import List, Dict, Optional, Any
from ragtune.core.pool import PoolItem
from ragtune.core.types import RAGtuneContext
from ragtune.core.interfaces import BaseReranker
from ragtune.registry import registry
from ragtune.utils.config import config

@registry.reranker("noop")
class NoOpReranker(BaseReranker):
    """Identity reranker that returns documents as is."""
    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        return {doc.doc_id: doc.final_score() for doc in documents}

@registry.reranker("cross-encoder")
class CrossEncoderReranker(BaseReranker):
    """Local reranking using SentenceTransformers CrossEncoder models."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        if not documents:
            return {}
            
        pairs = [[context.query, doc.content] for doc in documents]
        scores = self.model.predict(pairs)
        
        return {doc.doc_id: float(score) for doc, score in zip(documents, scores)}

@registry.reranker("llm")
class LLMReranker(BaseReranker):
    """API-based reranking using LiteLLM for broad model support."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        import litellm
        self.model = model_name

    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        if not documents:
            return {}
            
        import litellm
        
        prompts = config.get("prompts.reranking.pointwise")
        system_prompt = prompts.get("system")
        user_prompt_template = prompts.get("user")
        
        scores = {}
        for doc in documents:
            user_prompt = user_prompt_template.format(query=context.query, document=doc.content)
            
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            try:
                import json
                result = json.loads(response.choices[0].message.content)
                score = float(result.get("relevance_score", 0.0))
                scores[doc.doc_id] = score
            except (ValueError, KeyError, AttributeError):
                scores[doc.doc_id] = 0.0
                
        return scores

@registry.reranker("simulated")
class SimulatedReranker(BaseReranker):
    """Placeholder for testing."""
    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        scores = {}
        for doc in documents:
            is_match = context.query.lower() in doc.content.lower()
            reranker_score = 0.95 if is_match else 0.3
            scores[doc.doc_id] = reranker_score
        return scores

@registry.reranker("ollama-listwise")
class OllamaListwiseReranker(BaseReranker):
    """Listwise reranking using Ollama for local LLM inference."""
    def __init__(self, model_name: str = "deepseek-r1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = f"ollama/{model_name}"
        self.api_base = base_url

    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        if not documents:
            return {}
            
        try:
            import litellm
            import json
            
            prompts = config.get_prompt("reranking.listwise_ranking")
            if not prompts:
                # Fallback to hardcoded defaults or raise error
                sys_prompt = "You are a helpful assistant that ranks documents by relevance."
                user_template = "Rank these documents for query '{query}':\n{documents}"
            else:
                sys_prompt = prompts.get("system", "You are a helpful assistant that ranks documents by relevance.")
                user_template = prompts.get("user", "")

            doc_list = ""
            for doc in documents:
                doc_list += f"Document ID: {doc.doc_id}\nContent: {doc.content[:500]}\n---\n"
                
            user_prompt = user_template.format(query=context.query, documents=doc_list)
            
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
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            if not data:
                rankings = []
            elif isinstance(data, list):
                rankings = data
            else:
                rankings = data.get("rankings") or data.get("scores") or []
            
            score_map = {}
            for item in rankings:
                if isinstance(item, dict):
                    doc_id = str(item.get("doc_id") or item.get("id", ""))
                    if doc_id:
                        score_map[doc_id] = float(item.get("relevance_score") or item.get("score", 0.0))
                elif isinstance(item, str):
                    # Fallback for list of IDs
                    score_map[item] = 1.0 # Or some default rank-based score
            
            return {doc.doc_id: score_map.get(doc.doc_id, 0.0) for doc in documents}
            
        except Exception as e:
            print(f"Ollama Rerank Error: {e}")
            return {doc.doc_id: 0.0 for doc in documents}

@registry.reranker("multi-strategy")
class MultiStrategyReranker(BaseReranker):
    """Router for multiple reranking strategies."""
    def __init__(self, strategies: Dict[str, BaseReranker], default_strategy: Optional[str] = None):
        self.strategies = strategies
        self.default_strategy = default_strategy or "identity"

    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        effective_strategy = strategy if strategy and strategy in self.strategies else self.default_strategy
        if effective_strategy not in self.strategies:
            return {doc.doc_id: 0.0 for doc in documents}
        return self.strategies[effective_strategy].rerank(documents, context)
