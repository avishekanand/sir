from typing import List, Optional
from ragtune.core.interfaces import BaseReranker
from ragtune.core.types import ScoredDocument

class NoOpReranker(BaseReranker):
    """Identity reranker that returns documents as is."""
    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        return documents

class CrossEncoderReranker(BaseReranker):
    """Local reranking using SentenceTransformers CrossEncoder models."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        if not documents:
            return []
        
        pairs = [[query, doc.content] for doc in documents]
        scores = self.model.predict(pairs)
        
        results = []
        for doc, score in zip(documents, scores):
            results.append(doc.model_copy(update={
                "score": float(score),
                "reranker_score": float(score)
            }))
        return results

class LLMReranker(BaseReranker):
    """API-based reranking using LiteLLM for broad model support."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        import litellm
        self.model_name = model_name

    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        if not documents:
            return []
            
        import litellm
        import json
        
        # Construct a scoring prompt
        prompt = f"Query: {query}\n\nRate the following documents from 0.0 to 1.0 based on relevance. Output a JSON list of scores only.\n"
        for i, doc in enumerate(documents):
            prompt += f"[{i}] {doc.content}\n"
            
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
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

class SimulatedReranker(BaseReranker):
    """Placeholder for testing."""
    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        results = []
        for doc in documents:
            is_match = query.lower() in doc.content.lower()
            reranker_score = 0.95 if is_match else 0.3
            results.append(doc.model_copy(update={
                "score": reranker_score,
                "reranker_score": reranker_score
            }))
        return results
