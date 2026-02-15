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

class OllamaListwiseReranker(BaseReranker):
    """Listwise reranking using Ollama for local LLM inference."""
    def __init__(self, model_name: str = "deepseek-r1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = f"ollama/{model_name}"
        self.api_base = base_url

    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        if not documents:
            return []
            
        import litellm
        import json
        
        # Construct a listwise ranking prompt
        prompt = (
            f"Question: {query}\n\n"
            "Rank the following documents based on their relevance to the question. "
            "Assign a score between 0.0 (not relevant) and 1.0 (highly relevant) to each document. "
            "Output the results as a JSON list where each item has 'id' and 'relevance_score'.\n\n"
        )
        
        for doc in documents:
            prompt += f"Document ID: {doc.id}\nContent: {doc.content[:500]}\n---\n"
            
        try:
            response = litellm.completion(
                model=self.model_name,
                api_base=self.api_base,
                messages=[{"role": "user", "content": prompt}],
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
