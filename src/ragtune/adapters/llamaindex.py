from typing import List, Any, Optional
from ragtune.core.interfaces import BaseRetriever
from ragtune.core.types import ScoredDocument

class LlamaIndexRetriever(BaseRetriever):
    """
    Adapter for LlamaIndex retrievers to work within the RAGtune ecosystem.
    """
    def __init__(self, li_retriever: Any):
        """
        Args:
            li_retriever: Any LlamaIndex retriever that implements retrieve() 
                         returning a list of NodeWithScore objects.
        """
        self.li_retriever = li_retriever

    def retrieve(self, query: str, top_k: int = 10) -> List[ScoredDocument]:
        """
        Retrieve documents from the underlying LlamaIndex retriever and 
        convert them to RAGtune ScoredDocuments.
        """
        # LlamaIndex retrieval returns List[NodeWithScore]
        li_nodes = self.li_retriever.retrieve(query)
        
        results = []
        for i, node_with_score in enumerate(li_nodes[:top_k]):
            node = node_with_score.node
            score = node_with_score.score if node_with_score.score is not None else 1.0 / (i + 1)
            
            # Extract metadata and content
            doc_id = node.node_id
            content = node.get_content()
            metadata = node.metadata or {}
            
            # Estimate token count
            token_count = metadata.get("tokens", len(content.split()) * 1.3)
            
            results.append(ScoredDocument(
                id=doc_id,
                content=content,
                metadata=metadata,
                score=float(score),
                original_score=float(score),
                token_count=int(token_count)
            ))
            
        return results

    async def aretrieve(self, query: str, top_k: int = 10) -> List[ScoredDocument]:
        """
        Async version of LlamaIndex retrieval.
        Using LlamaIndex's aretrieve if available.
        """
        if hasattr(self.li_retriever, "aretrieve"):
            li_nodes = await self.li_retriever.aretrieve(query)
        else:
            li_nodes = self.li_retriever.retrieve(query)
            
        results = []
        for i, node_with_score in enumerate(li_nodes[:top_k]):
            node = node_with_score.node
            score = node_with_score.score if node_with_score.score is not None else 1.0 / (i + 1)
            
            # Extract metadata and content
            doc_id = node.node_id
            content = node.get_content()
            metadata = node.metadata or {}
            
            # Estimate token count
            token_count = metadata.get("tokens", len(content.split()) * 1.3)
            
            results.append(ScoredDocument(
                id=doc_id,
                content=content,
                metadata=metadata,
                score=float(score),
                original_score=float(score),
                token_count=int(token_count)
            ))
            
        return results
