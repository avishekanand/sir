from typing import List, Any
from ragtune.core.interfaces import BaseRetriever
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.registry import registry

@registry.retriever("langchain")
class LangChainRetriever(BaseRetriever):
    """
    Adapter for LangChain retrievers to work within the RAGtune ecosystem.
    """
    def __init__(self, lc_retriever: Any):
        """
        Args:
            lc_retriever: Any object that implements invoke() or get_relevant_documents() 
                         returning a list of LangChain Document objects.
        """
        self.lc_retriever = lc_retriever

    def retrieve(self, context: RAGtuneContext, top_k: int = 10) -> List[ScoredDocument]:
        """
        Retrieve documents from the underlying LangChain retriever and 
        convert them to RAGtune ScoredDocuments.
        """
        # LangChain's newer invoke() method or older get_relevant_documents()
        if hasattr(self.lc_retriever, "invoke"):
            lc_docs = self.lc_retriever.invoke(context.query)
        else:
            lc_docs = self.lc_retriever.get_relevant_documents(context.query)
            
        results = []
        for i, doc in enumerate(lc_docs[:top_k]):
            # Use metadata if ID is missing
            doc_id = doc.metadata.get("id", str(i))
            
            # Estimate token count if not provided
            # (In a real setup, we'd use a tokenizer)
            token_count = doc.metadata.get("tokens", len(doc.page_content.split()) * 1.3)
            
            results.append(ScoredDocument(
                id=doc_id,
                content=doc.page_content,
                metadata=doc.metadata,
                score=1.0 / (i + 1), # Default reciprocal score if retriever doesn't provide one
                token_count=int(token_count)
            ))
            
        return results

class RAGtuneLangChainAdapter:
    """
    Exposes a RAGtuneController as a LangChain-compatible retriever.
    """
    def __init__(self, controller: Any):
        self.controller = controller

    def invoke(self, query: str, **kwargs) -> List[Any]:
        from langchain_core.documents import Document
        output = self.controller.run(query)
        
        return [
            Document(
                page_content=doc.content,
                metadata={
                    "id": doc.id,
                    "final_score": doc.score,
                    **doc.metadata
                }
            ) for doc in output.documents
        ]

    async def ainvoke(self, query: str, **kwargs) -> List[Any]:
        from langchain_core.documents import Document
        output = await self.controller.arun(query)
        
        return [
            Document(
                page_content=doc.content,
                metadata={
                    "id": doc.id,
                    "final_score": doc.score,
                    **doc.metadata
                }
            ) for doc in output.documents
        ]
