import os
import pandas as pd
from typing import List, Any, Optional
from ragtune.core.interfaces import BaseRetriever
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.core.controller import RAGtuneController
from ragtune.registry import registry

try:
    import pyterrier as pt
except ImportError:
    # PyTerrier might not be installed in all environments
    pt = None

@registry.retriever("pyterrier")
class PyTerrierRetriever(BaseRetriever):
    """
    Adapter for PyTerrier transformers (indices) to work as RAGtune retrievers.
    """
    def __init__(self, pt_transformer: Any = None, index_path: Optional[str] = None):
        """
        Args:
            pt_transformer: A PyTerrier transformer (e.g., pt.BatchRetrieve)
            index_path: Path to a PyTerrier index (used if pt_transformer is None)
        """
        if pt_transformer is not None:
            self.pt_transformer = pt_transformer
        elif index_path is not None:
            if pt is None:
                raise ImportError("PyTerrier not installed but index_path provided.")
            
            abs_path = os.path.abspath(index_path)
            # If it's a directory, look for data.properties
            if os.path.isdir(abs_path):
                props_path = os.path.join(abs_path, "data.properties")
                if os.path.exists(props_path):
                    abs_path = props_path

            # Use the more modern and reliable Retriever loader
            try:
                self.pt_transformer = pt.terrier.Retriever(abs_path, wmodel="BM25")
            except Exception:
                # Fallback to BatchRetrieve
                self.pt_transformer = pt.BatchRetrieve(abs_path, wmodel="BM25")
        else:
            raise ValueError("Either pt_transformer or index_path must be provided.")

    def retrieve(self, context: RAGtuneContext, top_k: int = 10) -> List[ScoredDocument]:
        """
        Retrieve documents using PyTerrier and convert to RAGtune format.
        """
        # Create a single-row DataFrame for the query
        queries_df = pd.DataFrame([{"qid": "q1", "query": context.query}])
        
        # Run retrieval
        res = self.pt_transformer.transform(queries_df)
        # print(f"DEBUG: PyTerrier retrieved {len(res)} rows for top_k={top_k}")
        
        # Sort by score just in case and take top_k
        res = res.sort_values("score", ascending=False).head(top_k)
        
        results = []
        for i, row in res.iterrows():
            # PyTerrier usually has 'docno', 'text' (if added), and 'score'
            doc_id = str(row.get("docno", row.get("docid", i)))
            content = row.get("text", row.get("content", ""))
            score = float(row.get("score", 0.0))
            
            # Extract metadata (any non-standard columns)
            # Include 'rank' even if it's standard in PyTerrier to preserve initial retrieval order
            metadata = {k: v for k, v in row.to_dict().items() if k not in ["qid", "query", "docno", "docid", "text", "content", "score"]}
            metadata["initial_rank"] = int(row.get("rank", i))
            
            results.append(ScoredDocument(
                id=doc_id,
                content=content,
                metadata=metadata,
                score=score,
                original_score=score,
                token_count=int(len(content.split()) * 1.3)
            ))
            
        return results

class RAGtuneTransformer:
    """
    A PyTerrier-compatible transformer that wraps a RAGtuneController.
    Allows RAGtune to be used in a pipeline like: bm25 >> ragtune_transformer
    """
    def __init__(self, controller: RAGtuneController, verbose: bool = False):
        self.controller = controller
        self.verbose = verbose

    def transform(self, queries_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each query in the DataFrame, run the RAGtune iterative loop.
        """
        all_results = []
        
        for _, row in queries_df.iterrows():
            qid = row["qid"]
            query = row["query"]
            
            # Run RAGtune
            output = self.controller.run(query)
            
            # Convert ScoredDocuments to PyTerrier format
            for i, doc in enumerate(output.documents):
                res_row = {
                    "qid": qid,
                    "docno": doc.id,
                    "rank": i,
                    "score": doc.score,
                    "text": doc.content
                }
                # Add metadata back
                res_row.update(doc.metadata)
                all_results.append(res_row)
                
        return pd.DataFrame(all_results)
    
    def __rshift__(self, other):
        # Support PyTerrier's >> operator if pt is available
        if pt is not None:
            return pt.Transformer.from_df(self.transform) >> other
        return super().__rshift__(other)
