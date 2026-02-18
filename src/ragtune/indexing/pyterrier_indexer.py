import os
import pyterrier as pt
from typing import Dict, Any
from pathlib import Path
from ragtune.core.interfaces import BaseIndexer
from ragtune.registry import registry

@registry.indexer("pyterrier")
class PyTerrierIndexer(BaseIndexer):
    """Indexer implementation using PyTerrier."""
    
    def build(self, collection_path: str, format: str, fields: Dict[str, str], **params) -> bool:
        if not pt.started():
            pt.init()
            
        index_path = params.get("index_path")
        if not index_path:
            raise ValueError("index_path is required for PyTerrierIndexer")
        
        index_path = os.path.abspath(index_path)
            
        # Create index dir if not exists
        os.makedirs(index_path, exist_ok=True)
        
        # JSON / JSONL indexing
        import json
        if format in ["json", "jsonl"]:
            def iter_docs():
                if format == "jsonl":
                    with open(collection_path, "r") as f:
                        for line in f:
                            yield json.loads(line)
                else:
                    with open(collection_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for doc in data:
                                yield doc
                        elif isinstance(data, dict):
                            # Handle cases where the whole json is a dict (e.g. BRIGHT format if it's not a list)
                            # But usually it's a list. Let's assume list for now or adapt if needed.
                            yield data

            def mapped_iter():
                for doc in iter_docs():
                    text_val = doc.get(fields.get("text_field", "text"))
                    if text_val is None:
                        text_val = ""
                    
                    yield {
                        "docno": str(doc.get(fields.get("id_field", "doc_id"), "")),
                        "text": str(text_val),
                        **{k: doc.get(v) for k, v in fields.get("metadata_fields", {}).items() if k not in ["docno", "text"]}
                    }
            
            docs_to_index = list(mapped_iter())
            
            indexer = pt.IterDictIndexer(index_path, overwrite=True)
            indexer.index(docs_to_index)

            # Verify index properties
            try:
                props_path = os.path.join(index_path, "data.properties")
                if os.path.exists(props_path):
                    with open(props_path, "r") as f:
                        props = f.read()
                        if "num.Pointers=0" in props:
                            print(f"WARNING: Index built at {index_path} has 0 pointers. Inverted index may be missing.")
            except Exception:
                pass

            return True
        else:
            raise NotImplementedError(f"Format {format} not yet supported in PyTerrierIndexer adapter")

    def exists(self, index_path: str) -> bool:
        # PyTerrier index is usually a folder or a data.properties file
        path = Path(index_path)
        return path.exists() and (path.is_dir() or (path / "data.properties").exists())
