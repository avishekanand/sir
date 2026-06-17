"""
IndexFactory — creates BaseIndexer instances by registry key.

Usage
-----
    from ragtune.indexing.factory import IndexFactory

    # By type string:
    indexer = IndexFactory.create("pyterrier")
    indexer = IndexFactory.create("faiss", model_name_or_path="all-MiniLM-L6-v2")
    indexer = IndexFactory.create("flex", model_name="qwen3", batch_size=32)

    # From a config object (IndexConfig from ragtune.config.models):
    indexer = IndexFactory.from_config(pipeline_config.index)
"""

from ragtune.indexing.base import BaseIndexer
from ragtune.registry import registry


class IndexFactory:

    @staticmethod
    def create(index_type: str, **kwargs) -> BaseIndexer:
        """
        Instantiate an indexer by its registry key.

        Parameters
        ----------
        index_type : str
            Registry key: "pyterrier", "faiss", "flex", "pyserini", ...
        **kwargs
            Constructor arguments forwarded to the indexer class.

        Raises
        ------
        ValueError
            If index_type is not registered.
        """
        cls = registry.get_indexer(index_type)
        if cls is None:
            available = sorted(registry.list_all().get("indexer", {}).keys())
            raise ValueError(
                f"Unknown indexer type: {index_type!r}. Available: {available}"
            )
        return cls(**kwargs)

    @staticmethod
    def from_config(index_config) -> BaseIndexer:
        """
        Build an indexer from an IndexConfig.

        Rules
        -----
        type = "sparse"  →  PyTerrierIndexer (BM25, no encoder)
        type = "dense"   →  backend must be set ("faiss" | "flex");
                            index_config.model.name identifies the model,
                            index_config.params (device, batch_size,
                            max_length, use_fp16, ...) is forwarded as-is.
        """
        if index_config.type == "sparse":
            return IndexFactory.create("pyterrier")

        if index_config.type == "dense":
            if not index_config.backend:
                raise ValueError(
                    "index.backend is required when index.type = 'dense'. "
                    "Choose one of: faiss, flex"
                )
            model_name = index_config.model.name if index_config.model else None
            if not model_name:
                raise ValueError(
                    "index.model.name is required when index.type = 'dense'."
                )

            kwargs = dict(index_config.params)
            # FaissIndexer/NumpyIndexer (DenseIndexer subclasses) use
            # model_name_or_path; FlexIndexer uses model_name.
            if index_config.backend in ("faiss", "numpy"):
                kwargs["model_name_or_path"] = model_name
            else:
                kwargs["model_name"] = model_name
            return IndexFactory.create(index_config.backend, **kwargs)

        raise ValueError(
            f"Unknown index.type: {index_config.type!r}. Choose 'sparse' or 'dense'."
        )
