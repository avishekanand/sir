from typing import Dict, Any, Optional

class Registry:
    def __init__(self):
        self._rerankers: Dict[str, Any] = {}
        self._retrievers: Dict[str, Any] = {}
        self._reformulators: Dict[str, Any] = {}
        self._assemblers: Dict[str, Any] = {}
        self._schedulers: Dict[str, Any] = {}
        self._estimators: Dict[str, Any] = {}
        self._indexers: Dict[str, Any] = {}
        self._feedback: Dict[str, Any] = {}

    def reranker(self, name: str):
        def wrapper(cls_or_func):
            self._rerankers[name] = cls_or_func
            return cls_or_func
        return wrapper

    def retriever(self, name: str):
        def wrapper(cls_or_func):
            self._retrievers[name] = cls_or_func
            return cls_or_func
        return wrapper

    def reformulator(self, name: str):
        def wrapper(cls_or_func):
            self._reformulators[name] = cls_or_func
            return cls_or_func
        return wrapper

    def assembler(self, name: str):
        def wrapper(cls_or_func):
            self._assemblers[name] = cls_or_func
            return cls_or_func
        return wrapper

    def scheduler(self, name: str):
        def wrapper(cls_or_func):
            self._schedulers[name] = cls_or_func
            return cls_or_func
        return wrapper

    def estimator(self, name: str):
        def wrapper(cls_or_func):
            self._estimators[name] = cls_or_func
            return cls_or_func
        return wrapper

    def indexer(self, name: str):
        def wrapper(cls_or_func):
            self._indexers[name] = cls_or_func
            return cls_or_func
        return wrapper

    def feedback(self, name: str):
        def wrapper(cls_or_func):
            self._feedback[name] = cls_or_func
            return cls_or_func
        return wrapper

    def get_reranker(self, name: str):
        return self._rerankers.get(name)

    def get_retriever(self, name: str):
        return self._retrievers.get(name)

    def get_reformulator(self, name: str):
        return self._reformulators.get(name)

    def get_assembler(self, name: str):
        return self._assemblers.get(name)

    def get_scheduler(self, name: str):
        return self._schedulers.get(name)

    def get_estimator(self, name: str):
        return self._estimators.get(name)

    def get_indexer(self, name: str):
        return self._indexers.get(name)

    def get_feedback(self, name: str):
        return self._feedback.get(name)

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        return {
            "reranker": self._rerankers,
            "retriever": self._retrievers,
            "reformulator": self._reformulators,
            "assembler": self._assemblers,
            "scheduler": self._schedulers,
            "estimator": self._estimators,
            "indexer": self._indexers,
            "feedback": self._feedback
        }

# Global registry instance
registry = Registry()
