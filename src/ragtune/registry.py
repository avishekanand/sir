from typing import Dict, Any, Type, Callable, Optional
import functools

class Registry:
    def __init__(self):
        self._rerankers: Dict[str, Any] = {}
        self._retrievers: Dict[str, Any] = {}
        self._reformulators: Dict[str, Any] = {}
        self._assemblers: Dict[str, Any] = {}
        self._schedulers: Dict[str, Any] = {}

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

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        return {
            "reranker": self._rerankers,
            "retriever": self._retrievers,
            "reformulator": self._reformulators,
            "assembler": self._assemblers,
            "scheduler": self._schedulers
        }

# Global registry instance
registry = Registry()
