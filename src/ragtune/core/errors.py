class RAGtuneError(Exception):
    """Base exception for RAGtune."""
    pass

class ConfigError(RAGtuneError):
    """Configuration related errors."""
    pass

class BudgetExhaustedError(RAGtuneError):
    """Raised when budget is exhausted and no fallback is available."""
    pass
