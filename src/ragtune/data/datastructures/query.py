from typing import Optional


class Query:
    """
    Holds all details about a query for retrieval.

    Attributes:
        _text (str): The text of the query.
        _idx (str | int): The ID of the query.
        reasoning (str | None): Optional chain-of-thought reasoning for the query
            (used by BRIGHT reasoning-augmented subsets).
    """

    def __init__(self, text: str, idx=None, reasoning: Optional[str] = None):
        self._text = text
        self._idx = idx
        self.reasoning = reasoning
        self.attention_mask = None

    def text(self) -> str:
        return self._text

    def id(self):
        return self._idx

    def set_id(self, idx):
        self._idx = idx

    def set_attention_mask(self, attention_mask):
        self.attention_mask = attention_mask

    def __repr__(self):
        return f"Query(id={self._idx}, text={self._text[:60]!r})"

    def __eq__(self, other):
        if isinstance(other, Query):
            return self._idx == other._idx
        return False

    def __hash__(self):
        return hash(self._idx)
