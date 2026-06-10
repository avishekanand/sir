from typing import Optional

from src.ragtune.data.datastructures.answer import Answer
from src.ragtune.data.datastructures.context import Context
from src.ragtune.data.datastructures.query import Query


class Sample:
    """
    Holds a single evaluation sample: a query with its corresponding
    evidence document and optional answer.

    Attributes:
        idx (str | int): Sample ID (usually the query ID).
        query (Query): The query.
        evidences (Context | None): A relevant document for this sample.
        answer (Answer | None): Optional answer.
    """

    def __init__(
        self,
        idx,
        query: Query,
        evidences: Optional[Context] = None,
        answer: Optional[Answer] = None,
    ):
        self.idx = idx
        self.query = query
        self.evidences = evidences
        self.answer = answer

    def __repr__(self):
        return f"Sample(idx={self.idx}, query={self.query!r})"
