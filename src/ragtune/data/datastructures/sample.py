from typing import Optional

from answer import Answer
from context import Context
from query import Query


class Sample:
    """
    A base class to hold one datapoint/sample with a question ans its corresponding answer

    Attributes:
        question (Question): question of the sample.
        answer (Answer): answer of the given question.
        context (Answer): Optional context/evidence for the given question.
        _idx (int): The ID of the answer.
    """
    def __init__(self, idx, query: Query, answer: Answer, evidences: Optional[Context] = None):
        self.query = query
        self.evidences = evidences
        self.answer = answer
        self.idx = idx