from typing import List


class Answer:
    """
    Holds answer details for question-answering tasks.

    Attributes:
        _text (str): The text of the answer.
        _idx (str | int | None): Optional ID.
    """

    def __init__(self, text: str, idx=None):
        self._text = text
        self._idx = idx

    def text(self) -> str:
        return self._text

    def id(self):
        return self._idx

    def flatten(self) -> List[str]:
        """Flatten complex answer structures to a list of strings."""
        return [self._text]

    def __repr__(self):
        return f"Answer(id={self._idx}, text={self._text[:60]!r})"
