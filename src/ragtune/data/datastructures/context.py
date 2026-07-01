from typing import Optional


class Context:
    """
    Data class to hold a retrieved document / passage (evidence).

    Attributes:
        _text (str): Text of the passage.
        _idx (str | int): Document ID.
        _title (str | None): Optional title.
    """

    def __init__(self, text: str, idx=None, title: Optional[str] = None):
        self._text = text
        self._idx = idx
        self._title = title

    def text(self) -> str:
        return self._text

    def id(self):
        return self._idx

    def title(self) -> Optional[str]:
        return self._title

    def __repr__(self):
        return f"Context(id={self._idx}, title={self._title!r})"
