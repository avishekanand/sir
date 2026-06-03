from typing import List


class Answer:
    """
    A base class to hold all details about the answer aspect in question answering.

    Attributes:
        _text (str): The text of the answer.
        _idx (int): The ID of the answer.
    """
    def __init__(self, text: str, idx=None):
        self._text = text
        self._idx = idx

    def text(self)->str:
        return self._text

    def id(self):
        return self._idx

    def flatten(self)->List[str]:
        """
        Flattends the answer structure if complex into a simple list of answer texts.
        """
        return [self._text]

    # def set_id(self, id):
    #     self._id = id
    #
    # def set_attention_mask(self, attention_mask):
    #     self.attention_mask = attention_mask