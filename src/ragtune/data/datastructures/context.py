class Context:
    """ Data class to hold evidence/context for Question Answering

    Args:
        text : text of evidence passage
        title : title of evidence passage
        idx : index of evidence passage
    """
    def __init__(self, text: str, idx=None, title: str = None):
        self._text = text
        self._idx = idx
        self._title = title

    def text(self):
        return self._text

    def id(self):
        return self._idx

    def title(self):
        return self._title