"""
Qwen3-Embedding encoder for pyterrier_dr.

pyterrier-dr (PyPI, 0.7.0) has no built-in Qwen3 support. This follows the
same per-family design pyterrier_dr itself uses for every other model
family (own pooling, own instruction prefix, own defaults) rather than a
generic encoder: last-token pooling that respects left/right padding,
an instruction prefix applied to queries only, optional fp16, and a
max_length default of 8192 (Qwen3 is a long-context model).

Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
"""

from typing import List, Optional

import numpy as np
import pyterrier_dr


def _last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.shape[0]), seq_lens]


def _detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


class Qwen3Encoder(pyterrier_dr.BiEncoder):
    """
    Bi-encoder for Qwen3-Embedding models (e.g. Qwen/Qwen3-Embedding-0.6B).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default "Qwen/Qwen3-Embedding-0.6B".
    max_length : int
        Token truncation length. Qwen3 supports long context; default 8192.
    use_fp16 : bool
        Load model weights in half precision (GPU only).
    task_description : str
        Instruction text injected into queries (documents are left as-is).
    add_instruction_to_query : bool
        Set False to disable the instruction prefix entirely.

    Example
    -------
    >>> encoder = Qwen3Encoder(max_length=2048, use_fp16=True)
    >>> pipeline = encoder >> flex_index
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 32,
        max_length: int = 8192,
        text_field: str = "text",
        verbose: bool = False,
        device: Optional[str] = None,
        use_fp16: bool = False,
        task_description: str = (
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
        add_instruction_to_query: bool = True,
    ):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.task_description = task_description
        self.add_instruction_to_query = add_instruction_to_query

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        model = AutoModel.from_pretrained(model_name)
        if use_fp16:
            model = model.half()
        self.model = model.to(self.device).eval()

    def _encode_batch(self, texts: List[str], batch_size: Optional[int]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        bs = batch_size or self.batch_size
        all_embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                batch = texts[start : start + bs]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                emb = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu().numpy())
        if not all_embeddings:
            return np.empty((0, 0))
        return np.vstack(all_embeddings)

    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        texts = list(texts)
        if self.add_instruction_to_query:
            texts = [_detailed_instruct(self.task_description, t) for t in texts]
        return self._encode_batch(texts, batch_size)

    def encode_docs(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        return self._encode_batch(list(texts), batch_size)

    def __repr__(self):
        return f"Qwen3Encoder({self.model_name!r})"
