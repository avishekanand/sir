"""
Encoder registry used by FlexIndexer.

Resolution order (see resolve_encoder)
---------------------------------------
1. Exact shorthand match in BUILTIN_ENCODERS ("bge-m3", "tasb", "ance", "tct")
   → pyterrier_dr's own built-ins, using their internal model defaults.
2. Family substring match in FAMILY_ENCODERS ("qwen3" anywhere in the name,
   case-insensitive) → our own encoder class for that family. This lets
   model_name be either the bare shorthand ("qwen3", → family default
   checkpoint) or a full HF id picking a specific size, e.g.
   "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B",
   "Qwen/Qwen3-Embedding-8B" — all route to Qwen3Encoder, with the real id
   passed straight through to transformers.
3. Fallback: GenericHFEncoder, given the raw model_name as-is.
"""

from typing import Dict, Optional, Tuple, Type

import pyterrier_dr

from ragtune.indexing.encoders.generic_hf import GenericHFEncoder
from ragtune.indexing.encoders.qwen import Qwen3Encoder

# Exact-match shorthands → pyterrier_dr built-ins (already correct for these families).
BUILTIN_ENCODERS: Dict[str, Tuple[Type, dict]] = {
    "bge-m3": (pyterrier_dr.BGEM3, {}),
    "tasb":   (pyterrier_dr.TasB, {}),
    "ance":   (pyterrier_dr.Ance, {}),
    "tct":    (pyterrier_dr.TctColBert, {}),
}

# Substring-match families → our own encoder classes. Covers every checkpoint
# size in the family without enumerating each one (e.g. all Qwen3-Embedding sizes).
FAMILY_ENCODERS: Dict[str, Type] = {
    "qwen3": Qwen3Encoder,
}


def resolve_encoder(model_name: str) -> Tuple[Type, dict, Optional[str]]:
    """
    Resolve a model_name string to (encoder_class, default_kwargs, model_name_to_pass).

    model_name_to_pass is None when the bare shorthand was used (let the
    encoder class's own default checkpoint apply); otherwise it's the
    original model_name, to be forwarded to the encoder's constructor.
    """
    if model_name in BUILTIN_ENCODERS:
        cls, defaults = BUILTIN_ENCODERS[model_name]
        return cls, defaults, None

    lname = model_name.lower()
    for family_key, cls in FAMILY_ENCODERS.items():
        if family_key in lname:
            resolved = None if lname == family_key else model_name
            return cls, {}, resolved

    return GenericHFEncoder, {}, model_name


__all__ = [
    "BUILTIN_ENCODERS",
    "FAMILY_ENCODERS",
    "resolve_encoder",
    "Qwen3Encoder",
    "GenericHFEncoder",
]
