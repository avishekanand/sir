"""
Unit tests for ragtune.indexing.encoders.resolve_encoder routing logic.
"""

import pyterrier_dr

from ragtune.indexing.encoders import GenericHFEncoder, Qwen3Encoder, resolve_encoder


class TestExactShorthandMatches:

    def test_bge_m3(self):
        cls, defaults, resolved = resolve_encoder("bge-m3")
        assert cls is pyterrier_dr.BGEM3
        assert resolved is None

    def test_tasb(self):
        cls, defaults, resolved = resolve_encoder("tasb")
        assert cls is pyterrier_dr.TasB
        assert resolved is None

    def test_ance(self):
        cls, defaults, resolved = resolve_encoder("ance")
        assert cls is pyterrier_dr.Ance
        assert resolved is None

    def test_tct(self):
        cls, defaults, resolved = resolve_encoder("tct")
        assert cls is pyterrier_dr.TctColBert
        assert resolved is None


class TestQwen3FamilyMatch:

    def test_bare_shorthand_uses_class_default(self):
        cls, defaults, resolved = resolve_encoder("qwen3")
        assert cls is Qwen3Encoder
        assert resolved is None  # let Qwen3Encoder's own default checkpoint apply

    def test_explicit_06b_checkpoint_passed_through(self):
        cls, defaults, resolved = resolve_encoder("Qwen/Qwen3-Embedding-0.6B")
        assert cls is Qwen3Encoder
        assert resolved == "Qwen/Qwen3-Embedding-0.6B"

    def test_explicit_4b_checkpoint_passed_through(self):
        cls, defaults, resolved = resolve_encoder("Qwen/Qwen3-Embedding-4B")
        assert cls is Qwen3Encoder
        assert resolved == "Qwen/Qwen3-Embedding-4B"

    def test_explicit_8b_checkpoint_passed_through(self):
        cls, defaults, resolved = resolve_encoder("Qwen/Qwen3-Embedding-8B")
        assert cls is Qwen3Encoder
        assert resolved == "Qwen/Qwen3-Embedding-8B"

    def test_case_insensitive_match(self):
        cls, defaults, resolved = resolve_encoder("QWEN/QWEN3-EMBEDDING-4B")
        assert cls is Qwen3Encoder
        assert resolved == "QWEN/QWEN3-EMBEDDING-4B"


class TestFallbackToGenericHF:

    def test_unrelated_hf_model_id_falls_back(self):
        cls, defaults, resolved = resolve_encoder("intfloat/e5-base-v2")
        assert cls is GenericHFEncoder
        assert resolved == "intfloat/e5-base-v2"

    def test_arbitrary_unknown_shorthand_falls_back(self):
        cls, defaults, resolved = resolve_encoder("some-random-model")
        assert cls is GenericHFEncoder
        assert resolved == "some-random-model"
