"""Vendored inference port of fnlp/MOSS-Speech-Codec (snapshot eeec733e)."""

from .modeling import MossSpeechCodec
from .utils import WhisperVQConfig

__all__ = ["MossSpeechCodec", "WhisperVQConfig"]
