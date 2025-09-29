"""Sentence and character based chunker."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)

_SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextSplitOptions:
    chunk_size: int = 800
    chunk_overlap: int = 100
    keep_ordinals: bool = True


@dataclass
class TextChunk:
    text: str
    ordinal: int | None = None


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using a naive regex."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_text(text: str, options: TextSplitOptions) -> Iterator[TextChunk]:
    """Chunk text into segments constrained by ``chunk_size``."""

    sentences = split_sentences(text)
    buffer: list[str] = []
    current_length = 0
    ordinal = 0

    for sentence in sentences:
        if current_length + len(sentence) > options.chunk_size and buffer:
            combined = " ".join(buffer)
            yield TextChunk(combined, ordinal if options.keep_ordinals else None)
            ordinal += 1
            overlap = combined[-options.chunk_overlap :] if options.chunk_overlap else ""
            buffer = [overlap, sentence] if overlap else [sentence]
            current_length = sum(len(part) for part in buffer)
        else:
            buffer.append(sentence)
            current_length += len(sentence)

    if buffer:
        combined = " ".join(buffer)
        yield TextChunk(combined, ordinal if options.keep_ordinals else None)

    _LOGGER.debug("Chunked text into %d segments", ordinal + 1)


def chunk_pages(pages: Sequence[str], options: TextSplitOptions) -> Iterable[TextChunk]:
    """Chunk each page individually."""

    for idx, page in enumerate(pages):
        yield TextChunk(page, idx if options.keep_ordinals else None)
