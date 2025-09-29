"""Layout-aware chunker leveraging optional document layout extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class LayoutChunk:
    """Represents a layout-aware chunk."""

    text: str
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    ordinal: int | None = None


@dataclass
class LayoutChunkOptions:
    chunk_size: int = 800
    chunk_overlap: int = 100
    keep_ordinals: bool = True


def chunk_paragraphs(paragraphs: Sequence[str], options: LayoutChunkOptions) -> List[LayoutChunk]:
    """Chunk paragraphs while respecting the configured size and overlap."""

    chunks: List[LayoutChunk] = []
    buffer: list[str] = []
    current_length = 0
    ordinal = 0

    for para in paragraphs:
        if current_length + len(para) > options.chunk_size and buffer:
            text = "\n\n".join(buffer)
            chunks.append(LayoutChunk(text=text, ordinal=ordinal if options.keep_ordinals else None))
            ordinal += 1
            overlap_text = text[-options.chunk_overlap :] if options.chunk_overlap else ""
            buffer = [overlap_text, para] if overlap_text else [para]
            current_length = sum(len(p) for p in buffer)
        else:
            buffer.append(para)
            current_length += len(para)

    if buffer:
        text = "\n\n".join(buffer)
        chunks.append(LayoutChunk(text=text, ordinal=ordinal if options.keep_ordinals else None))

    _LOGGER.debug("Generated %d layout chunks", len(chunks))
    return chunks


def fallback_pdf_paragraphs(pdf_path: str) -> Iterable[str]:
    """Attempt to extract paragraphs from a PDF path using pdfplumber if available."""

    try:
        import pdfplumber  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        _LOGGER.warning("pdfplumber is not installed; cannot perform layout extraction")
        return []

    paragraphs: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            paragraphs.extend(p.extract_text() or "" for p in page.extract_words(use_text_flow=True))
    return [p.strip() for p in paragraphs if p and p.strip()]


def chunk_pdf(pdf_path: str, options: LayoutChunkOptions) -> Iterator[LayoutChunk]:
    """Chunk a PDF document using fallback layout extraction."""

    paragraphs = list(fallback_pdf_paragraphs(pdf_path))
    yield from chunk_paragraphs(paragraphs, options)
