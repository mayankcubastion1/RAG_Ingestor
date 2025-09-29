"""Client-side ingestion pipeline using Azure AI Search push API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence

from azure.core.exceptions import HttpResponseError  # type: ignore
from azure.search.documents import SearchClient  # type: ignore
from ..chunking.layout_chunker import LayoutChunkOptions, chunk_paragraphs
from ..chunking.textsplit_chunker import TextSplitOptions, chunk_text
from ..embeddings.azure_openai import EmbeddingConfig, embed_texts
from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class PushPipelineConfig:
    search_endpoint: str
    index_name: str
    credential: object
    embedding_config: EmbeddingConfig | None = None
    chunk_options: TextSplitOptions | LayoutChunkOptions | None = None


class PushIngestionPipeline:
    """Push documents to Azure AI Search using client-side chunking and embeddings."""

    def __init__(self, config: PushPipelineConfig) -> None:
        self.config = config
        self.search_client = SearchClient(
            endpoint=config.search_endpoint,
            index_name=config.index_name,
            credential=config.credential,
        )

    def ingest_chunks(self, chunks: Sequence[str], metadata: dict[str, object]) -> None:
        """Embed (if configured) and upload chunks."""

        if self.config.embedding_config:
            vectors = embed_texts(chunks, self.config.embedding_config)
        else:
            vectors = [None] * len(chunks)

        documents = []
        for idx, text in enumerate(chunks):
            document = {
                "id": f"{metadata.get('doc_id', 'doc')}-{idx}",
                "chunk_index": idx,
                "chunk_text": text,
                "doc_id": metadata.get("doc_id"),
                "source_type": metadata.get("source_type"),
                "content_vector": vectors[idx],
                "metadata_json": metadata,
            }
            documents.append(document)

        self._upload_documents(documents)

    def _upload_documents(self, documents: Sequence[dict[str, object]]) -> None:
        """Upload documents with retries."""

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                tic = time.perf_counter()
                result = self.search_client.upload_documents(documents)  # type: ignore[no-untyped-call]
                elapsed = time.perf_counter() - tic
                failed = [r for r in result if not r.succeeded]
                if failed:
                    raise HttpResponseError(message=f"Failed documents: {failed}")
                _LOGGER.info("Uploaded %d documents in %.2fs", len(documents), elapsed)
                return
            except HttpResponseError as exc:  # pragma: no cover - network interaction
                wait = min(2 ** attempt, 30)
                _LOGGER.warning("Upload attempt %d failed: %s. Retrying in %ss", attempt, exc, wait)
                time.sleep(wait)
        raise RuntimeError("Failed to upload documents after retries")


def collect_chunks_from_text(text: str, options: TextSplitOptions) -> List[str]:
    return [chunk.text for chunk in chunk_text(text, options)]


def collect_chunks_from_paragraphs(paragraphs: Sequence[str], options: LayoutChunkOptions) -> List[str]:
    layout_chunks = chunk_paragraphs(paragraphs, options)
    return [chunk.text for chunk in layout_chunks]
