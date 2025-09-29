"""Azure OpenAI embedding helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import backoff  # type: ignore
from openai import AzureOpenAI, OpenAI  # type: ignore

from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    deployment: str
    endpoint: str | None = None
    api_key: str | None = None
    model: str | None = None
    api_version: str | None = None
    dimensions: int | None = None
    use_float16: bool = False
    quantize: bool = False


def _create_client(config: EmbeddingConfig) -> AzureOpenAI | OpenAI:
    if config.endpoint:
        return AzureOpenAI(
            api_key=config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=config.endpoint,
            api_version=config.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
    return OpenAI(api_key=config.api_key or os.getenv("OPENAI_API_KEY"))


@backoff.on_exception(backoff.expo, Exception, max_time=120)
def embed_texts(texts: Sequence[str], config: EmbeddingConfig) -> List[list[float]]:
    """Generate embeddings for ``texts`` using Azure/OpenAI."""

    client = _create_client(config)
    tic = time.perf_counter()
    response = client.embeddings.create(
        input=list(texts),
        model=config.model or config.deployment,
        dimensions=config.dimensions,
    )
    elapsed = time.perf_counter() - tic
    _LOGGER.info("Embedded %d chunks in %.2fs", len(texts), elapsed)
    vectors = [item.embedding for item in response.data]

    if config.use_float16:
        vectors = [[float(val) for val in vector] for vector in vectors]

    if config.quantize:
        from .quantize import quantize_vectors

        vectors = quantize_vectors(vectors)

    return vectors
