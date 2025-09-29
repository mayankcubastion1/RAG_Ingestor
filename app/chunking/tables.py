"""Table extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd  # type: ignore

from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class TableExtractionResult:
    markdown: str
    html: str
    page_number: int | None = None


def extract_tables_from_csv(path: str) -> List[TableExtractionResult]:
    """Load CSV tables for ingestion."""

    frame = pd.read_csv(path)
    markdown = frame.to_markdown(index=False)
    html = frame.to_html(index=False)
    _LOGGER.debug("Extracted table from %s", path)
    return [TableExtractionResult(markdown=markdown, html=html)]


def extract_tables_from_dfs(dfs: Sequence[pd.DataFrame]) -> Iterable[TableExtractionResult]:
    for df in dfs:
        yield TableExtractionResult(markdown=df.to_markdown(index=False), html=df.to_html(index=False))
