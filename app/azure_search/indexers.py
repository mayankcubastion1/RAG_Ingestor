"""Indexer helpers for Azure AI Search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from azure.search.documents.indexes import SearchIndexerClient  # type: ignore
from azure.search.documents.indexes.models import (  # type: ignore
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataSourceType,
)

from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class DataSourceOptions:
    name: str
    storage_connection_string: str
    container_name: str
    description: str = "Blob data source"
    path_prefix: str | None = None


@dataclass
class IndexerOptions:
    name: str
    data_source_name: str
    skillset_name: str
    target_index_name: str
    schedule_interval: Optional[str] = None
    batch_size: Optional[int] = None
    image_action: Optional[str] = None


def ensure_data_source(client: SearchIndexerClient, options: DataSourceOptions) -> SearchIndexerDataSourceConnection:
    """Create or update a blob data source."""

    container = SearchIndexerDataContainer(name=options.container_name, query=options.path_prefix)
    data_source = SearchIndexerDataSourceConnection(
        name=options.name,
        type=SearchIndexerDataSourceType.AZURE_BLOB,
        connection_string=options.storage_connection_string,
        container=container,
        description=options.description,
    )
    client.create_or_update_data_source_connection(data_source)
    _LOGGER.info("Ensured data source %s", options.name)
    return data_source


def ensure_indexer(client: SearchIndexerClient, options: IndexerOptions) -> SearchIndexer:
    """Create or update an indexer."""

    parameters: Dict[str, Any] = {}
    if options.batch_size:
        parameters["batchSize"] = options.batch_size
    if options.image_action:
        parameters.setdefault("configuration", {})["imageAction"] = options.image_action

    indexer = SearchIndexer(
        name=options.name,
        data_source_name=options.data_source_name,
        skillset_name=options.skillset_name,
        target_index_name=options.target_index_name,
        parameters=parameters or None,
    )

    if options.schedule_interval:
        indexer.schedule = {"interval": options.schedule_interval}

    client.create_or_update_indexer(indexer)
    _LOGGER.info("Ensured indexer %s", options.name)
    return indexer


def run_indexer(client: SearchIndexerClient, indexer_name: str) -> None:
    client.run_indexer(indexer_name)
    _LOGGER.info("Triggered indexer %s", indexer_name)
