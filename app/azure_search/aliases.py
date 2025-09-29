"""Helpers for Azure AI Search index aliases."""

from __future__ import annotations

from dataclasses import dataclass

from azure.core.exceptions import ResourceNotFoundError  # type: ignore
from azure.search.documents.indexes import SearchIndexClient  # type: ignore
from azure.search.documents.indexes.models import SearchAlias  # type: ignore

from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class AliasOptions:
    alias_name: str
    index_name: str


def swap_alias(client: SearchIndexClient, options: AliasOptions) -> SearchAlias:
    """Create or update an alias pointing to ``index_name``."""

    alias = SearchAlias(name=options.alias_name, indexes=[options.index_name])
    client.create_or_update_alias(alias)
    _LOGGER.info("Alias %s now points to %s", options.alias_name, options.index_name)
    return alias


def delete_alias(client: SearchIndexClient, alias_name: str) -> None:
    try:
        client.delete_alias(alias_name)
        _LOGGER.info("Deleted alias %s", alias_name)
    except ResourceNotFoundError:
        _LOGGER.debug("Alias %s not found", alias_name)
