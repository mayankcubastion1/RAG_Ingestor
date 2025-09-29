"""Helpers for Azure AI Search index aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

from azure.core.exceptions import ResourceNotFoundError  # type: ignore
from azure.search.documents.indexes import SearchIndexClient  # type: ignore

try:  # pragma: no cover - depends on azure-search-documents version
    from azure.search.documents.indexes.models import SearchAlias  # type: ignore
except ImportError:  # pragma: no cover - depends on azure-search-documents version
    SearchAlias = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from azure.search.documents.indexes.models import SearchAlias as _SearchAlias
else:  # pragma: no cover - runtime fallback
    _SearchAlias = Any

from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class AliasOptions:
    alias_name: str
    index_name: str


class AliasFeatureUnavailableError(RuntimeError):
    """Raised when the installed SDK does not expose index alias APIs."""


def _alias_support_message() -> str:
    return (
        "Azure AI Search alias operations require the preview SDK "
        "(azure-search-documents>=11.6.0b12). Install the preview package to "
        "enable alias management."
    )


def alias_feature_status() -> Tuple[bool, str | None]:
    """Return whether the SDK exposes alias APIs and a message for the UI."""

    has_models = SearchAlias is not None
    has_methods = hasattr(SearchIndexClient, "create_or_update_alias")
    if has_models and has_methods:
        return True, None
    return False, _alias_support_message()


def _require_alias_support() -> None:
    supported, _ = alias_feature_status()
    if not supported:
        raise AliasFeatureUnavailableError(_alias_support_message())


def swap_alias(client: SearchIndexClient, options: AliasOptions) -> _SearchAlias:
    """Create or update an alias pointing to ``index_name``."""

    _require_alias_support()
    alias = SearchAlias(name=options.alias_name, indexes=[options.index_name])  # type: ignore[call-arg]
    client.create_or_update_alias(alias)
    _LOGGER.info("Alias %s now points to %s", options.alias_name, options.index_name)
    return alias


def delete_alias(client: SearchIndexClient, alias_name: str) -> None:
    _require_alias_support()
    try:
        client.delete_alias(alias_name)
        _LOGGER.info("Deleted alias %s", alias_name)
    except ResourceNotFoundError:
        _LOGGER.debug("Alias %s not found", alias_name)
