"""Authentication helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from azure.core.credentials import AzureKeyCredential  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore


@lru_cache(maxsize=1)
def get_default_credential() -> DefaultAzureCredential:
    """Return a shared ``DefaultAzureCredential`` instance."""

    return DefaultAzureCredential(exclude_interactive_browser_credential=False)


def resolve_search_credential(api_key: str | None) -> Any:
    """Resolve credentials for Azure AI Search."""

    if api_key:
        return AzureKeyCredential(api_key)
    return get_default_credential()
