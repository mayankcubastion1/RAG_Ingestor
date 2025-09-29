"""Azure Blob Storage helpers."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import (  # type: ignore
    BlobClient,
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    ContainerClient,
    generate_blob_sas,
)

from .logging import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class BlobSettings:
    """Configuration required to interact with a blob container."""

    account_url: str
    container_name: str
    credential: object | None = None

    def container_client(self) -> ContainerClient:
        return BlobServiceClient(account_url=self.account_url, credential=self.credential).get_container_client(
            self.container_name
        )


def ensure_container(settings: BlobSettings) -> ContainerClient:
    """Ensure the container exists and return the client."""

    container = settings.container_client()
    try:
        container.create_container()
        _LOGGER.info("Created container %s", settings.container_name)
    except ResourceExistsError:
        _LOGGER.debug("Container %s already exists", settings.container_name)
    return container


def upload_files(
    settings: BlobSettings,
    files: Iterable[Path],
    *,
    prefix: str = "uploads",
    overwrite: bool = True,
) -> list[str]:
    """Upload local files to blob storage and return blob paths."""

    container = ensure_container(settings)
    uploaded_paths: list[str] = []
    for file_path in files:
        blob_name = f"{prefix.rstrip('/')}/{file_path.name}"
        blob: BlobClient = container.get_blob_client(blob_name)
        with file_path.open("rb") as data:
            content_settings = ContentSettings(content_type=_guess_content_type(file_path))
            blob.upload_blob(data, overwrite=overwrite, content_settings=content_settings)
            uploaded_paths.append(blob_name)
            _LOGGER.info("Uploaded %s to %s", file_path, blob_name)
    return uploaded_paths


def generate_sas_url(
    settings: BlobSettings,
    blob_name: str,
    *,
    expiry: dt.timedelta = dt.timedelta(hours=1),
    permissions: Optional[BlobSasPermissions] = None,
) -> str:
    """Generate a SAS URL for the given blob."""

    permissions = permissions or BlobSasPermissions(read=True, list=True)
    sas_token = generate_blob_sas(
        account_name=settings.account_url.split("//")[1].split(".")[0],
        container_name=settings.container_name,
        blob_name=blob_name,
        permission=permissions,
        expiry=dt.datetime.utcnow() + expiry,
        credential=settings.credential,
    )
    return f"{settings.account_url}/{settings.container_name}/{blob_name}?{sas_token}"


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".pdf"}:
        return "application/pdf"
    if suffix in {".png"}:
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix in {".txt"}:
        return "text/plain"
    if suffix in {".json"}:
        return "application/json"
    return "application/octet-stream"
