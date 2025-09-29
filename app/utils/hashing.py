"""Hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO, Iterable


def sha256_bytes(data: bytes) -> str:
    """Return the SHA256 hex digest of ``data``."""

    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def sha256_stream(stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest for a file-like object."""

    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest for a file path."""

    with Path(path).open("rb") as file_obj:
        return sha256_stream(file_obj, chunk_size=chunk_size)


def fingerprints_for_files(paths: Iterable[Path | str]) -> dict[str, str]:
    """Compute digests for multiple paths."""

    results: dict[str, str] = {}
    for path in paths:
        results[str(path)] = sha256_path(path)
    return results
