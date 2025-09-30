"""Pre-built ingestion pipelines."""

from __future__ import annotations

__all__ = [
    "ShopManualPipeline",
    "ShopManualConfig",
]

from .shop_manual import ShopManualConfig, ShopManualPipeline
