"""Simple vector quantization utilities."""

from __future__ import annotations

from typing import Iterable, List


def quantize_vectors(vectors: Iterable[Iterable[float]]) -> List[List[float]]:
    """Quantize vectors to 8-bit integers scaled back to floats."""

    quantized: List[List[float]] = []
    for vector in vectors:
        max_val = max(abs(value) for value in vector) or 1.0
        scale = max_val / 127.0
        quantized_vector = [float(int(value / scale)) * scale for value in vector]
        quantized.append(quantized_vector)
    return quantized
