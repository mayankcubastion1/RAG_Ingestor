"""Input validation helpers for the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Represents validation outcome."""

    valid: bool
    message: str = ""

    @property
    def is_ok(self) -> bool:  # pragma: no cover - thin wrapper
        return self.valid


def require_non_empty(value: str, field_name: str) -> ValidationResult:
    """Validate that ``value`` is not empty."""

    if not value:
        return ValidationResult(False, f"{field_name} is required")
    return ValidationResult(True)


def require_positive_int(value: int, field_name: str) -> ValidationResult:
    if value <= 0:
        return ValidationResult(False, f"{field_name} must be positive")
    return ValidationResult(True)
