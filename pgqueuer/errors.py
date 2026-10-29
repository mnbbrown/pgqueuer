"""Backward-compatibility shim. Canonical: pgqueuer.domain.errors."""

from __future__ import annotations

from pgqueuer.domain.errors import (
    DelayRequested,
    DuplicateJobError,
    FailingListenerError,
    NonRetryableError,
    PgqException,
    RetryException,
    RetryRequested,
)

__all__ = [
    "DelayRequested",
    "DuplicateJobError",
    "FailingListenerError",
    "NonRetryableError",
    "PgqException",
    "RetryException",
    "RetryRequested",
]
