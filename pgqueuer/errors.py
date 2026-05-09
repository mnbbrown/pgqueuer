"""Backward-compatibility shim. Canonical: pgqueuer.domain.errors"""

from pgqueuer.domain.errors import (
    DuplicateJobError,
    FailingListenerError,
    NonRetryableError,
    PgqException,
    RetryException,
    RetryRequested,
)

__all__ = [
    "DuplicateJobError",
    "FailingListenerError",
    "NonRetryableError",
    "PgqException",
    "RetryException",
    "RetryRequested",
]
