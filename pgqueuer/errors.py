from __future__ import annotations

from datetime import datetime


class PgqException(Exception):
    """Base class for all exceptions raised by PGQueuer."""


class RetryException(PgqException):
    """Exception raised for retry-related errors in PGQueuer."""


class RetryableException(RetryException):
    """Exception raised for retry-related errors in PGQueuer."""

    def __init__(self, schedule_for: datetime | None):
        self.schedule_for = schedule_for


class MaxRetriesExceeded(RetryException):
    """Exception raised when all retry attempts have been exhausted."""


class MaxTimeExceeded(RetryException):
    """Exception raised when the maximum time limit for retries has been exceeded."""


class DuplicateJobError(PgqException):
    """Raised when enqueue violates a deduplication constraint."""

    def __init__(self, dedupe_key: list[str | None]) -> None:
        super().__init__()
        self.dedupe_key = dedupe_key


class FailingListenerError(PgqException):
    """Raised when a listener fails to process a job."""


class NonRetryableError(PgqException):
    """
    Raise this from a job to skip retries and fail immediately.

    Use for permanent failures like:
    - Invalid input data
    - Resource not found (404)
    - Permission denied
    - Business logic violations
    """

    pass
