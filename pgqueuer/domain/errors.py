from __future__ import annotations

from datetime import timedelta

from pgqueuer.domain.types import ColumnName, SqlType, TableName


class PgqException(Exception):  # noqa: N818 - public API name
    """Base class for all exceptions raised by PgQueuer."""


class RetryException(PgqException):
    """Exception raised for retry-related errors in PgQueuer."""


class RetryRequested(RetryException):
    """Raise inside a job handler to request a database-level retry.

    Instead of marking the job as a terminal failure, the job is re-queued
    with status 'queued', a bumped execute_after, and incremented attempts.

    Attributes:
        delay: Time to wait before the next attempt.
        reason: Optional human-readable explanation for the retry.
    """

    def __init__(
        self,
        delay: timedelta = timedelta(0),
        reason: str | None = None,
    ) -> None:
        super().__init__(reason or "Retry requested")
        self.delay = delay
        self.reason = reason


class DelayRequested(RetryRequested):
    """Reschedule the same job without incrementing its failure attempts.

    Existing attempts, payload and queue keys are retained. Normal retries
    still increment attempts. Set an overall expiry where pending work needs one.

    Usage example::

        raise DelayRequested(delay=timedelta(seconds=30), reason="Rate limited")
    """


class NonRetryableError(PgqException):
    """Raise inside a job handler to skip retry and fail terminally.

    When raised inside a handler wrapped by :class:`DatabaseRetryEntrypointExecutor`,
    the executor bypasses its retry logic and lets the exception propagate as a
    terminal failure (delete or hold, depending on the entrypoint's ``on_failure``
    setting).
    """


def retry_request(exc: BaseException) -> RetryRequested | None:
    """Return the retry *exc* asks for, or ``None``.

    A group made only of retry signals retains the longest requested delay.
    It remains a delay only if every member requests a delay; a normal retry
    consumes an attempt even when another member requests a longer delay.
    """
    if isinstance(exc, RetryRequested):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        found = [retry_request(inner) for inner in exc.exceptions]
        retries = [retry for retry in found if retry is not None]
        if retries and len(retries) == len(found):
            longest = max(retries, key=lambda retry: retry.delay)
            if isinstance(longest, DelayRequested) and any(
                not isinstance(retry, DelayRequested) for retry in retries
            ):
                return RetryRequested(delay=longest.delay, reason=longest.reason)
            return longest
    return None


class DuplicateJobError(PgqException):
    """Raised when enqueue violates a deduplication constraint."""

    def __init__(self, dedupe_key: list[str | None]) -> None:
        super().__init__()
        self.dedupe_key = dedupe_key


class FailingListenerError(PgqException):
    """Raised when a listener fails to process a job."""


class SchemaDriftError(PgqException):
    """Raised when an upgrade meets a schema change it cannot safely convert."""

    def __init__(
        self,
        *,
        table: TableName,
        column: ColumnName,
        installed: SqlType,
        declared: SqlType,
    ) -> None:
        super().__init__(
            f"{table}.{column} is {installed} in the database but declared {declared}; "
            "no supported conversion. Alter the column by hand, then re-run upgrade."
        )
        self.table = table
        self.column = column
        self.installed = installed
        self.declared = declared
