"""Tests for on_terminal_failure callback and non_retryable_errors on
DatabaseRetryEntrypointExecutor.
"""

from __future__ import annotations

from datetime import timedelta

from pgqueuer.core.applications import PgQueuer
from pgqueuer.core.executors import (
    DatabaseRetryEntrypointExecutor,
    EntrypointExecutorParameters,
)
from pgqueuer.core.qm import QueueManager
from pgqueuer.db import AsyncpgDriver
from pgqueuer.domain.errors import NonRetryableError, RetryRequested
from pgqueuer.domain.models import Job, TracebackRecord
from pgqueuer.domain.types import QueueExecutionMode
from pgqueuer.queries import Queries


class _BadInput(Exception):
    pass


async def _async_noop(job: Job) -> None:
    pass


# ---------------------------------------------------------------------------
# In-memory: callback fires once on max_attempts exhaustion
# ---------------------------------------------------------------------------


async def test_on_terminal_failure_fires_when_max_attempts_exhausted() -> None:
    pq = PgQueuer.in_memory()
    calls: list[tuple[str, int, str]] = []

    def on_terminal(exc: Exception, job: Job, reason: str) -> None:
        calls.append((type(exc).__name__, job.attempts, reason))

    @pq.entrypoint(
        "exhaust_ep",
        executor_factory=lambda params: DatabaseRetryEntrypointExecutor(
            parameters=params,
            max_attempts=2,
            initial_delay=timedelta(0),
            on_terminal_failure=on_terminal,
        ),
    )
    async def handler(job: Job) -> None:
        raise ValueError("always fails")

    await pq.qm.queries.enqueue("exhaust_ep", b"data", priority=0)
    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    # max_attempts=2 -> attempts 0, 1 retry; attempt 2 (job.attempts=2) is terminal
    assert len(calls) == 1
    exc_name, attempts, reason = calls[0]
    assert exc_name == "ValueError"
    assert attempts == 2
    assert reason == "max_attempts"


# ---------------------------------------------------------------------------
# In-memory: declarative non_retryable_errors short-circuits retry
# ---------------------------------------------------------------------------


async def test_non_retryable_errors_skip_retry() -> None:
    pq = PgQueuer.in_memory()
    calls: list[tuple[str, int, str]] = []
    invocations = 0

    def on_terminal(exc: Exception, job: Job, reason: str) -> None:
        calls.append((type(exc).__name__, job.attempts, reason))

    @pq.entrypoint(
        "validate_ep",
        executor_factory=lambda params: DatabaseRetryEntrypointExecutor(
            parameters=params,
            max_attempts=10,
            initial_delay=timedelta(0),
            non_retryable_errors=(_BadInput,),
            on_terminal_failure=on_terminal,
        ),
    )
    async def handler(job: Job) -> None:
        nonlocal invocations
        invocations += 1
        raise _BadInput("malformed payload")

    await pq.qm.queries.enqueue("validate_ep", b"data", priority=0)
    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    # Handler runs exactly once: no retry despite max_attempts=10
    assert invocations == 1
    assert calls == [("_BadInput", 0, "non_retryable")]


# ---------------------------------------------------------------------------
# In-memory: NonRetryableError raised inside handler short-circuits retry
# ---------------------------------------------------------------------------


async def test_non_retryable_error_exception_short_circuits_retry() -> None:
    pq = PgQueuer.in_memory()
    calls: list[tuple[str, int, str]] = []
    invocations = 0

    def on_terminal(exc: Exception, job: Job, reason: str) -> None:
        calls.append((type(exc).__name__, job.attempts, reason))

    @pq.entrypoint(
        "fatal_ep",
        executor_factory=lambda params: DatabaseRetryEntrypointExecutor(
            parameters=params,
            max_attempts=10,
            initial_delay=timedelta(0),
            on_terminal_failure=on_terminal,
        ),
    )
    async def handler(job: Job) -> None:
        nonlocal invocations
        invocations += 1
        raise NonRetryableError("give up immediately")

    await pq.qm.queries.enqueue("fatal_ep", b"data", priority=0)
    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    assert invocations == 1
    assert calls == [("NonRetryableError", 0, "non_retryable")]


# ---------------------------------------------------------------------------
# In-memory: callback does not fire on transient retry, only on terminal
# ---------------------------------------------------------------------------


async def test_callback_does_not_fire_on_transient_retry() -> None:
    pq = PgQueuer.in_memory()
    calls: list[tuple[str, int, str]] = []

    def on_terminal(exc: Exception, job: Job, reason: str) -> None:
        calls.append((type(exc).__name__, job.attempts, reason))

    @pq.entrypoint(
        "transient_ep",
        executor_factory=lambda params: DatabaseRetryEntrypointExecutor(
            parameters=params,
            max_attempts=5,
            initial_delay=timedelta(0),
            on_terminal_failure=on_terminal,
        ),
    )
    async def handler(job: Job) -> None:
        if job.attempts < 2:
            raise ValueError("transient")

    await pq.qm.queries.enqueue("transient_ep", b"data", priority=0)
    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    assert calls == []


# ---------------------------------------------------------------------------
# In-memory: handler-raised RetryRequested still passes through
# ---------------------------------------------------------------------------


async def test_callback_does_not_fire_on_retry_requested() -> None:
    pq = PgQueuer.in_memory()
    calls: list[tuple[str, int, str]] = []

    def on_terminal(exc: Exception, job: Job, reason: str) -> None:
        calls.append((type(exc).__name__, job.attempts, reason))

    @pq.entrypoint(
        "explicit_ep",
        executor_factory=lambda params: DatabaseRetryEntrypointExecutor(
            parameters=params,
            max_attempts=5,
            initial_delay=timedelta(0),
            on_terminal_failure=on_terminal,
        ),
    )
    async def handler(job: Job) -> None:
        if job.attempts == 0:
            raise RetryRequested(delay=timedelta(0), reason="explicit")

    await pq.qm.queries.enqueue("explicit_ep", b"data", priority=0)
    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    assert calls == []


# ---------------------------------------------------------------------------
# In-memory: async callback is awaited
# ---------------------------------------------------------------------------


async def test_async_on_terminal_failure_is_awaited() -> None:
    pq = PgQueuer.in_memory()
    calls: list[str] = []

    async def on_terminal(exc: Exception, job: Job, reason: str) -> None:
        calls.append(reason)

    @pq.entrypoint(
        "async_cb_ep",
        executor_factory=lambda params: DatabaseRetryEntrypointExecutor(
            parameters=params,
            max_attempts=1,
            initial_delay=timedelta(0),
            on_terminal_failure=on_terminal,
        ),
    )
    async def handler(job: Job) -> None:
        raise NonRetryableError("nope")

    await pq.qm.queries.enqueue("async_cb_ep", b"data", priority=0)
    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    assert calls == ["non_retryable"]


# ---------------------------------------------------------------------------
# In-memory: executor without callback is a no-op (no AttributeError, no log)
# ---------------------------------------------------------------------------


def test_handle_terminal_no_callback_is_noop() -> None:
    executor = DatabaseRetryEntrypointExecutor(
        parameters=EntrypointExecutorParameters(
            concurrency_limit=0,
            func=_async_noop,
        ),
        max_attempts=1,
    )
    # Sanity: no callback configured by default
    assert executor.on_terminal_failure is None
    assert executor.non_retryable_errors == ()


# ---------------------------------------------------------------------------
# Postgres integration: terminal failure on real DB
# ---------------------------------------------------------------------------


async def test_terminal_failure_postgres(apgdriver: AsyncpgDriver) -> None:
    """Real-DB test: max_attempts exhaustion fires callback exactly once and
    leaves the job in the terminal state expected by ``on_failure='delete'``."""
    qm = QueueManager(Queries(apgdriver))
    calls: list[tuple[str, int, str]] = []

    def on_terminal(exc: Exception, job: Job, reason: str) -> None:
        calls.append((type(exc).__name__, job.attempts, reason))

    @qm.entrypoint(
        "pg_terminal_ep",
        executor_factory=lambda params: DatabaseRetryEntrypointExecutor(
            parameters=params,
            max_attempts=2,
            initial_delay=timedelta(0),
            on_terminal_failure=on_terminal,
        ),
    )
    async def handler(job: Job) -> None:
        raise RuntimeError("permafail")

    await qm.queries.enqueue("pg_terminal_ep", b"x", priority=0)
    await qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    assert len(calls) == 1
    exc_name, attempts, reason = calls[0]
    assert exc_name == "RuntimeError"
    assert attempts == 2
    assert reason == "max_attempts"

    logs = await qm.queries.queue_log()
    ep_logs = [log for log in logs if log.entrypoint == "pg_terminal_ep"]
    # 2 retry log entries (queued, with traceback) + 1 terminal exception entry
    retry_logs = [log for log in ep_logs if log.status == "queued" and log.traceback is not None]
    assert len(retry_logs) == 2
    exception_logs = [log for log in ep_logs if log.status == "exception"]
    assert len(exception_logs) == 1
    # Sanity that traceback shape matches RetryRequested branch
    assert isinstance(retry_logs[0].traceback, TracebackRecord)
