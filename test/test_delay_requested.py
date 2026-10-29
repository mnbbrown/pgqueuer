"""Expected deferrals retain the failure budget on both storage adapters."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from pgqueuer import DelayRequested, Job, PgQueuer
from pgqueuer.db import AsyncpgDriver
from pgqueuer.domain.errors import RetryRequested, retry_request
from pgqueuer.executors import DatabaseRetryEntrypointExecutor
from pgqueuer.types import QueueExecutionMode


@pytest.mark.parametrize("postgres", [False, True])
async def test_delays_preserve_attempts_for_later_failure(
    apgdriver: AsyncpgDriver,
    *,
    postgres: bool,
) -> None:
    pq = PgQueuer(apgdriver) if postgres else PgQueuer.in_memory()
    seen: list[int] = []

    @pq.entrypoint(
        "delay_ep",
        executor_factory=lambda parameters: DatabaseRetryEntrypointExecutor(
            parameters=parameters,
            max_attempts=1,
            initial_delay=timedelta(0),
        ),
    )
    async def handler(job: Job) -> None:
        seen.append(job.attempts)
        if len(seen) <= 3:
            raise DelayRequested(delay=timedelta(0), reason="rate limited")
        if len(seen) == 4:
            raise ConnectionError("temporary provider failure")

    ids = await pq.qm.queries.enqueue("delay_ep", b"payload", priority=0)
    async with asyncio.timeout(5):
        await pq.qm.run(mode=QueueExecutionMode.drain)
    assert seen == [0, 0, 0, 0, 1]
    logs = await pq.qm.queries.queue_log()
    assert [log.job_id for log in logs if log.status == "successful"] == ids


@pytest.mark.parametrize("with_failure_retry", [False, True])
def test_grouped_delays_keep_longest_delay_without_hiding_failure(
    *,
    with_failure_retry: bool,
) -> None:
    short = RetryRequested() if with_failure_retry else DelayRequested()
    grouped = ExceptionGroup("pending", [short, DelayRequested(delay=timedelta(seconds=30))])
    result = retry_request(grouped)
    assert result is not None
    assert result.delay == timedelta(seconds=30)
    assert isinstance(result, DelayRequested) is not with_failure_retry
