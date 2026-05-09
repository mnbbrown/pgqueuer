"""Tests for QueueManager observability callbacks."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from typing import cast

import pytest

from pgqueuer.core.applications import PgQueuer
from pgqueuer.core.qm import QueueManager
from pgqueuer.db import AsyncpgDriver
from pgqueuer.domain import errors
from pgqueuer.domain.models import HealthCheckEvent, Job
from pgqueuer.domain.types import QueueExecutionMode
from pgqueuer.queries import Queries

# ---------------------------------------------------------------------------
# on_dequeue
# ---------------------------------------------------------------------------


async def test_on_dequeue_fires_with_duration_and_count() -> None:
    pq = PgQueuer.in_memory()
    samples: list[tuple[float, int]] = []
    pq.qm.on_dequeue = lambda duration, count: samples.append((duration, count))

    @pq.entrypoint("a")
    async def _(job: Job) -> None:
        return None

    await pq.qm.queries.enqueue("a", b"1", priority=0)
    await pq.qm.queries.enqueue("a", b"2", priority=0)

    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    # At least one dequeue saw both jobs; all samples have non-negative duration
    assert samples, "on_dequeue was never called"
    assert any(count == 2 for _, count in samples)
    assert all(duration >= 0 for duration, _ in samples)


async def test_on_dequeue_fires_even_when_queue_is_empty() -> None:
    pq = PgQueuer.in_memory()
    samples: list[tuple[float, int]] = []
    pq.qm.on_dequeue = lambda duration, count: samples.append((duration, count))

    @pq.entrypoint("a")
    async def _(job: Job) -> None:
        return None

    # Drain mode with no jobs — fetch_jobs is invoked once, gets 0, breaks out
    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=100,
        dequeue_timeout=timedelta(seconds=1),
    )

    assert samples, "on_dequeue was never called"
    assert all(count == 0 for _, count in samples)


# ---------------------------------------------------------------------------
# on_dispatch
# ---------------------------------------------------------------------------


async def test_on_dispatch_fires_per_dispatched_job() -> None:
    pq = PgQueuer.in_memory()
    samples: list[tuple[int, int]] = []
    pq.qm.on_dispatch = lambda active, mx: samples.append((active, mx))

    @pq.entrypoint("a")
    async def _(job: Job) -> None:
        return None

    for i in range(3):
        await pq.qm.queries.enqueue("a", str(i).encode(), priority=0)

    await pq.qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=50,
        dequeue_timeout=timedelta(seconds=1),
    )

    # Three dispatches → three samples; max_concurrent should be 50 in each
    assert len(samples) == 3
    assert all(mx == 50 for _, mx in samples)
    # active count is observed >= 1 immediately after task creation
    assert all(active >= 1 for active, _ in samples)


# ---------------------------------------------------------------------------
# on_listener_health_check
# ---------------------------------------------------------------------------


async def test_on_listener_health_check_fires_on_healthy_check() -> None:
    pq = PgQueuer.in_memory()
    samples: list[tuple[bool, float]] = []
    pq.qm.on_listener_health_check = lambda healthy, dur: samples.append((healthy, dur))

    # Stub listener_healthy to return successfully (in-memory has no real listener
    # responding to health-check notifications). Return value is unused by the
    # periodic loop, so a cast suffices.
    async def _ok_listener_healthy(
        timeout: timedelta = timedelta(seconds=10),
    ) -> HealthCheckEvent:
        return cast(HealthCheckEvent, None)

    pq.qm.listener_healthy = _ok_listener_healthy  # type: ignore[method-assign]

    async def _stop_after_one() -> None:
        await asyncio.sleep(0.05)
        pq.qm.shutdown.set()

    stopper = asyncio.create_task(_stop_after_one())
    try:
        await pq.qm._run_periodic_health_check(interval=timedelta(milliseconds=10))
    finally:
        stopper.cancel()
        with contextlib.suppress(BaseException):
            await stopper

    assert samples, "on_listener_health_check was never called"
    assert all(healthy is True for healthy, _ in samples)
    assert all(dur >= 0 for _, dur in samples)


async def test_on_listener_health_check_fires_on_failure() -> None:
    pq = PgQueuer.in_memory()
    samples: list[tuple[bool, float]] = []
    pq.qm.on_listener_health_check = lambda healthy, dur: samples.append((healthy, dur))

    async def _broken_listener_healthy(
        timeout: timedelta = timedelta(seconds=10),
    ) -> HealthCheckEvent:
        raise errors.FailingListenerError

    pq.qm.listener_healthy = _broken_listener_healthy  # type: ignore[method-assign]

    with pytest.raises(errors.FailingListenerError):
        await pq.qm._run_periodic_health_check(interval=timedelta(milliseconds=10))

    assert samples, "on_listener_health_check was never called"
    assert samples[0][0] is False  # healthy=False on failure
    assert samples[0][1] >= 0


# ---------------------------------------------------------------------------
# Defaults — no callbacks set, no surprises
# ---------------------------------------------------------------------------


def test_default_callbacks_are_none() -> None:
    pq = PgQueuer.in_memory()
    assert pq.qm.on_dequeue is None
    assert pq.qm.on_dispatch is None
    assert pq.qm.on_listener_health_check is None


# ---------------------------------------------------------------------------
# Postgres integration
# ---------------------------------------------------------------------------


async def test_dequeue_and_dispatch_hooks_postgres(apgdriver: AsyncpgDriver) -> None:
    qm = QueueManager(Queries(apgdriver))
    dequeue_samples: list[tuple[float, int]] = []
    dispatch_samples: list[tuple[int, int]] = []
    qm.on_dequeue = lambda dur, count: dequeue_samples.append((dur, count))
    qm.on_dispatch = lambda active, mx: dispatch_samples.append((active, mx))

    @qm.entrypoint("pg_obs_ep")
    async def _(job: Job) -> None:
        return None

    for i in range(3):
        await qm.queries.enqueue("pg_obs_ep", str(i).encode(), priority=0)

    await qm.run(
        batch_size=10,
        mode=QueueExecutionMode.drain,
        max_concurrent_tasks=25,
        dequeue_timeout=timedelta(seconds=1),
    )

    assert dequeue_samples, "on_dequeue was never called"
    assert any(count == 3 for _, count in dequeue_samples)
    assert len(dispatch_samples) == 3
    assert all(mx == 25 for _, mx in dispatch_samples)
