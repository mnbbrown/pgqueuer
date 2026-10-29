from __future__ import annotations

import asyncio
from datetime import timedelta
from itertools import pairwise
from time import perf_counter
from unittest.mock import AsyncMock

import pytest

from pgqueuer import PgQueuer
from pgqueuer.adapters.drivers.asyncpg import AsyncpgDriver
from pgqueuer.adapters.inmemory import InMemoryQueries
from pgqueuer.adapters.persistence.queries import Queries
from pgqueuer.domain.models import Job, TableChangedEvent, utc_now
from pgqueuer.domain.types import QueueEntrypoint, QueueManagerId
from pgqueuer.ports.repository import EntrypointExecutionParameter


@pytest.mark.parametrize("postgres", [False, True])
async def test_cooldown_spaces_claims_across_batches(
    monkeypatch: pytest.MonkeyPatch,
    queries: InMemoryQueries,
    apgdriver: AsyncpgDriver,
    postgres: bool,
) -> None:
    repository = Queries(driver=apgdriver) if postgres else queries
    queue = PgQueuer(connection=repository.driver, queries=repository)
    queue.qm.fetch_cooldown = timedelta(milliseconds=40)
    starts: list[float] = []
    completed: list[int] = []
    original = repository.dequeue

    async def record_claim(
        batch_size: int,
        entrypoints: dict[QueueEntrypoint, EntrypointExecutionParameter],
        queue_manager_id: QueueManagerId,
        global_concurrency_limit: int | None,
        heartbeat_timeout: timedelta,
    ) -> list[Job]:
        starts.append(perf_counter())
        return await original(
            batch_size=batch_size,
            entrypoints=entrypoints,
            queue_manager_id=queue_manager_id,
            global_concurrency_limit=global_concurrency_limit,
            heartbeat_timeout=heartbeat_timeout,
        )

    monkeypatch.setattr(repository, "dequeue", record_claim)

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        completed.append(job.id)
        if len(completed) == 4:
            queue.shutdown.set()

    await repository.enqueue(entrypoint=["work"] * 4, payload=[None] * 4, priority=[0] * 4)
    async with asyncio.timeout(3):
        await queue.run(batch_size=1, log_aggregation_interval=timedelta(0))

    assert len(completed) == 4
    assert all(later - earlier >= 0.035 for earlier, later in pairwise(starts))


async def test_notification_during_cooldown_is_not_lost() -> None:
    queue = PgQueuer.in_memory()
    queue.qm.fetch_cooldown = timedelta(milliseconds=100)
    checked = asyncio.Event()
    handled = asyncio.Event()
    queue.qm.on_dequeue = lambda _duration, _count: checked.set()

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        handled.set()
        queue.shutdown.set()

    async def enqueue() -> None:
        await checked.wait()
        await queue.qm.queries.enqueue(entrypoint="work", payload=None)

    async with asyncio.timeout(2):
        await asyncio.gather(
            queue.run(dequeue_timeout=timedelta(hours=1)),
            enqueue(),
        )
    assert handled.is_set()


async def test_shutdown_interrupts_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = PgQueuer.in_memory()
    queue.qm.fetch_cooldown = timedelta(minutes=1)
    checked = asyncio.Event()
    queue.qm.on_dequeue = lambda _duration, _count: checked.set()
    claim = AsyncMock(wraps=queue.qm.queries.dequeue)
    monkeypatch.setattr(queue.qm.queries, "dequeue", claim)

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        pytest.fail("Shutdown must interrupt the next claim")

    async def shutdown() -> None:
        await checked.wait()
        await queue.qm.queries.enqueue(entrypoint="work", payload=None)
        await asyncio.sleep(0.05)
        queue.shutdown.set()

    async with asyncio.timeout(1):
        await asyncio.gather(queue.run(), shutdown())
    assert claim.await_count == 1


@pytest.mark.parametrize("postgres", [False, True])
async def test_full_worker_skips_queries_and_wakes_on_completion(
    monkeypatch: pytest.MonkeyPatch,
    queries: InMemoryQueries,
    apgdriver: AsyncpgDriver,
    postgres: bool,
) -> None:
    repository = Queries(driver=apgdriver) if postgres else queries
    queue = PgQueuer(connection=repository.driver, queries=repository)
    queue.qm.skip_dequeue_when_full = True
    release = asyncio.Event()
    full = asyncio.Event()
    next_job = asyncio.Event()
    started: list[int] = []
    claim = AsyncMock(wraps=repository.dequeue)
    probe = AsyncMock(wraps=repository.eligible_queued_work)
    monkeypatch.setattr(repository, "dequeue", claim)
    monkeypatch.setattr(repository, "eligible_queued_work", probe)
    # Completion must wake locally, independently of the delete notification or polling.
    timeout_probe = AsyncMock(return_value=timedelta(hours=1))
    monkeypatch.setattr(queue.qm, "_effective_dequeue_timeout", timeout_probe)

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        started.append(job.id)
        if len(started) <= 2:
            if len(started) == 2:
                full.set()
            await release.wait()
        else:
            next_job.set()
            queue.shutdown.set()

    await repository.enqueue(entrypoint=["work"] * 3, payload=[None] * 3, priority=[0] * 3)

    async def check_full() -> None:
        await full.wait()
        await asyncio.sleep(0.02)
        before = claim.await_count
        assert before == 2
        for _ in range(5):
            event = TableChangedEvent(
                channel=repository.settings.channel,
                type="table_changed_event",
                operation="insert",
                table="pgqueuer",
                sent_at=utc_now(),
                entrypoints=[QueueEntrypoint("work")],
            )
            await repository.driver.notify(
                channel=repository.settings.channel, payload=event.model_dump_json()
            )
            await asyncio.sleep(0.01)
        assert claim.await_count == before
        probe.assert_not_awaited()
        timeout_probe.assert_not_awaited()
        # Suppress any completion notification to exercise the local wakeup path.
        monkeypatch.setattr(repository.driver, "notify", AsyncMock())
        release.set()
        async with asyncio.timeout(0.5):
            await next_job.wait()

    # The independent connection consumes PG notifications in the repository driver.
    async with asyncio.timeout(3):
        await asyncio.gather(
            queue.run(batch_size=1, max_concurrent_tasks=2, log_aggregation_interval=timedelta(0)),
            check_full(),
        )
    assert len(started) == 3


async def test_negative_cooldown_is_rejected() -> None:
    queue = PgQueuer.in_memory()
    queue.qm.fetch_cooldown = timedelta(milliseconds=-1)
    with pytest.raises(ValueError, match="fetch_cooldown"):
        await queue.run()


@pytest.mark.parametrize("gate", [False, True])
async def test_unlimited_worker_does_not_wait_for_local_capacity(gate: bool) -> None:
    queue = PgQueuer.in_memory()
    queue.qm.skip_dequeue_when_full = gate
    completed: list[int] = []

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        completed.append(job.id)
        if len(completed) == 3:
            queue.shutdown.set()

    await queue.qm.queries.enqueue(
        entrypoint=["work"] * 3,
        payload=[None] * 3,
        priority=[0] * 3,
    )
    async with asyncio.timeout(1):
        await queue.run(batch_size=1)
    assert len(completed) == 3


async def test_busy_notifications_do_not_bypass_empty_claim_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = PgQueuer.in_memory()
    queue.qm.fetch_cooldown = timedelta(milliseconds=40)
    first_check = asyncio.Event()
    starts: list[float] = []

    async def empty_claim(
        batch_size: int,
        entrypoints: dict[QueueEntrypoint, EntrypointExecutionParameter],
        queue_manager_id: QueueManagerId,
        global_concurrency_limit: int | None,
        heartbeat_timeout: timedelta,
    ) -> list[Job]:
        starts.append(perf_counter())
        first_check.set()
        return []

    monkeypatch.setattr(queue.qm.queries, "dequeue", empty_claim)

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        pytest.fail("There are no jobs")

    async def notify() -> None:
        await first_check.wait()
        for _ in range(60):
            event = TableChangedEvent(
                channel=queue.qm.channel,
                type="table_changed_event",
                operation="insert",
                table="pgqueuer",
                sent_at=utc_now(),
                entrypoints=[QueueEntrypoint("work")],
            )
            await queue.qm.queries.driver.notify(
                channel=queue.qm.channel,
                payload=event.model_dump_json(),
            )
            await asyncio.sleep(0.002)
        queue.shutdown.set()

    async with asyncio.timeout(2):
        await asyncio.gather(queue.run(), notify())
    assert len(starts) >= 2
    assert all(later - earlier >= 0.035 for earlier, later in pairwise(starts))


async def test_listener_failure_interrupts_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = PgQueuer.in_memory()
    queue.qm.fetch_cooldown = timedelta(minutes=1)
    checked = asyncio.Event()
    fail_health = asyncio.Event()
    queue.qm.on_dequeue = lambda _duration, _count: checked.set()

    async def health() -> None:
        await fail_health.wait()
        raise RuntimeError("listener failed")

    monkeypatch.setattr(queue.qm, "_run_periodic_health_check", health)

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        pytest.fail("Listener failure must stop the next claim")

    async def fail() -> None:
        await checked.wait()
        await queue.qm.queries.enqueue(entrypoint="work", payload=None)
        await asyncio.sleep(0.02)
        fail_health.set()

    async with asyncio.timeout(1):
        with pytest.raises(RuntimeError, match="listener failed"):
            await asyncio.gather(queue.run(shutdown_on_listener_failure=True), fail())
