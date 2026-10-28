from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import asyncpg
import pytest

from pgqueuer import PgQueuer, QueueManager
from pgqueuer.adapters.drivers.asyncpg import AsyncpgDriver
from pgqueuer.adapters.inmemory import InMemoryQueries
from pgqueuer.adapters.persistence.queries import Queries
from pgqueuer.domain.models import AnyEvent, Job, TableChangedEvent
from pgqueuer.domain.types import QueueEntrypoint


class LegacyWakeupQueries:
    def __init__(self, queries: InMemoryQueries) -> None:
        self.queries = queries
        self.qbe = queries.qbe

    async def eligible_queued_work(self, entrypoints: list[QueueEntrypoint]) -> int:
        return await self.queries.eligible_queued_work(entrypoints)

    async def next_deferred_eta(self, entrypoints: list[QueueEntrypoint]) -> timedelta | None:
        return await self.queries.next_deferred_eta(entrypoints)

    async def queued_work(self, entrypoints: list[QueueEntrypoint]) -> int:
        return await self.queries.queued_work(entrypoints)


async def test_legacy_repository_keeps_existing_polling(queries: InMemoryQueries) -> None:
    manager = QueueManager(LegacyWakeupQueries(queries))  # type: ignore[arg-type]

    @manager.entrypoint("work")
    async def work(job: Job) -> None:
        pytest.fail("The queue is empty")

    assert await manager._effective_dequeue_timeout(timedelta(seconds=30)) == timedelta(seconds=30)


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.parametrize("delays", [[], [-1], [30], [-1, 30]])
async def test_queued_eta_covers_ready_and_future_jobs(
    queries: InMemoryQueries,
    apgdriver: AsyncpgDriver,
    postgres: bool,
    delays: list[int],
) -> None:
    repository = Queries(apgdriver) if postgres else queries
    for delay in delays:
        await repository.enqueue("work", None, execute_after=timedelta(seconds=delay))
    await repository.enqueue("other", None, execute_after=timedelta(seconds=-10))
    eta = await repository.next_queued_eta([QueueEntrypoint("work")])
    if not delays:
        assert eta is None
    else:
        assert eta is not None
        assert abs(eta.total_seconds() - min(delays)) < 1


@pytest.mark.parametrize(("eta", "elapsed", "expected"), [(2.0, 1.0, 1.0), (0.5, 1.0, 0.1)])
async def test_probe_duration_is_removed_from_deadline(
    queries: InMemoryQueries,
    monkeypatch: pytest.MonkeyPatch,
    eta: float,
    elapsed: float,
    expected: float,
) -> None:
    queue = PgQueuer(queries.driver, queries=queries)

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        pytest.fail("The queue is empty")

    monkeypatch.setattr(queries, "next_queued_eta", AsyncMock(return_value=timedelta(seconds=eta)))
    clock = iter([10.0, 10.0 + elapsed])
    monkeypatch.setattr("pgqueuer.core.qm.perf_counter", lambda: next(clock))
    assert await queue.qm._effective_dequeue_timeout(timedelta(seconds=30)) == timedelta(
        seconds=expected
    )


@pytest.mark.parametrize("lost", [False, True])
async def test_enqueue_after_probe_snapshot_is_not_stranded(
    apgdriver: AsyncpgDriver,
    dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    lost: bool,
) -> None:
    """An enqueue after the probe snapshot must survive a delivered or lost notification."""
    repository = Queries(apgdriver)
    queue = PgQueuer(repository.driver, queries=repository)
    snapshot = asyncio.Event()
    release = asyncio.Event()
    delivered = asyncio.Event()
    handled = asyncio.Event()
    original_probe = repository.next_queued_eta
    first = True

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        handled.set()
        queue.qm.shutdown.set()

    async def hold_response(entrypoints: list[QueueEntrypoint]) -> timedelta | None:
        nonlocal first
        result = await original_probe(entrypoints)
        if first:
            first = False
            assert result is None
            snapshot.set()
            await release.wait()
        return result

    def observe(payload: str | bytes | bytearray) -> None:
        if isinstance(AnyEvent.model_validate_json(payload).root, TableChangedEvent):
            delivered.set()

    await repository.driver.add_listener(repository.settings.channel, observe)
    monkeypatch.setattr(repository, "next_queued_eta", hold_response)

    async def enqueue_after_snapshot() -> None:
        await snapshot.wait()
        producer = await asyncpg.connect(dsn=dsn)
        try:
            writer = Queries.from_asyncpg_connection(producer)
            if lost:
                async with producer.transaction():
                    await producer.execute("SET LOCAL session_replication_role='replica'")
                    await writer.enqueue("work", None)
            else:
                await writer.enqueue("work", None)
                await delivered.wait()
        finally:
            await producer.close()
        release.set()

    async with asyncio.timeout(2):
        await asyncio.gather(
            queue.qm.run(
                dequeue_timeout=timedelta(milliseconds=100), log_aggregation_interval=timedelta(0)
            ),
            enqueue_after_snapshot(),
        )
    assert handled.is_set()
