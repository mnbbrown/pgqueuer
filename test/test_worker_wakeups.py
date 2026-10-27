from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from pgqueuer import PgQueuer
from pgqueuer.adapters.drivers.asyncpg import AsyncpgDriver
from pgqueuer.adapters.inmemory import InMemoryQueries
from pgqueuer.adapters.persistence.queries import Queries
from pgqueuer.core import listeners
from pgqueuer.domain.models import AnyEvent, Job, TableChangedEvent, utc_now
from pgqueuer.domain.types import QueueEntrypoint, QueueManagerId
from pgqueuer.ports.repository import EntrypointExecutionParameter


@pytest.mark.parametrize("postgres", [False, True])
async def test_notification_burst_requires_one_queue_recheck(
    monkeypatch: pytest.MonkeyPatch,
    queries: InMemoryQueries,
    apgdriver: AsyncpgDriver,
    postgres: bool,
) -> None:
    """A burst while dequeue is in flight must not cause one poll per notification."""
    repository = Queries(apgdriver) if postgres else queries
    queue = PgQueuer(repository.driver, queries=repository)
    received = 0
    delivered = asyncio.Event()

    def observe(payload: str | bytes | bytearray) -> None:
        nonlocal received
        if isinstance(AnyEvent.model_validate_json(payload).root, TableChangedEvent):
            received += 1
            if received == 100:
                delivered.set()

    await repository.driver.add_listener(repository.settings.channel, observe)
    started = asyncio.Event()
    release = asyncio.Event()
    original_dequeue = queue.qm.queries.dequeue
    dequeue = AsyncMock(wraps=original_dequeue)
    original_wait = listeners.wait_for_notice_event

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        pytest.fail("The queue is empty")

    async def hold_first_dequeue(
        batch_size: int,
        entrypoints: dict[QueueEntrypoint, EntrypointExecutionParameter],
        queue_manager_id: QueueManagerId,
        global_concurrency_limit: int | None,
        heartbeat_timeout: timedelta,
    ) -> list[Job]:
        if dequeue.await_count == 1:
            started.set()
            await release.wait()
        return await original_dequeue(
            batch_size=batch_size,
            entrypoints=entrypoints,
            queue_manager_id=queue_manager_id,
            global_concurrency_limit=global_concurrency_limit,
            heartbeat_timeout=heartbeat_timeout,
        )

    def stop_when_idle(
        notices: listeners.PGNoticeEventListener,
        timeout: timedelta,
    ) -> asyncio.Task[TableChangedEvent | None]:
        if notices.empty():
            queue.qm.shutdown.set()
        return original_wait(notices, timeout)

    dequeue.side_effect = hold_first_dequeue
    monkeypatch.setattr(queue.qm.queries, "dequeue", dequeue)
    monkeypatch.setattr(listeners, "wait_for_notice_event", stop_when_idle)

    async def notify() -> None:
        await started.wait()
        now = utc_now()
        for index in range(100):
            event = TableChangedEvent(
                channel=repository.settings.channel,
                type="table_changed_event",
                operation="insert",
                table="pgqueuer",
                sent_at=now + timedelta(microseconds=index),
            )
            await repository.driver.notify(repository.settings.channel, event.model_dump_json())
        await delivered.wait()
        release.set()

    async with asyncio.timeout(2):
        await asyncio.gather(
            queue.qm.run(log_aggregation_interval=timedelta(0)),
            notify(),
        )

    assert received == 100
    assert dequeue.await_count == 2


async def test_notification_during_dequeue_wakes_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enqueue after an empty snapshot must survive until the next dequeue."""
    queue = PgQueuer.in_memory()
    empty_snapshot = asyncio.Event()
    release = asyncio.Event()
    handled = asyncio.Event()
    original_dequeue = queue.qm.queries.dequeue
    dequeue = AsyncMock(wraps=original_dequeue)

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        handled.set()
        queue.qm.shutdown.set()

    async def hold_empty_snapshot(
        batch_size: int,
        entrypoints: dict[QueueEntrypoint, EntrypointExecutionParameter],
        queue_manager_id: QueueManagerId,
        global_concurrency_limit: int | None,
        heartbeat_timeout: timedelta,
    ) -> list[Job]:
        result = await original_dequeue(
            batch_size=batch_size,
            entrypoints=entrypoints,
            queue_manager_id=queue_manager_id,
            global_concurrency_limit=global_concurrency_limit,
            heartbeat_timeout=heartbeat_timeout,
        )
        if dequeue.await_count == 1:
            assert result == []
            empty_snapshot.set()
            await release.wait()
        return result

    dequeue.side_effect = hold_empty_snapshot
    monkeypatch.setattr(queue.qm.queries, "dequeue", dequeue)
    # Isolate the notification path from the short-timeout eligibility fallback.
    monkeypatch.setattr(
        queue.qm,
        "_effective_dequeue_timeout",
        AsyncMock(return_value=timedelta(seconds=30)),
    )

    async def enqueue_during_query() -> None:
        await empty_snapshot.wait()
        await queue.qm.queries.enqueue("work", None)
        release.set()

    async with asyncio.timeout(2):
        await asyncio.gather(
            queue.qm.run(log_aggregation_interval=timedelta(0)),
            enqueue_during_query(),
        )

    assert handled.is_set()


async def test_fallback_poll_finds_job_without_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coalescing must leave periodic pickup working when NOTIFY is lost."""
    queue = PgQueuer.in_memory()
    checked = asyncio.Event()
    handled = asyncio.Event()
    queue.qm.on_dequeue = lambda _duration, _count: checked.set()
    monkeypatch.setattr(queue.qm.queries, "emit_table_changed", AsyncMock())

    @queue.entrypoint("work")
    async def work(job: Job) -> None:
        handled.set()
        queue.qm.shutdown.set()

    async def enqueue_silently() -> None:
        await checked.wait()
        await queue.qm.queries.enqueue("work", None)

    async with asyncio.timeout(2):
        await asyncio.gather(
            queue.qm.run(
                dequeue_timeout=timedelta(milliseconds=20),
                log_aggregation_interval=timedelta(0),
            ),
            enqueue_silently(),
        )

    assert handled.is_set()
