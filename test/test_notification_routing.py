from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest
from anyio import CancelScope

from pgqueuer import QueueManager
from pgqueuer.adapters.drivers.asyncpg import AsyncpgDriver
from pgqueuer.adapters.inmemory import InMemoryQueries
from pgqueuer.adapters.persistence.queries import Queries
from pgqueuer.core.listeners import PGNoticeEventListener, default_event_router
from pgqueuer.domain.models import (
    AnyEvent,
    CancellationEvent,
    Context,
    HealthCheckEvent,
    Job,
    TableChangedEvent,
    utc_now,
)
from pgqueuer.domain.types import Channel, HealthCheckId, JobId, QueueEntrypoint, QueueManagerId
from pgqueuer.ports.repository import EntrypointExecutionParameter


@pytest.mark.parametrize(
    ("targets", "wakes"),
    [(None, True), ([], False), (["ours"], True), (["other"], False), (["other", "ours"], True)],
)
def test_router_only_wakes_for_registered_entrypoints(
    targets: list[str] | None,
    wakes: bool,
) -> None:
    notices = PGNoticeEventListener()
    router = default_event_router(
        notice_event_queue=notices,
        canceled={},
        pending_health_check={},
        entrypoints={QueueEntrypoint("ours")},
    )
    router(
        AnyEvent.model_validate(
            {
                "type": "table_changed_event",
                "channel": "ch_pgqueuer",
                "sent_at": utc_now(),
                "operation": "insert",
                "table": "pgqueuer",
                "entrypoints": targets,
            }
        )
    )
    assert (not notices.empty()) == wakes


async def test_routing_preserves_cancellation_and_health_checks() -> None:
    notices = PGNoticeEventListener()
    context = Context(cancellation=CancelScope(), resources={})
    health_id = HealthCheckId(uuid4())
    future = asyncio.Future[HealthCheckEvent]()
    router = default_event_router(
        notice_event_queue=notices,
        canceled={JobId(42): context},
        pending_health_check={health_id: future},
        entrypoints={QueueEntrypoint("ours")},
    )
    router(
        AnyEvent(
            root=CancellationEvent(
                channel=Channel("ch_pgqueuer"),
                sent_at=utc_now(),
                type="cancellation_event",
                ids=[JobId(42)],
            )
        )
    )
    router(
        AnyEvent(
            root=HealthCheckEvent(
                channel=Channel("ch_pgqueuer"),
                sent_at=utc_now(),
                type="health_check_event",
                id=health_id,
            )
        )
    )
    assert context.cancellation.cancel_called
    assert future.result().id == health_id
    assert notices.empty()


async def table_changes(queries: Queries) -> list[TableChangedEvent]:
    events: list[TableChangedEvent] = []

    def receive(payload: str | bytes | bytearray) -> None:
        event = AnyEvent.model_validate_json(payload).root
        if isinstance(event, TableChangedEvent):
            events.append(event)

    await queries.driver.add_listener(queries.settings.channel, receive)
    return events


async def test_bulk_insert_and_delete_emit_one_scoped_notification(
    apgdriver: AsyncpgDriver,
) -> None:
    queries = Queries(apgdriver)
    events = await table_changes(queries)
    ids = await queries.enqueue(["ours", "other", "ours"], [None] * 3, [0] * 3)
    await asyncio.sleep(0.02)
    assert len(events) == 1
    assert events[0].entrypoints == ["other", "ours"]
    events.clear()
    await apgdriver.execute("DELETE FROM pgqueuer WHERE id = ANY($1::bigint[])", ids)
    await asyncio.sleep(0.02)
    assert len(events) == 1
    assert events[0].entrypoints == ["other", "ours"]


async def test_large_routing_envelope_falls_back_to_broadcast(apgdriver: AsyncpgDriver) -> None:
    queries = Queries(apgdriver)
    events = await table_changes(queries)
    await queries.enqueue([f"task-{i}-" + "x" * 120 for i in range(100)], [None] * 100, [0] * 100)
    await asyncio.sleep(0.02)
    assert len(events) == 1
    assert events[0].entrypoints is None


async def test_zero_row_changes_do_not_wake_workers(apgdriver: AsyncpgDriver) -> None:
    queries = Queries(apgdriver)
    events = await table_changes(queries)
    await apgdriver.execute("DELETE FROM pgqueuer WHERE false")
    await asyncio.sleep(0.02)
    assert events == []


async def test_legacy_trigger_still_works_with_replaced_function(apgdriver: AsyncpgDriver) -> None:
    queries = Queries(apgdriver)
    await apgdriver.execute("DROP TRIGGER tg_pgqueuer_changed ON pgqueuer")
    await apgdriver.execute("DROP TRIGGER tg_pgqueuer_changed_insert ON pgqueuer")
    await apgdriver.execute("DROP TRIGGER tg_pgqueuer_changed_delete ON pgqueuer")
    await apgdriver.execute("""
        CREATE TRIGGER tg_pgqueuer_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE
        ON pgqueuer EXECUTE FUNCTION fn_pgqueuer_changed()
    """)
    events = await table_changes(queries)
    await queries.enqueue("ours", None)
    await asyncio.sleep(0.02)
    assert len(events) == 1
    assert events[0].entrypoints is None
    await queries.upgrade()
    assert (await queries.plan_upgrade()).statements == ()


async def test_rollback_emits_no_routing_notification(apgdriver: AsyncpgDriver) -> None:
    queries = Queries(apgdriver)
    events = await table_changes(queries)
    await apgdriver.execute("BEGIN")
    await queries.enqueue("ours", None)
    await apgdriver.execute("ROLLBACK")
    await asyncio.sleep(0.02)
    assert events == []


async def test_retry_and_requeue_notifications_are_scoped(apgdriver: AsyncpgDriver) -> None:
    queries = Queries(apgdriver)
    ids = await queries.enqueue("ours", None)
    await apgdriver.execute("UPDATE pgqueuer SET status = 'failed' WHERE id = $1", ids[0])
    events = await table_changes(queries)
    await queries.requeue_jobs(ids)
    await asyncio.sleep(0.02)
    assert len(events) == 1
    assert events[0].entrypoints == ["ours"]

    jobs = await queries.dequeue(
        batch_size=1,
        entrypoints={QueueEntrypoint("ours"): EntrypointExecutionParameter(0)},
        queue_manager_id=QueueManagerId(uuid4()),
        global_concurrency_limit=None,
        heartbeat_timeout=timedelta(seconds=30),
    )
    events.clear()
    await queries.retry_job(jobs[0], timedelta(0), None)
    await asyncio.sleep(0.02)
    assert len(events) == 1
    assert events[0].entrypoints == ["ours"]


@pytest.mark.parametrize("postgres", [False, True])
async def test_other_pool_does_not_query_for_inserted_jobs(
    apgdriver: AsyncpgDriver,
    queries: InMemoryQueries,
    postgres: bool,
) -> None:
    repository = Queries(apgdriver) if postgres else queries
    ours = QueueManager(repository)
    other = QueueManager(repository)
    ours_ready = asyncio.Event()
    other_ready = asyncio.Event()
    handled = asyncio.Event()
    other_checks = 0
    ours.on_dequeue = lambda _duration, _count: ours_ready.set()

    def checked(duration: float, count: int) -> None:
        nonlocal other_checks
        other_checks += 1
        other_ready.set()

    other.on_dequeue = checked

    @ours.entrypoint("ours")
    async def handle(job: Job) -> None:
        handled.set()

    @other.entrypoint("other")
    async def unexpected(job: Job) -> None:
        pytest.fail("The other pool must not claim our job")

    tasks = [
        asyncio.create_task(
            manager.run(
                dequeue_timeout=timedelta(seconds=30),
                log_aggregation_interval=timedelta(0),
            )
        )
        for manager in (ours, other)
    ]
    try:
        async with asyncio.timeout(3):
            await ours_ready.wait()
            await other_ready.wait()
            await repository.enqueue("ours", None)
            await handled.wait()
            await asyncio.sleep(0.02)
        assert other_checks == 1
    finally:
        ours.shutdown.set()
        other.shutdown.set()
        await asyncio.gather(*tasks)


async def test_large_explicit_routing_envelope_falls_back(apgdriver: AsyncpgDriver) -> None:
    queries = Queries(apgdriver)
    events = await table_changes(queries)
    await queries.notify_table_changed("update", [f"task-{i}-" + "x" * 120 for i in range(100)])
    await asyncio.sleep(0.02)
    assert len(events) == 1
    assert events[0].entrypoints is None


async def test_held_failure_notifies_workers_after_releasing_capacity(
    apgdriver: AsyncpgDriver,
) -> None:
    queries = Queries(apgdriver)
    await queries.enqueue(["ours", "ours"], [None, None], [0, 0])
    owner = QueueManagerId(uuid4())
    parameters = {QueueEntrypoint("ours"): EntrypointExecutionParameter(1)}
    first = await queries.dequeue(
        batch_size=1,
        entrypoints=parameters,
        queue_manager_id=owner,
        global_concurrency_limit=None,
        heartbeat_timeout=timedelta(seconds=30),
    )
    events = await table_changes(queries)
    await queries.log_jobs([(first[0], "failed", None)])
    await asyncio.sleep(0.02)
    assert len(events) == 1
    assert events[0].entrypoints == ["ours"]
    remaining = await queries.dequeue(
        batch_size=1,
        entrypoints=parameters,
        queue_manager_id=owner,
        global_concurrency_limit=None,
        heartbeat_timeout=timedelta(seconds=30),
    )
    assert len(remaining) == 1
    assert remaining[0].id != first[0].id
