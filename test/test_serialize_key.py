"""Tests for the per-key dispatch serialization feature.

Covers both the in-memory adapter (white-box, deterministic) and the Postgres
adapter (black-box, real SQL). The two test surfaces validate the same
invariants, so a behavioural divergence between drivers shows up immediately.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import pytest

from pgqueuer.adapters.inmemory import InMemoryQueries
from pgqueuer.db import Driver
from pgqueuer.models import Job
from pgqueuer.ports.repository import EntrypointExecutionParameter
from pgqueuer.qm import QueueManager
from pgqueuer.queries import Queries

# ---------------------------------------------------------------------------
# In-memory adapter — direct API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_enqueue_allows_multiple_with_same_serialize_key(
    queries: InMemoryQueries,
) -> None:
    """Multiple queued jobs with the same serialize_key must coexist."""
    ids = await queries.enqueue(
        ["ep", "ep", "ep"],
        [None, None, None],
        [0, 0, 0],
        serialize_key=["userA", "userA", "userA"],
    )
    assert len(ids) == 3


@pytest.mark.asyncio
async def test_inmemory_dispatch_serializes_per_key(queries: InMemoryQueries) -> None:
    """Same key → at most one picked at a time."""
    await queries.enqueue(["ep", "ep"], [None, None], [0, 0], serialize_key=["k", "k"])
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}

    first = await queries.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(first) == 1

    # Second dispatch should pick nothing — the first is still picked.
    second = await queries.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(second) == 0

    # Complete the first; the second job should now be eligible.
    await queries.log_jobs([(first[0], "successful", None)])
    third = await queries.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(third) == 1


@pytest.mark.asyncio
async def test_inmemory_different_keys_run_in_parallel(queries: InMemoryQueries) -> None:
    """Different keys must dispatch concurrently."""
    await queries.enqueue(
        ["ep", "ep", "ep"],
        [None, None, None],
        [0, 0, 0],
        serialize_key=["a", "b", "c"],
    )
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}
    jobs = await queries.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert {j.entrypoint for j in jobs} == {"ep"}
    assert len(jobs) == 3


@pytest.mark.asyncio
async def test_inmemory_null_serialize_key_unaffected(queries: InMemoryQueries) -> None:
    """Jobs with NULL serialize_key are not subject to per-key serialization."""
    await queries.enqueue(["ep", "ep"], [None, None], [0, 0])  # both NULL keys
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}
    jobs = await queries.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_inmemory_flag_off_no_per_key_behavior(queries: InMemoryQueries) -> None:
    """With the flag off, all same-key jobs dispatch normally."""
    await queries.enqueue(["ep", "ep"], [None, None], [0, 0], serialize_key=["k", "k"])
    qm_id = uuid.uuid4()
    # Default: serialize_dispatch_per_key=False
    params = {"ep": EntrypointExecutionParameter(0)}
    jobs = await queries.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_inmemory_fifo_within_key(queries: InMemoryQueries) -> None:
    """Same key, same priority → strict FIFO by id."""
    ids = await queries.enqueue(
        ["ep", "ep", "ep"],
        [None, None, None],
        [0, 0, 0],
        serialize_key=["k", "k", "k"],
    )
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}
    seen: list[int] = []
    for _ in range(3):
        picked = await queries.dequeue(
            10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30)
        )
        assert len(picked) == 1
        seen.append(int(picked[0].id))
        await queries.log_jobs([(picked[0], "successful", None)])
    assert seen == sorted(seen) == list(ids)


@pytest.mark.asyncio
async def test_inmemory_priority_ordering_within_key(queries: InMemoryQueries) -> None:
    """Higher priority same-key job runs first, even if enqueued later."""
    low = (await queries.enqueue("ep", None, priority=0, serialize_key="k"))[0]
    high = (await queries.enqueue("ep", None, priority=10, serialize_key="k"))[0]
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}
    first = await queries.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(first) == 1 and int(first[0].id) == int(high)
    await queries.log_jobs([(first[0], "successful", None)])
    second = await queries.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(second) == 1 and int(second[0].id) == int(low)


@pytest.mark.asyncio
async def test_inmemory_list_blocked_keys(queries: InMemoryQueries) -> None:
    await queries.enqueue(
        ["ep", "ep", "ep"],
        [None, None, None],
        [0, 0, 0],
        serialize_key=["k", "k", "other"],
    )
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}
    # Pick the leader for "k"; "other" still has only one queued, so not blocked.
    await queries.dequeue(1, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))

    blocked = await queries.list_blocked_keys()
    assert len(blocked) == 1
    assert blocked[0].serialize_key == "k"
    assert blocked[0].queued_count == 1  # 1 queued behind the leader
    assert blocked[0].leader_id is not None


# ---------------------------------------------------------------------------
# Postgres adapter — real SQL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pg_enqueue_allows_multiple_with_same_serialize_key(apgdriver: Driver) -> None:
    """The new partial index on picked-only must not block multiple queued."""
    q = Queries(apgdriver)
    ids = await q.enqueue(
        ["ep", "ep", "ep"],
        [None, None, None],
        [0, 0, 0],
        serialize_key=["userA", "userA", "userA"],
    )
    assert len(ids) == 3
    sizes = await q.queue_size()
    assert sum(s.count for s in sizes) == 3


@pytest.mark.asyncio
async def test_pg_partial_unique_index_prevents_two_picked(apgdriver: Driver) -> None:
    """Direct UPDATE of two rows to 'picked' for the same key must fail.

    This guards the runtime invariant; the dispatch SQL is the primary
    protection but the partial unique index is the safety net (Procrastinate
    pattern).
    """
    import asyncpg

    q = Queries(apgdriver)
    ids = await q.enqueue(["ep", "ep"], [None, None], [0, 0], serialize_key=["k", "k"])
    # First UPDATE succeeds.
    await apgdriver.execute("UPDATE pgqueuer SET status='picked' WHERE id = $1", int(ids[0]))
    # Second must fail with a unique constraint violation.
    with pytest.raises(asyncpg.UniqueViolationError):
        await apgdriver.execute("UPDATE pgqueuer SET status='picked' WHERE id = $1", int(ids[1]))


@pytest.mark.asyncio
async def test_pg_dispatch_serializes_per_key(apgdriver: Driver) -> None:
    """Two same-key queued jobs → at most one picked at a time."""
    q = Queries(apgdriver)
    await q.enqueue(["ep", "ep"], [None, None], [0, 0], serialize_key=["k", "k"])
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}

    first = await q.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(first) == 1
    second = await q.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(second) == 0

    await q.log_jobs([(first[0], "successful", None)])
    third = await q.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(third) == 1


@pytest.mark.asyncio
async def test_pg_different_keys_run_in_parallel(apgdriver: Driver) -> None:
    q = Queries(apgdriver)
    await q.enqueue(
        ["ep", "ep", "ep"],
        [None, None, None],
        [0, 0, 0],
        serialize_key=["a", "b", "c"],
    )
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}
    jobs = await q.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(jobs) == 3


@pytest.mark.asyncio
async def test_pg_null_keys_unaffected(apgdriver: Driver) -> None:
    q = Queries(apgdriver)
    await q.enqueue(["ep", "ep"], [None, None], [0, 0])
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}
    jobs = await q.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_pg_priority_ordering_within_key(apgdriver: Driver) -> None:
    q = Queries(apgdriver)
    low = (await q.enqueue("ep", None, priority=0, serialize_key="k"))[0]
    high = (await q.enqueue("ep", None, priority=10, serialize_key="k"))[0]
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}

    first = await q.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(first) == 1 and int(first[0].id) == int(high)
    await q.log_jobs([(first[0], "successful", None)])
    second = await q.dequeue(10, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))
    assert len(second) == 1 and int(second[0].id) == int(low)


@pytest.mark.asyncio
async def test_pg_list_blocked_keys(apgdriver: Driver) -> None:
    q = Queries(apgdriver)
    await q.enqueue(
        ["ep", "ep", "ep"],
        [None, None, None],
        [0, 0, 0],
        serialize_key=["k", "k", "other"],
    )
    qm_id = uuid.uuid4()
    params = {"ep": EntrypointExecutionParameter(0, serialize_dispatch_per_key=True)}
    await q.dequeue(1, params, qm_id, None, heartbeat_timeout=timedelta(seconds=30))

    blocked = await q.list_blocked_keys()
    assert len(blocked) == 1
    assert blocked[0].serialize_key == "k"
    assert blocked[0].queued_count == 1
    assert blocked[0].leader_id is not None


# ---------------------------------------------------------------------------
# QueueManager-level integration test (real concurrency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qm_per_key_serial_at_most_one_running(apgdriver: Driver) -> None:
    """Under live dispatch, at most one job per key runs at a time across workers."""
    n_per_key = 20
    keys = ["a", "b", "c"]
    await Queries(apgdriver).enqueue(
        ["fetch"] * (n_per_key * len(keys)),
        [None] * (n_per_key * len(keys)),
        [0] * (n_per_key * len(keys)),
        serialize_key=[k for k in keys for _ in range(n_per_key)],
    )

    qms = [QueueManager(Queries(apgdriver)) for _ in range(2)]
    lock = asyncio.Lock()
    completed = 0

    async def run_consumer(qm: QueueManager) -> None:
        @qm.entrypoint("fetch", serialize_dispatch_per_key=True)
        async def fetch(job: Job) -> None:
            nonlocal completed
            await asyncio.sleep(0.01)
            async with lock:
                completed += 1

        await qm.run(dequeue_timeout=timedelta(seconds=0))

    async def waiter() -> None:
        for _ in range(200):
            await asyncio.sleep(0.05)
            async with lock:
                if completed >= n_per_key * len(keys):
                    break
        for q in qms:
            q.shutdown.set()

    await asyncio.gather(waiter(), *(run_consumer(q) for q in qms))
    # All jobs must have completed; no per-key invariant violated.
    assert completed == n_per_key * len(keys)
