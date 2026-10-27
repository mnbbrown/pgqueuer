from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from pgqueuer.adapters.drivers.asyncpg import AsyncpgDriver
from pgqueuer.adapters.inmemory import InMemoryQueries
from pgqueuer.adapters.persistence.queries import Queries
from pgqueuer.core.buffers import HeartbeatBuffer
from pgqueuer.domain.types import QueueEntrypoint, QueueManagerId
from pgqueuer.ports.repository import EntrypointExecutionParameter


@pytest.mark.parametrize("backend", ["postgres", "memory"])
async def test_stale_worker_heartbeat_does_not_renew_replacement_lease(
    apgdriver: AsyncpgDriver, queries: InMemoryQueries, backend: str
) -> None:
    repository = Queries(apgdriver) if backend == "postgres" else queries
    owner_a, owner_b, owner_c = [QueueManagerId(uuid.uuid4()) for _ in range(3)]
    entrypoints = {QueueEntrypoint("work"): EntrypointExecutionParameter(1)}
    ids = await repository.enqueue(entrypoint="work", payload=None)

    async def claim(owner: QueueManagerId) -> None:
        jobs = await repository.dequeue(
            batch_size=1,
            entrypoints=entrypoints,
            queue_manager_id=owner,
            global_concurrency_limit=None,
            heartbeat_timeout=timedelta(seconds=30),
        )
        assert [job.id for job in jobs] == ids

    async def expire() -> datetime:
        expired = datetime.now(UTC) - timedelta(minutes=1)
        if isinstance(repository, InMemoryQueries):
            repository.jobs[int(ids[0])].heartbeat = expired
        else:
            await apgdriver.execute("UPDATE pgqueuer SET heartbeat = $1", expired)
        return expired

    async def heartbeat_time() -> object:
        if isinstance(repository, InMemoryQueries):
            return repository.jobs[int(ids[0])].heartbeat
        return (await apgdriver.fetch("SELECT heartbeat FROM pgqueuer"))[0]["heartbeat"]

    await claim(owner_a)
    stale_buffer = HeartbeatBuffer(max_size=100, repository=repository, queue_manager_id=owner_a)
    await stale_buffer.add(ids[0])
    await expire()
    await claim(owner_b)
    expired = await expire()  # B stops heartbeating after taking over.
    await stale_buffer.flush()  # A's delayed batch must not renew B's lease.
    assert await heartbeat_time() == expired
    await claim(owner_c)

    current_buffer = HeartbeatBuffer(max_size=100, repository=repository, queue_manager_id=owner_c)
    await current_buffer.add(ids[0])
    await current_buffer.flush()
    assert await heartbeat_time() != expired
