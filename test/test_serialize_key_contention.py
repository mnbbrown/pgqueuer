from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
import pytest

from pgqueuer.adapters.inmemory import InMemoryQueries
from pgqueuer.db import AsyncpgDriver
from pgqueuer.domain.types import QueueEntrypoint, QueueManagerId
from pgqueuer.models import Job
from pgqueuer.queries import EntrypointExecutionParameter, Queries


async def test_lost_key_claim_leaves_worker_usable(dsn: str) -> None:
    async with AsyncExitStack() as stack:
        first = await asyncpg.connect(dsn=dsn)
        stack.push_async_callback(first.close)
        second = await asyncpg.connect(dsn=dsn)
        stack.push_async_callback(second.close)
        winner = Queries(AsyncpgDriver(first))
        loser = Queries(AsyncpgDriver(second))
        queued = await winner.enqueue(entrypoint="ep", payload=None, serialize_key="key")

        async def claim(queries: Queries) -> list[Job]:
            return await queries.dequeue(
                batch_size=1,
                entrypoints={
                    QueueEntrypoint("ep"): EntrypointExecutionParameter(
                        0, serialize_dispatch_per_key=True
                    )
                },
                queue_manager_id=QueueManagerId(uuid4()),
                global_concurrency_limit=None,
                heartbeat_timeout=timedelta(seconds=600),
            )

        async with first.transaction():
            await winner.enqueue(
                entrypoint="ep",
                payload=None,
                priority=10,
                serialize_key="key",
                execute_after=timedelta(seconds=-1),
            )
            picked = await claim(queries=winner)
            assert len(picked) == 1
            assert picked[0].id != queued[0]
            competing = asyncio.create_task(claim(queries=loser))
            # The second snapshot cannot see the first worker's uncommitted job.
            async with asyncio.timeout(5):
                while not await first.fetchval(
                    "SELECT $1 = ANY(pg_blocking_pids($2))",
                    first.get_server_pid(),
                    second.get_server_pid(),
                ):
                    await asyncio.sleep(0.01)

        assert await competing == []
        assert await first.fetchval("SELECT count(*) FROM pgqueuer WHERE status='picked'") == 1
        await winner.log_jobs(job_status=[(picked[0], "successful", None)])
        next_jobs = await claim(queries=loser)
        assert [job.id for job in next_jobs] == queued


@pytest.mark.parametrize("priority", [0, 10])
async def test_stale_key_owner_recovers_before_queued_sibling(dsn: str, priority: int) -> None:
    async with AsyncExitStack() as stack:
        connection = await asyncpg.connect(dsn=dsn)
        stack.push_async_callback(connection.close)
        queries = Queries(AsyncpgDriver(connection))
        waiting = await queries.enqueue(
            entrypoint="ep",
            payload=None,
            serialize_key="key",
            priority=priority,
            execute_after=timedelta(hours=1),
        )
        running = await queries.enqueue(entrypoint="ep", payload=None, serialize_key="key")

        async def claim() -> list[Job]:
            return await queries.dequeue(
                batch_size=10,
                entrypoints={QueueEntrypoint("ep"): EntrypointExecutionParameter(1)},
                queue_manager_id=QueueManagerId(uuid4()),
                global_concurrency_limit=None,
                heartbeat_timeout=timedelta(seconds=30),
            )

        assert [job.id for job in await claim()] == running
        await connection.execute(
            "UPDATE pgqueuer SET execute_after = NOW() - INTERVAL '1 second' WHERE id = $1",
            waiting[0],
        )
        assert await claim() == []
        await connection.execute(
            "UPDATE pgqueuer SET heartbeat = NOW() - INTERVAL '1 minute' WHERE id = $1",
            running[0],
        )
        recovered = await claim()
        assert [job.id for job in recovered] == running
        assert await claim() == []
        await queries.log_jobs(job_status=[(recovered[0], "successful", None)])
        assert [job.id for job in await claim()] == waiting


async def test_inmemory_stale_key_owner_recovers_before_queued_sibling(
    queries: InMemoryQueries,
) -> None:
    running = await queries.enqueue(entrypoint="ep", payload=None, serialize_key="key")

    async def claim() -> list[Job]:
        return await queries.dequeue(
            batch_size=10,
            entrypoints={QueueEntrypoint("ep"): EntrypointExecutionParameter(1)},
            queue_manager_id=QueueManagerId(uuid4()),
            global_concurrency_limit=None,
            heartbeat_timeout=timedelta(seconds=30),
        )

    assert [job.id for job in await claim()] == running
    waiting = await queries.enqueue(
        entrypoint="ep",
        payload=None,
        serialize_key="key",
        priority=10,
    )
    assert await claim() == []
    queries.jobs[running[0]].heartbeat = datetime.now(UTC) - timedelta(minutes=1)
    recovered = await claim()
    assert [job.id for job in recovered] == running
    assert await claim() == []
    await queries.log_jobs(job_status=[(recovered[0], "successful", None)])
    assert [job.id for job in await claim()] == waiting
