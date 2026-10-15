from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
from uuid import uuid4

import async_timeout
import asyncpg

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
            async with async_timeout.timeout(5):
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
