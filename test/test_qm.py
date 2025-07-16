import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import async_timeout
import pytest

from pgqueuer import db
from pgqueuer.models import Job, UpdateJobStatus
from pgqueuer.qm import QueueManager
from pgqueuer.queries import Queries
from pgqueuer.types import QueueExecutionMode


async def wait_until_empty_queue(
    q: Queries,
    qms: list[QueueManager],
) -> None:
    while sum(x.count for x in await q.queue_size()) > 0:
        await asyncio.sleep(0.01)

    for qm in qms:
        qm.shutdown.set()


@pytest.mark.parametrize("N", (1, 2, 32))
async def test_job_queuing(
    apgdriver: db.Driver,
    N: int,
) -> None:
    c = QueueManager(apgdriver, resources={"test": "job_queuing"})
    seen = list[int]()

    @c.entrypoint("fetch")
    async def fetch(context: Job) -> None:
        assert context.payload is not None
        assert c.resources["test"] == "job_queuing"
        seen.append(int(context.payload))

    await c.queries.enqueue(
        ["fetch"] * N,
        [f"{n}".encode() for n in range(N)],
        [0] * N,
    )

    await asyncio.gather(
        c.run(),
        wait_until_empty_queue(c.queries, [c]),
    )

    assert seen == list(range(N))


@pytest.mark.parametrize("N", (1, 2, 32))
@pytest.mark.parametrize("concurrency", (1, 2, 3, 4))
async def test_job_fetch(
    apgdriver: db.Driver,
    N: int,
    concurrency: int,
) -> None:
    q = Queries(apgdriver)
    qmpool = [QueueManager(apgdriver, resources={"test": "job_fetch"}) for _ in range(concurrency)]
    seen = list[int]()

    for qm in qmpool:

        @qm.entrypoint("fetch")
        async def fetch(context: Job) -> None:
            assert context.payload is not None
            assert qm.resources["test"] == "job_fetch"
            seen.append(int(context.payload))

    await q.enqueue(
        ["fetch"] * N,
        [f"{n}".encode() for n in range(N)],
        [0] * N,
    )

    await asyncio.gather(
        asyncio.gather(*[qm.run() for qm in qmpool]),
        wait_until_empty_queue(q, qmpool),
    )

    assert sorted(seen) == list(range(N))


@pytest.mark.parametrize("N", (1, 2, 32))
@pytest.mark.parametrize("concurrency", (1, 2, 3, 4))
async def test_sync_entrypoint(
    apgdriver: db.Driver,
    N: int,
    concurrency: int,
) -> None:
    q = Queries(apgdriver)
    qmpool = [
        QueueManager(apgdriver, resources={"test": "sync_entrypoint"}) for _ in range(concurrency)
    ]
    seen = list[int]()

    for qm in qmpool:

        @qm.entrypoint("fetch")
        def fetch(context: Job) -> None:
            time.sleep(1)  # Sim. heavy CPU/IO.
            assert context.payload is not None
            assert qm.resources["test"] == "sync_entrypoint"
            seen.append(int(context.payload))

    await q.enqueue(
        ["fetch"] * N,
        [f"{n}".encode() for n in range(N)],
        [0] * N,
    )

    await asyncio.gather(
        asyncio.gather(*[qm.run() for qm in qmpool]),
        wait_until_empty_queue(q, qmpool),
    )
    assert sorted(seen) == list(range(N))


async def test_pick_local_entrypoints(
    apgdriver: db.Driver,
    N: int = 100,
) -> None:
    q = Queries(apgdriver)
    qm = QueueManager(apgdriver, resources={"test": "pick_local"})
    pikced_by = list[str]()

    @qm.entrypoint("to_be_picked")
    async def to_be_picked(job: Job) -> None:
        pikced_by.append(job.entrypoint)
        assert qm.resources["test"] == "pick_local"

    await q.enqueue(["to_be_picked"] * N, [None] * N, [0] * N)
    await q.enqueue(["not_picked"] * N, [None] * N, [0] * N)

    async def waiter() -> None:
        while sum(x.count for x in await q.queue_size() if x.entrypoint == "to_be_picked"):
            await asyncio.sleep(0.01)
        qm.shutdown.set()

    await asyncio.gather(
        qm.run(dequeue_timeout=timedelta(seconds=0.01)),
        waiter(),
    )

    assert pikced_by == ["to_be_picked"] * N
    assert sum(s.count for s in await q.queue_size() if s.entrypoint == "to_be_picked") == 0
    assert sum(s.count for s in await q.queue_size() if s.entrypoint == "not_picked") == N


async def test_pick_set_queue_manager_id(
    apgdriver: db.Driver,
    N: int = 100,
) -> None:
    q = Queries(apgdriver)
    qm = QueueManager(apgdriver, resources={"test": "pick_qm_id"})
    qmids = set[uuid.UUID]()

    @qm.entrypoint("fetch")
    async def fetch(job: Job) -> None:
        assert job.queue_manager_id is not None
        assert qm.resources["test"] == "pick_qm_id"
        qmids.add(job.queue_manager_id)

    await q.enqueue(["fetch"] * N, [None] * N, [0] * N)

    async def waiter() -> None:
        while sum(x.count for x in await q.queue_size()):
            await asyncio.sleep(0.01)
        qm.shutdown.set()

    await asyncio.gather(
        qm.run(dequeue_timeout=timedelta(seconds=0.01)),
        waiter(),
    )

    assert len(qmids) == 1


@pytest.mark.parametrize("N", (1, 10, 100))
async def test_drain_mode(
    apgdriver: db.Driver,
    N: int,
) -> None:
    q = Queries(apgdriver)
    qm = QueueManager(apgdriver)
    jobs = list[Job]()

    @qm.entrypoint("fetch")
    async def fetch(job: Job) -> None:
        jobs.append(job)

    await q.enqueue(["fetch"] * N, [None] * N, [0] * N)

    async with async_timeout.timeout(10):
        await qm.run(mode=QueueExecutionMode.drain)

    assert len(jobs) == N


@pytest.mark.asyncio
async def test_handle_job_status_empty_list() -> None:
    """Test handle_job_status with empty list of events."""
    driver = MagicMock()
    qm = QueueManager(driver)

    # Mock the queries methods
    qm.queries.mark_jobs_as_retryable = AsyncMock()
    qm.queries.log_jobs = AsyncMock()

    # Call with empty list
    await qm.handle_job_status([])

    # Verify both methods were called with empty lists
    qm.queries.mark_jobs_as_retryable.assert_called_once_with([])
    qm.queries.log_jobs.assert_called_once_with([])


@pytest.mark.asyncio
async def test_handle_job_status_only_terminal_jobs() -> None:
    """Test handle_job_status with only terminal (non-retryable) jobs."""
    driver = MagicMock()
    qm = QueueManager(driver)

    # Mock the queries methods
    qm.queries.mark_jobs_as_retryable = AsyncMock()
    qm.queries.log_jobs = AsyncMock()

    # Create terminal job status events
    job_id_1 = uuid.uuid4()
    job_id_2 = uuid.uuid4()
    events = [
        UpdateJobStatus(job_id=job_id_1, status="success", retryable=False),
        UpdateJobStatus(job_id=job_id_2, status="failed", retryable=False),
    ]

    await qm.handle_job_status(events)

    # Verify retryable was called with empty list
    qm.queries.mark_jobs_as_retryable.assert_called_once_with([])

    # Verify terminal jobs were logged
    expected_terminal = [(job_id_1, "success"), (job_id_2, "failed")]
    qm.queries.log_jobs.assert_called_once_with(expected_terminal)


@pytest.mark.asyncio
async def test_handle_job_status_only_retryable_jobs() -> None:
    """Test handle_job_status with only retryable jobs."""
    driver = MagicMock()
    qm = QueueManager(driver)

    # Mock the queries methods
    qm.queries.mark_jobs_as_retryable = AsyncMock()
    qm.queries.log_jobs = AsyncMock()

    # Create retryable job status events
    job_id_1 = uuid.uuid4()
    job_id_2 = uuid.uuid4()
    reschedule_time = datetime.now(timezone.utc) + timedelta(minutes=5)

    events = [
        UpdateJobStatus(
            job_id=job_id_1, status="failed", retryable=True, reschedule_for=reschedule_time
        ),
        UpdateJobStatus(job_id=job_id_2, status="failed", retryable=True, reschedule_for=None),
    ]

    await qm.handle_job_status(events)

    # Verify retryable jobs were processed
    expected_retryable = [(job_id_1, "failed", reschedule_time), (job_id_2, "failed", None)]
    qm.queries.mark_jobs_as_retryable.assert_called_once_with(expected_retryable)

    # Verify log_jobs was called with empty list
    qm.queries.log_jobs.assert_called_once_with([])


@pytest.mark.asyncio
async def test_handle_job_status_mixed_jobs() -> None:
    """Test handle_job_status with both terminal and retryable jobs."""
    driver = MagicMock()
    qm = QueueManager(driver)

    # Mock the queries methods
    qm.queries.mark_jobs_as_retryable = AsyncMock()
    qm.queries.log_jobs = AsyncMock()

    # Create mixed job status events
    terminal_job_id = uuid.uuid4()
    retryable_job_id = uuid.uuid4()
    reschedule_time = datetime.now(timezone.utc) + timedelta(minutes=10)

    events = [
        UpdateJobStatus(job_id=terminal_job_id, status="success", retryable=False),
        UpdateJobStatus(
            job_id=retryable_job_id, status="failed", retryable=True, reschedule_for=reschedule_time
        ),
    ]

    await qm.handle_job_status(events)

    # Verify retryable jobs were processed
    expected_retryable = [(retryable_job_id, "failed", reschedule_time)]
    qm.queries.mark_jobs_as_retryable.assert_called_once_with(expected_retryable)

    # Verify terminal jobs were logged
    expected_terminal = [(terminal_job_id, "success")]
    qm.queries.log_jobs.assert_called_once_with(expected_terminal)


@pytest.mark.asyncio
async def test_handle_job_status_query_failures() -> None:
    """Test handle_job_status behavior when query methods fail."""
    driver = MagicMock()
    qm = QueueManager(driver)

    # Mock the queries methods to raise exceptions
    qm.queries.mark_jobs_as_retryable = AsyncMock(side_effect=Exception("Retryable query failed"))
    qm.queries.log_jobs = AsyncMock()

    # Create events
    job_id = uuid.uuid4()
    events = [
        UpdateJobStatus(job_id=job_id, status="failed", retryable=True),
    ]

    # Should raise exception due to asyncio.gather behavior
    with pytest.raises(Exception, match="Retryable query failed"):
        await qm.handle_job_status(events)

    # Verify both methods were called (due to asyncio.gather)
    qm.queries.mark_jobs_as_retryable.assert_called_once()
    qm.queries.log_jobs.assert_called_once()


@pytest.mark.asyncio
async def test_handle_job_status_logging(caplog) -> None:
    """Test that handle_job_status logs the correct debug message."""
    driver = MagicMock()
    qm = QueueManager(driver)

    # Mock the queries methods
    qm.queries.mark_jobs_as_retryable = AsyncMock()
    qm.queries.log_jobs = AsyncMock()

    # Create events
    events = [
        UpdateJobStatus(job_id=uuid.uuid4(), status="success", retryable=False),
        UpdateJobStatus(job_id=uuid.uuid4(), status="failed", retryable=True),
    ]

    with caplog.at_level("DEBUG"):
        await qm.handle_job_status(events)

    # Check that debug message was logged
    assert "Handling 2 job updates" in caplog.text


@pytest.mark.asyncio
async def test_handle_job_status_various_statuses() -> None:
    """Test handle_job_status with various job statuses."""
    driver = MagicMock()
    qm = QueueManager(driver)

    # Mock the queries methods
    qm.queries.mark_jobs_as_retryable = AsyncMock()
    qm.queries.log_jobs = AsyncMock()

    # Create events with different statuses
    job_ids = [uuid.uuid4() for _ in range(4)]
    events = [
        UpdateJobStatus(job_id=job_ids[0], status="success", retryable=False),
        UpdateJobStatus(job_id=job_ids[1], status="failed", retryable=False),
        UpdateJobStatus(job_id=job_ids[2], status="failed", retryable=True),
        UpdateJobStatus(job_id=job_ids[3], status="cancelled", retryable=False),
    ]

    await qm.handle_job_status(events)

    # Verify retryable jobs
    expected_retryable = [(job_ids[2], "failed", None)]
    qm.queries.mark_jobs_as_retryable.assert_called_once_with(expected_retryable)

    # Verify terminal jobs
    expected_terminal = [
        (job_ids[0], "success"),
        (job_ids[1], "failed"),
        (job_ids[3], "cancelled"),
    ]
    qm.queries.log_jobs.assert_called_once_with(expected_terminal)
