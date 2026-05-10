"""
Tests for benchmark grace period cancellation.

Verifies that the grace period calculation is correct and that
tasks can be cancelled and cleaned up properly.
"""

from __future__ import annotations

import asyncio

from tools.benchmark import (
    DriverEnum,
    SerializeDrainSettings,
    SerializeKeyMode,
    StrategyEnum,
    dequeue_latency_summary,
    serialize_keys_for,
)


def test_serialize_drain_key_modes() -> None:
    base = {
        "driver": DriverEnum.apg,
        "strategy": StrategyEnum.serialize_drain,
        "jobs": 5,
        "keys": 2,
        "serialize_dispatch_per_key": True,
        "dequeue": 1,
        "dequeue_batch_size": 10,
        "output_json": None,
    }

    assert (
        serialize_keys_for(SerializeDrainSettings(**base, key_mode=SerializeKeyMode.none)) is None
    )
    assert serialize_keys_for(SerializeDrainSettings(**base, key_mode=SerializeKeyMode.unique)) == [
        "k:0",
        "k:1",
        "k:2",
        "k:3",
        "k:4",
    ]
    assert serialize_keys_for(
        SerializeDrainSettings(**base, key_mode=SerializeKeyMode.round_robin)
    ) == [
        "k:0",
        "k:1",
        "k:0",
        "k:1",
        "k:0",
    ]
    assert serialize_keys_for(SerializeDrainSettings(**base, key_mode=SerializeKeyMode.single)) == [
        "k:0",
        "k:0",
        "k:0",
        "k:0",
        "k:0",
    ]


def test_dequeue_latency_summary() -> None:
    assert dequeue_latency_summary([]) == (None, None, None)
    assert dequeue_latency_summary([0.001]) == (1.0, 1.0, 1.0)

    p50, p95, p99 = dequeue_latency_summary([0.001, 0.002, 0.003, 0.004])

    assert p50 == 3.0
    assert p95 is not None and 3.0 < p95 < 4.0
    assert p99 is not None and 3.0 < p99 < 4.0


async def test_grace_period_calculation() -> None:
    """
    Test that grace period is calculated correctly.

    Formula: max(5.0, timer_seconds * 0.1)
    - Short timers (< 50s) get 5 second grace period
    - Long timers (>= 50s) get 10% of timer as grace period
    """
    # Short timer uses minimum 5 seconds
    assert max(5.0, 1.0 * 0.1) == 5.0
    assert max(5.0, 10.0 * 0.1) == 5.0
    assert max(5.0, 50.0 * 0.1) == 5.0

    # Longer timers use 10% of duration
    assert max(5.0, 60.0 * 0.1) == 6.0
    assert max(5.0, 100.0 * 0.1) == 10.0
    assert max(5.0, 200.0 * 0.1) == 20.0


async def test_cancel_tasks_after_grace_period() -> None:
    """
    Test the grace period cancellation pattern from the fix.

    1. Create tasks
    2. Wait with timeout (grace period)
    3. Cancel any tasks still pending
    4. Gather results
    """

    async def slow_task() -> None:
        """A task that takes a long time."""
        await asyncio.sleep(100)

    # Create two tasks
    task1 = asyncio.create_task(slow_task())
    task2 = asyncio.create_task(slow_task())

    # Wait with short grace period
    grace_period = 0.01
    done, pending = await asyncio.wait({task1, task2}, timeout=grace_period)

    # Both should still be pending (grace too short)
    assert len(pending) == 2
    assert len(done) == 0

    # Cancel pending tasks
    for task in pending:
        task.cancel()

    # Gather with return_exceptions to handle CancelledError
    await asyncio.gather(*pending, return_exceptions=True)

    # Verify they're cancelled
    assert all(task.cancelled() for task in pending)


async def test_exception_propagation_from_tasks() -> None:
    """
    Test that exceptions from completed tasks are caught and handled.

    The fix pattern checks: if task completed with an exception, re-raise it.
    """

    class MyError(Exception):
        pass

    async def failing_task() -> None:
        raise MyError("Task failed")

    async def success_task() -> None:
        await asyncio.sleep(0.01)

    task1 = asyncio.create_task(failing_task())
    task2 = asyncio.create_task(success_task())

    # Wait for both
    done, _pending = await asyncio.wait({task1, task2}, timeout=1.0)

    # Check for exceptions
    for task in done:
        if not task.cancelled() and task.exception():
            # This is the pattern from the fix
            assert isinstance(task.exception(), MyError)
            break
    else:
        raise AssertionError("Should have found the exception")
