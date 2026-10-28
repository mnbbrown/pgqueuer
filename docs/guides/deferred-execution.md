# Deferred Execution

The `execute_after` attribute controls when a job becomes eligible for processing.

## How it works

`execute_after` specifies the **earliest** time a job can be picked for execution. If not
provided, the job is eligible immediately (`NOW()`).

This is useful for:

- Delaying execution to allow external dependencies to settle
- Implementing retry-after-cooldown patterns
- Scheduling one-off future tasks without a full cron setup

## Example

Enqueue a job that should execute one minute from now:

```python
from datetime import timedelta
from pgqueuer.queries import Queries

await Queries(driver).enqueue(
    "my_task",
    payload=None,
    priority=0,
    execute_after=timedelta(minutes=1),
)
```

Jobs remain in the queue with `queued` status until their `execute_after` timestamp is
reached. The `QueueManager` uses `FOR UPDATE SKIP LOCKED` with a timestamp filter, so
deferred jobs are never picked early.

!!! note
    `execute_after` only accepts a `timedelta` offset from the current time (or `None` for
    immediate execution). Absolute `datetime` values are **not** supported.

## Combining with priority

Deferred jobs participate in the normal priority queue once their `execute_after` time passes:

```python
# High priority but deferred: will jump the queue once eligible
await queries.enqueue("urgent_task", b"data", priority=10, execute_after=timedelta(hours=1))
```

## Worker wakeup queries

After an empty dequeue, built-in repositories read the earliest queued timestamp
for each registered entrypoint through the `(entrypoint, execute_after)` partial
index. The minimum of those timestamps covers both ready and future work in one
snapshot. Each entrypoint contributes at most one candidate, regardless of its
future backlog. This query does not claim jobs or change their priority.

The worker subtracts probe duration from the returned ETA, retains the minimum
retry delay for ready work,
and caps its sleep at the configured fallback interval. Notifications received
during a query stay pending and interrupt the wait.

`QueueWakeupPort.next_queued_eta` is optional. Custom repositories without it
retain the existing readiness and deferred-ETA probes. The existing
`next_deferred_eta` method still returns future work only. No schema migration
is required when the entrypoint/time index is already installed.
