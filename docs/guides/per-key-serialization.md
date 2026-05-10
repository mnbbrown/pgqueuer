# Per-Key Dispatch Serialization

A common pattern is "at most one job per resource X running at a time, but
different X's run in parallel" — per-user state machines, per-tenant batch
jobs, per-mailbox sync, per-conversation LLM jobs, per-aggregate domain
commands. The shape mirrors AWS SQS FIFO `MessageGroupId`, GCP Pub/Sub
`ordering_key`, Kafka partition-by-key, and RabbitMQ Single Active Consumer.

PgQueuer implements this via a per-job `serialize_key` and a per-entrypoint
`serialize_dispatch_per_key` flag.

## Quick start

```python
from pgqueuer import PgQueuer
from pgqueuer.models import Job

pgq = PgQueuer.from_dsn(...)

@pgq.entrypoint("drain_inbox", serialize_dispatch_per_key=True)
async def drain_inbox(job: Job) -> None:
    ...

# Producer side: pin work to a resource
await pgq.queries.enqueue("drain_inbox", payload, serialize_key=f"user:{user_id}")
```

Now jobs sharing a `serialize_key` for `drain_inbox` are dispatched
**at most one-at-a-time, in priority then FIFO order**. Jobs with different
keys (or no key) run in parallel.

## Semantics

When `serialize_dispatch_per_key=True` on an entrypoint, the dispatch query
filters out any candidate row whose `(entrypoint, serialize_key)` is "blocked":

1. **Another job with the same key is already `picked`** — at-most-one-running
   per key is enforced. A partial unique index acts as a safety net.
2. **Another eligible *queued* peer with the same key has earlier
   priority/FIFO order** under `(priority DESC, id ASC)` — only the head of
   the currently runnable per-key queue is eligible.

Rows with `serialize_key IS NULL` bypass the check entirely. Jobs that are not
currently runnable, such as future `execute_after` rows or jobs parked as
`failed`, do not block the key.

### What counts as "earlier" (priority-aware FIFO)

- A higher-priority queued peer wins, regardless of insertion order.
- Among same-priority peers, the lower-id (older) row wins.

This mirrors Procrastinate's
[`procrastinate_fetch_job_v2`](https://github.com/procrastinate-org/procrastinate/pull/1411).

### Multiple queued, one running

Unlike `dedupe_key` (which is *unique* across `queued`+`picked`), multiple
jobs may be queued simultaneously with the same `serialize_key`. They are
dispatched one-at-a-time as the leader completes. This is the SQS-FIFO
`MessageGroupId` shape.

## Crash recovery

If a worker dies mid-job, the existing heartbeat-timeout machinery requeues
the stale `picked` row (status flips back to `queued`). When that happens, the
row leaves the partial index that gates the per-key check, so the key becomes
unblocked for any worker to pick up. No special recovery path is required.

## Diagnostics — `list_blocked_keys()`

```python
blocked = await pgq.queries.list_blocked_keys()
for b in blocked:
    print(
        f"{b.entrypoint}/{b.serialize_key}: {b.queued_count} queued, "
        f"leader id={b.leader_id} age={b.leader_age_seconds:.1f}s "
        f"heartbeat_age={b.leader_heartbeat_age_seconds:.1f}s"
    )
```

Returns one row per `(entrypoint, serialize_key)` with eligible queued work
blocked by another eligible queued job or by a running leader. Use it to spot
keys that have stopped making progress.

## Caveats

### Head-of-line blocking

A leader job that gets stuck — long-running, slow handler, or crashed worker
whose heartbeat has not timed out yet — wedges every eligible queued job behind
it for that key. PgQueuer does not bypass a running leader; this is the
explicit FIFO trade-off.

Mitigations:

- Set `max_time` on `DatabaseRetryEntrypointExecutor` so single attempts have
  a hard ceiling.
- Use `on_failure="hold"` deliberately. A terminal failure parked as `failed`
  leaves the active serialization lane, so later same-key jobs may continue.
  Requeueing the failed job later can therefore process it after some followers
  have already completed.
- Watch `list_blocked_keys()` — `oldest_queued_age_seconds` and
  `leader_heartbeat_age_seconds` together signal stuck keys before they
  cause user-visible problems.

### `serialize_key` vs. `dedupe_key`

| | `dedupe_key` | `serialize_key` |
|---|---|---|
| Purpose | Collapse concurrent enqueues | Serialize concurrent dispatch |
| Constraint | Unique across `queued`+`picked` | At-most-one in `picked` only |
| N queued for same key? | No (`DuplicateJobError`) | Yes |
| Default behavior | Always enforced | Opt-in per entrypoint |
| Use it for... | Idempotency at the API edge | Per-resource ordering |

You can use both on the same job — they answer different questions:

- `serialize_key` controls **concurrency**.
- `dedupe_key` controls **accumulation**.

#### Serialize every event for a resource

Use only `serialize_key` when every job matters, but same-resource jobs must
not run concurrently:

```python
await pgq.queries.enqueue(
    "sync_customer_event",
    {"customer_id": customer_id, "event_id": event_id},
    serialize_key=f"customer:{customer_id}",
)
```

All events are retained. Different customers can run in parallel, while one
customer's jobs are dispatched one-at-a-time.

#### Coalesce repeated work for a resource

Use both keys when repeated enqueue attempts should collapse while pending, but
the active job should still exclude same-resource followers:

```python
key = f"customer:{customer_id}"

await pgq.queries.enqueue(
    "sync_customer",
    {"customer_id": customer_id},
    dedupe_key=f"sync_customer:{key}",
    serialize_key=key,
)
```

This keeps at most one queued or picked `sync_customer` job per customer. Once
the active job finishes, a new enqueue with the same `dedupe_key` can create a
fresh follow-up job to capture newer changes.

#### Coalesce only queued follow-up work

Use `enqueue_if_no_queued()` when an already-running job should not suppress a
fresh follow-up. PgQueuer checks only for a queued job with the same
`(entrypoint, serialize_key)` and deliberately ignores picked jobs:

```python
key = f"customer:{customer_id}"

job_id = await pgq.queries.enqueue_if_no_queued(
    "sync_customer",
    {"customer_id": customer_id},
    serialize_key=key,
)
```

This usually keeps one queued follow-up per resource without suppressing work
behind a picked leader. Under concurrent producer races, more than one queued
follow-up can be inserted because this helper does not add a database-level
uniqueness constraint.

#### Dedupe narrower than serialization

Sometimes the serialization lane is broad, but duplicate detection is narrower:

```python
await pgq.queries.enqueue(
    "send_customer_notification",
    {"customer_id": customer_id, "notification_id": notification_id},
    serialize_key=f"customer:{customer_id}",
    dedupe_key=f"notification:{notification_id}",
)
```

Notifications for the same customer do not send concurrently, but distinct
notifications are still preserved. Only a duplicate notification enqueue is
rejected.

Treat `dedupe_key` values as application-level identifiers: prefix them with
the workflow name so unrelated entrypoints do not accidentally share a key.

### Performance

The dispatch query has two extra index lookups per candidate row when the flag
is enabled. Indexes used:

- `pgqueuer_picked_serialize_key_idx` — partial unique on
  `(entrypoint, serialize_key) WHERE status = 'picked'`. Bounded by your max
  concurrent picked count, so always small.
- `pgqueuer_queued_serialize_key_idx` — partial on
  `(entrypoint, serialize_key, priority DESC, id ASC) WHERE status = 'queued'`.
  Powers the leader-lookup.

Both are required. Run `pgq upgrade` to add them on existing installations.

Empirically: dispatch latency stays within ~30% of the no-feature baseline up
to 100k queue depth at 1k+ unique keys. Workloads with very high duplication
per key (≥1000 jobs per key with 1M+ depth) can push p99 dispatch latency
into the tens of milliseconds — but at that point head-of-line blocking is
likely the bigger operational problem.

## Comparison to other patterns

- **Bucketed entrypoints** (e.g. `entrypoint = f"drain_{hash(key) % N}"` with
  `concurrency_limit=1`) — works for low-cardinality keys, has fixed bucket
  cardinality, suffers head-of-line blocking *across* keys that happen to
  share a bucket.
- **Application-level locks** (Redis `SET NX`, advisory locks) — adds an
  external dependency or hidden coordination state. PgQueuer's primitive is
  visible in queries and durable across restarts.
- **Dedupe-key with retry-loop in the handler** — works only when the work
  is idempotent; doesn't preserve order; pollutes the application code with
  coordination logic.
