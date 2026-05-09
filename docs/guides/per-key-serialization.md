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
2. **Another *queued* peer with the same key has earlier priority/FIFO order**
   under `(priority DESC, id ASC)` — only the head of the per-key queue is
   eligible.

Rows with `serialize_key IS NULL` bypass the check entirely.

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

Returns one row per `(entrypoint, serialize_key)` with at least one peer
(another queued job, or a running leader). Use it to spot keys that have
stopped making progress.

## Caveats

### Head-of-line blocking

A leader job that gets stuck — long-running, slow handler, terminally failed
with `on_failure="hold"`, or ending in a `non_retryable_error` — wedges every
queued job behind it for that key. PgQueuer does not bypass the leader; this
is the explicit FIFO trade-off.

Mitigations:

- Set `max_time` on `DatabaseRetryEntrypointExecutor` so single attempts have
  a hard ceiling.
- Use `non_retryable_errors` deliberately. A non-retryable terminal failure
  with `on_failure="hold"` parks the leader as `failed` and blocks the key
  until you call `requeue_jobs` or `clear_queue` on the parked id. With
  `on_failure="delete"` (the default), terminal failures clear the row and
  unblock the key.
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

You can use both on the same job — they answer different questions.

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
