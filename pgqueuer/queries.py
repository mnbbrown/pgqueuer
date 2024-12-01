"""
Database query builder and executor for job queue operations.

This module provides classes and functions to construct and execute SQL queries
related to job queuing, such as installing the necessary database schema,
enqueueing and dequeueing jobs, logging job statuses, and managing job statistics.
It abstracts the SQL details and offers a high-level interface for interacting
with the database in the context of the pgqueuer application.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from datetime import timedelta
from typing import overload

from . import db, helpers, models, qb


@dataclasses.dataclass
class EntrypointExecutionParameter:
    """
    Job execution parameters like retry, concurrency.

    Attributes:
        retry_after (timedelta): Time to wait before retrying.
        serialized (bool): Whether execution is serialized.
        concurrency_limit (int): Max number of concurrent executions.
    """

    retry_after: timedelta
    serialized: bool
    concurrency_limit: int


@dataclasses.dataclass
class Queries:
    """
    High-level interface for executing job queue operations.

    This class provides methods to perform actions on the job queue and statistics
    tables, such as installing or uninstalling the schema, enqueueing and dequeuing jobs,
    logging job statuses, clearing the queue or logs, and retrieving statistics.
    It utilizes the SQL queries generated by `QueryBuilder` and executes them using
    the provided database driver.

    Attributes:
        driver (db.Driver): The database driver used to execute SQL commands.
        qb (QueryBuilder): An instance of `QueryBuilder` to generate SQL queries.
    """

    driver: db.Driver

    qbe: qb.QueryBuilderEnvironment = dataclasses.field(
        default_factory=qb.QueryBuilderEnvironment,
    )
    qbq: qb.QueryQueueBuilder = dataclasses.field(
        default_factory=qb.QueryQueueBuilder,
    )
    qbs: qb.QuerySchedulerBuilder = dataclasses.field(
        default_factory=qb.QuerySchedulerBuilder,
    )

    async def install(self) -> None:
        """
        Install the job queue schema in the database.

        Executes the SQL commands generated by `create_install_query` to set up
        the necessary tables, types, indexes, triggers, and functions required
        for the job queue system to operate.

        This method should be called during the initial setup of the application.
        """
        await self.driver.execute(self.qbe.create_install_query())

    async def uninstall(self) -> None:
        """
        Uninstall the job queue schema from the database.

        Executes the SQL commands generated by `create_uninstall_query` to remove
        all database objects created during installation. This includes dropping
        tables, types, triggers, and functions.

        Use this method with caution, as it will delete all data and schema
        related to the job queue system.
        """
        await self.driver.execute(self.qbe.create_uninstall_query())

    async def upgrade(self) -> None:
        """
        Upgrade the existing database schema to the latest version.

        Executes the SQL commands generated by `create_upgrade_queries` to modify
        the database schema as needed. This may involve adding columns, indexes,
        or updating functions to support new features.

        This method should be called when updating the application to a new version
        that requires schema changes.
        """
        await self.driver.execute("\n\n".join(self.qbe.create_upgrade_queries()))

    async def table_has_column(self, table: str, column: str) -> bool:
        """
        Check if the column exists in table.

        Returns:
            bool: True if the column exists, False otherwise.
        """
        rows = await self.driver.fetch(
            self.qbe.create_table_has_column_query(),
            table,
            column,
        )
        assert len(rows) == 1
        (row,) = rows
        return row["exists"]

    async def has_user_defined_enum(self, key: str, enum: str) -> bool:
        """Check if a value exists in a user-defined ENUM type."""
        rows = await self.driver.fetch(self.qbe.create_user_types_query())
        return (key, enum) in {(row["enumlabel"], row["typname"]) for row in rows}

    async def has_table(self, table: str) -> bool:
        rows = await self.driver.fetch(
            self.qbe.create_has_table_query(),
            table,
        )
        assert len(rows) == 1
        (row,) = rows
        return row["exists"]

    async def dequeue(
        self,
        batch_size: int,
        entrypoints: dict[str, EntrypointExecutionParameter],
        queue_manager_id: uuid.UUID,
    ) -> list[models.Job]:
        """
        Retrieve and update jobs from the queue to be processed.

        Selects jobs from the queue that match the specified entrypoints and updates
        their status to 'picked'. The selection prioritizes 'queued' jobs but can
        also include 'picked' jobs that have exceeded the retry timer, allowing
        for retries of stalled jobs.

        Args:
            batch_size (int): The maximum number of jobs to retrieve.
            entrypoints (set[str]): A set of entrypoints to filter the jobs.
            retry_timer (timedelta | None): The duration after which 'picked' jobs
                are considered for retry. If None, retry logic is skipped.

        Returns:
            list[models.Job]: A list of Job instances representing the dequeued jobs.

        Raises:
            ValueError: If batch_size is less than 1 or retry_timer is negative.
        """

        if batch_size < 1:
            raise ValueError("Batch size must be greater than or equal to one (1)")

        rows = await self.driver.fetch(
            self.qbq.create_dequeue_query(),
            batch_size,
            list(entrypoints.keys()),
            [x.retry_after for x in entrypoints.values()],
            [x.serialized for x in entrypoints.values()],
            [x.concurrency_limit for x in entrypoints.values()],
            queue_manager_id,
        )
        return [models.Job.model_validate(dict(row)) for row in rows]

    @overload
    async def enqueue(
        self,
        entrypoint: str,
        payload: bytes | None,
        priority: int = 0,
        execute_after: timedelta | None = None,
    ) -> list[models.JobId]: ...

    @overload
    async def enqueue(
        self,
        entrypoint: list[str],
        payload: list[bytes | None],
        priority: list[int],
        execute_after: list[timedelta | None] | None = None,
    ) -> list[models.JobId]: ...

    async def enqueue(
        self,
        entrypoint: str | list[str],
        payload: bytes | None | list[bytes | None],
        priority: int | list[int] = 0,
        execute_after: timedelta | None | list[timedelta | None] = None,
    ) -> list[models.JobId]:
        """
        Insert new jobs into the queue.

        Adds one or more jobs to the queue with specified entrypoints, payloads,
        and priorities. Supports inserting multiple jobs in a single operation
        by accepting lists for the parameters.

        Args:
            entrypoint (str | list[str]): The entrypoint(s) associated with the job(s).
            payload (bytes | None | list[bytes | None]): The payload(s) for the job(s).
            priority (int | list[int]): The priority level(s) for the job(s).

        Returns:
            list[models.JobId]: A list of JobId instances representing the IDs of the enqueued jobs.

        Raises:
            ValueError: If the lengths of the lists provided do not match when using multiple jobs.
        """

        # If they are not lists, create single-item lists for uniform processing
        normed_entrypoint = entrypoint if isinstance(entrypoint, list) else [entrypoint]
        normed_payload = payload if isinstance(payload, list) else [payload]
        normed_priority = priority if isinstance(priority, list) else [priority]

        execute_after = (
            [timedelta(seconds=0)] * len(normed_entrypoint)
            if execute_after is None
            else execute_after
        )

        normed_execute_after = (
            [x or timedelta(seconds=0) for x in execute_after]
            if isinstance(execute_after, list)
            else [execute_after or timedelta(seconds=0)]
        )

        return [
            models.JobId(row["id"])
            for row in await self.driver.fetch(
                self.qbq.create_enqueue_query(),
                normed_priority,
                normed_entrypoint,
                normed_payload,
                normed_execute_after,
            )
        ]

    async def clear_queue(self, entrypoint: str | list[str] | None = None) -> None:
        """
        Remove jobs from the queue, optionally filtered by entrypoints.

        Deletes jobs from the queue table. If entrypoints are provided, only jobs
        matching those entrypoints are removed; otherwise, the entire queue is cleared.

        Args:
            entrypoint (str | list[str] | None): The entrypoint(s) to filter jobs for deletion.
        """
        await (
            self.driver.execute(
                self.qbq.create_delete_from_queue_query(),
                [entrypoint] if isinstance(entrypoint, str) else entrypoint,
            )
            if entrypoint
            else self.driver.execute(self.qbq.create_truncate_queue_query())
        )

    async def mark_job_as_cancelled(self, ids: list[models.JobId]) -> None:
        """
        Mark specific jobs as cancelled and notify the system.

        Moves the specified jobs from the queue table to the statistics table with
        a status of 'canceled' and sends a cancellation event notification.

        Args:
            ids (list[models.JobId]): The IDs of the jobs to cancel.
        """
        await asyncio.gather(
            self.driver.execute(self.qbq.create_log_job_query(), ids, ["canceled"] * len(ids)),
            self.notify_job_cancellation(ids),
        )

    async def queue_size(self) -> list[models.QueueStatistics]:
        """
        Get statistics about the current size of the queue.

        Retrieves the number of jobs in the queue, grouped by entrypoint, priority,
        and status. This provides insight into the workload and helps with monitoring.

        Returns:
            list[models.QueueStatistics]: A list of statistics entries for the queue.
        """
        return [
            models.QueueStatistics.model_validate(dict(x))
            for x in await self.driver.fetch(self.qbq.create_queue_size_query())
        ]

    async def log_jobs(
        self,
        job_status: list[tuple[models.Job, models.STATUS_LOG]],
    ) -> None:
        """
        Move completed or failed jobs from the queue to the log table.

        Processes a list of jobs along with their final statuses, removing them
        from the queue table and recording their details in the statistics table.

        Args:
            job_status (list[tuple[models.Job, models.STATUS_LOG]]): A list of tuples
                containing jobs and their corresponding statuses
                ('successful', 'exception', or 'canceled').
        """
        await self.driver.execute(
            self.qbq.create_log_job_query(),
            [j.id for j, _ in job_status],
            [s for _, s in job_status],
        )

    async def clear_log(self, entrypoint: str | list[str] | None = None) -> None:
        """
        Remove entries from the statistics (log) table.

        Deletes log entries from the statistics table. If entrypoints are provided,
        only entries matching those entrypoints are removed; otherwise, the entire
        log is cleared.

        Args:
            entrypoint (str | list[str] | None): The entrypoint(s) to filter log
                entries for deletion.
        """
        await (
            self.driver.execute(
                self.qbq.create_delete_from_log_query(),
                [entrypoint] if isinstance(entrypoint, str) else entrypoint,
            )
            if entrypoint
            else self.driver.execute(self.qbq.create_truncate_log_query())
        )

    async def log_statistics(
        self,
        tail: int | None,
        last: timedelta | None = None,
    ) -> list[models.LogStatistics]:
        """
        Retrieve job processing statistics from the log.

        Fetches entries from the statistics table, optionally limited by the number
        of recent entries (`tail`) and a time window (`last`). This information
        can be used for monitoring and analysis.

        Args:
            tail (int | None): The maximum number of recent entries to retrieve.
            last (timedelta | None): The time window to consider (e.g., last hour).

        Returns:
            list[models.LogStatistics]: A list of log statistics entries.
        """
        return [
            models.LogStatistics.model_validate(dict(x))
            for x in await self.driver.fetch(
                self.qbq.create_log_statistics_query(),
                tail,
                None if last is None else last.total_seconds(),
            )
        ]

    async def notify_debounce_event(self, entrypoint_count: dict[str, int]) -> None:
        """
        Send a requests-per-second event notification for an entrypoint.

        Emits a 'requests_per_second_event' notification via the PostgreSQL NOTIFY
        system to inform other components about the current request rate for an
        entrypoint. This can be used to adjust processing rates or trigger scaling.

        Args:
            entrypoint (str): The entrypoint for which the event is being sent.
            quantity (int): The number of requests per second to report.
        """
        await self.driver.execute(
            self.qbq.create_notify_query(),
            models.RequestsPerSecondEvent(
                channel=self.qbq.settings.channel,
                entrypoint_count=entrypoint_count,
                sent_at=helpers.utc_now(),
                type="requests_per_second_event",
            ).model_dump_json(),
        )

    async def notify_job_cancellation(self, ids: list[models.JobId]) -> None:
        """
        Send a cancellation event notification for specific job IDs.

        Emits a 'cancellation_event' notification via the PostgreSQL NOTIFY system
        to inform other components that certain jobs have been cancelled. This
        allows running tasks to check for cancellation and terminate if necessary.

        Args:
            ids (list[models.JobId]): The IDs of the jobs that have been cancelled.
        """
        await self.driver.execute(
            self.qbq.create_notify_query(),
            models.CancellationEvent(
                channel=self.qbq.settings.channel,
                ids=ids,
                sent_at=helpers.utc_now(),
                type="cancellation_event",
            ).model_dump_json(),
        )

    async def update_heartbeat(self, job_ids: list[models.JobId]) -> None:
        await self.driver.execute(
            self.qbq.create_update_heartbeat_query(),
            list(set(job_ids)),
        )

    async def insert_schedule(
        self,
        schedules: dict[models.CronExpressionEntrypoint, timedelta],
    ) -> None:
        await self.driver.execute(
            self.qbs.create_insert_schedule_query(),
            [k.expression for k in schedules],
            [k.entrypoint for k in schedules],
            list(schedules.values()),
        )

    async def fetch_schedule(
        self,
        entrypoints: dict[models.CronExpressionEntrypoint, timedelta],
    ) -> list[models.Schedule]:
        return [
            models.Schedule.model_validate(dict(row))
            for row in await self.driver.fetch(
                self.qbs.create_fetch_schedule_query(),
                [s for _, s in entrypoints],
                [n for n, _ in entrypoints],
                list(entrypoints.values()),
            )
        ]

    async def set_schedule_queued(self, ids: set[models.ScheduleId]) -> None:
        await self.driver.execute(
            self.qbs.create_set_schedule_queued_query(),
            list(ids),
        )

    async def update_schedule_heartbeat(self, ids: set[models.ScheduleId]) -> None:
        await self.driver.execute(
            self.qbs.create_update_schedule_heartbeat(),
            list(ids),
        )

    async def peak_schedule(self) -> list[models.Schedule]:
        return [
            models.Schedule.model_validate(dict(row))
            for row in await self.driver.fetch(
                self.qbs.create_peak_schedule_query(),
            )
        ]

    async def delete_schedule(
        self,
        ids: set[models.ScheduleId],
        entrypoints: set[str],
    ) -> None:
        await self.driver.execute(
            self.qbs.create_delete_schedule_query(),
            list(ids),
            list(entrypoints),
        )

    async def clear_schedule(self) -> None:
        await self.driver.execute(
            self.qbs.create_truncate_schedule_query(),
        )
