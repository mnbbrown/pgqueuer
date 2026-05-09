# executors.py

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Literal, TypeAlias, TypeVar, cast

from croniter import croniter

from pgqueuer.domain import errors, models, types
from pgqueuer.domain.models import utc_now

AsyncEntrypoint: TypeAlias = Callable[[models.Job], Awaitable[None]]
AsyncContextEntrypoint: TypeAlias = Callable[[models.Job, models.Context], Awaitable[None]]
Entrypoint: TypeAlias = AsyncEntrypoint | AsyncContextEntrypoint
EntrypointTypeVar = TypeVar("EntrypointTypeVar", bound=Entrypoint)


AsyncCrontab: TypeAlias = Callable[[models.Schedule], Awaitable[None]]
AsyncContextCrontab: TypeAlias = Callable[
    [models.Schedule, models.ScheduleContext], Awaitable[None]
]
ScheduleCrontab: TypeAlias = AsyncCrontab | AsyncContextCrontab


TerminalFailureReason = Literal["max_attempts", "non_retryable", "timeout"]
OnTerminalFailure: TypeAlias = Callable[
    [Exception, models.Job, TerminalFailureReason],
    None | Awaitable[None],
]


def is_async_callable(obj: Callable[..., object] | object) -> bool:
    """
    Determines whether an object is an asynchronous callable.
    """
    while isinstance(obj, functools.partial):
        obj = obj.func

    return inspect.iscoroutinefunction(obj) or (
        callable(obj) and inspect.iscoroutinefunction(getattr(obj, "__call__", None))
    )


@dataclasses.dataclass
class EntrypointExecutorParameters:
    concurrency_limit: int
    func: Entrypoint
    accepts_context: bool = False
    on_failure: types.OnFailure = "delete"
    serialize_dispatch_per_key: bool = False


@dataclasses.dataclass
class AbstractEntrypointExecutor(ABC):
    """
    Abstract base class for job executors.

    Users can subclass this to create custom job executors.
    """

    parameters: EntrypointExecutorParameters

    @abstractmethod
    async def execute(self, job: models.Job, context: models.Context) -> None:
        """
        Execute the given job.

        Args:
            job (models.Job): The job to execute.
            context (models.Context): The context for the job.
        """


@dataclasses.dataclass
class EntrypointExecutor(AbstractEntrypointExecutor):
    """
    Job executor that wraps an async entrypoint function.

    Executes the provided function when processing a job.
    """

    def __post_init__(self) -> None:
        if not is_async_callable(cast(Callable[..., object], self.parameters.func)):
            raise TypeError(
                "Entrypoint function must be async (defined with 'async def'). "
                "Sync entrypoints are no longer supported. "
                "Wrap blocking code with asyncio.to_thread(): "
                "async def my_entry(job): await asyncio.to_thread(blocking_fn, job)"
            )

    async def execute(self, job: models.Job, context: models.Context) -> None:
        """
        Execute the job using the wrapped function.

        Args:
            job (models.Job): The job to execute.
            context (models.Context): The context for the job.
        """
        if self.parameters.accepts_context:
            await cast(AsyncContextEntrypoint, self.parameters.func)(job, context)
        else:
            await cast(AsyncEntrypoint, self.parameters.func)(job)


@dataclasses.dataclass
class DatabaseRetryEntrypointExecutor(EntrypointExecutor):
    """Executor that converts exceptions into database-level retries.

    On failure the job is re-queued via :class:`~pgqueuer.domain.errors.RetryRequested`
    with exponential backoff derived from ``job.attempts``.  After *max_attempts*
    consecutive failures the exception propagates as a terminal failure.

    Two opt-in escape hatches sit alongside the default retry loop:

    * ``non_retryable_errors`` — a tuple of exception types that should bypass
      retry and propagate immediately (e.g. ``(ValidationError,)``).
    * :class:`~pgqueuer.domain.errors.NonRetryableError` raised inside the
      handler does the same thing programmatically.

    ``on_terminal_failure`` fires once when retries are exhausted or skipped,
    giving observability hooks (Sentry capture, custom metrics) a single
    well-defined call site for terminal failures.

    ``max_time`` optionally bounds a single attempt with ``asyncio.wait_for``.
    On timeout the callback fires with ``reason='timeout'`` and the error
    propagates terminally (no retry). ``None`` (default) disables the timeout.
    """

    max_attempts: int = 5
    initial_delay: timedelta = dataclasses.field(default_factory=lambda: timedelta(seconds=1))
    max_delay: timedelta = dataclasses.field(default_factory=lambda: timedelta(minutes=5))
    backoff_multiplier: float = 2.0
    non_retryable_errors: tuple[type[Exception], ...] = dataclasses.field(default_factory=tuple)
    on_terminal_failure: OnTerminalFailure | None = None
    max_time: timedelta | None = None

    async def execute(self, job: models.Job, context: models.Context) -> None:
        try:
            if self.max_time is None:
                await super().execute(job, context)
            else:
                await asyncio.wait_for(
                    super().execute(job, context),
                    self.max_time.total_seconds(),
                )
        except errors.RetryRequested:
            raise
        except errors.NonRetryableError as e:
            await self._handle_terminal(e, job, "non_retryable")
            raise
        except (TimeoutError, asyncio.TimeoutError) as e:
            # Treat as terminal "timeout" only when our wait_for produced it.
            # When max_time is None, a handler-raised TimeoutError falls through
            # to the generic retry classification.
            if self.max_time is not None:
                await self._handle_terminal(e, job, "timeout")
                raise
            await self._classify_and_retry(e, job)
        except Exception as e:
            await self._classify_and_retry(e, job)

    async def _classify_and_retry(self, exc: Exception, job: models.Job) -> None:
        if self.non_retryable_errors and isinstance(exc, self.non_retryable_errors):
            await self._handle_terminal(exc, job, "non_retryable")
            raise exc
        if job.attempts >= self.max_attempts:
            await self._handle_terminal(exc, job, "max_attempts")
            raise exc
        delay = min(
            self.initial_delay * (self.backoff_multiplier**job.attempts),
            self.max_delay,
        )
        raise errors.RetryRequested(delay=delay, reason=str(exc)) from exc

    async def _handle_terminal(
        self,
        exc: Exception,
        job: models.Job,
        reason: TerminalFailureReason,
    ) -> None:
        if self.on_terminal_failure is None:
            return
        result = self.on_terminal_failure(exc, job, reason)
        if inspect.isawaitable(result):
            await result


######## Schedulers ########


@dataclasses.dataclass
class ScheduleExecutorFactoryParameters:
    entrypoint: str
    expression: str
    func: ScheduleCrontab
    clean_old: bool
    accepts_context: bool = False


@dataclasses.dataclass
class AbstractScheduleExecutor(ABC):
    """
    Abstract base class for job executors.

    This class provides a blueprint for creating job executors that run according to a schedule.
    Users should subclass this to create custom job executors, defining specific execution logic.
    """

    parameters: ScheduleExecutorFactoryParameters

    @abstractmethod
    async def execute(self, schedule: models.Schedule, context: models.ScheduleContext) -> None:
        """
        Execute the given crontab.

        This method must be implemented by subclasses to define the specific behavior of job
        execution.

        Args:
            schedule (models.Schedule): The schedule being executed.
            context (models.ScheduleContext): The context for the scheduled task.
        """

    def get_next(self) -> datetime:
        """
        Calculate the next scheduled run time based on the cron expression.

        Returns:
            datetime: The next scheduled datetime in UTC.
        """
        return datetime.fromtimestamp(
            croniter(self.parameters.expression, start_time=utc_now()).get_next(),
            timezone.utc,
        )

    def next_in(self) -> timedelta:
        """
        Calculate the time remaining until the next scheduled run.

        Returns:
            timedelta: The time difference between now and the next scheduled run.
        """
        return self.get_next() - utc_now()


@dataclasses.dataclass
class ScheduleExecutor(AbstractScheduleExecutor):
    """
    Job executor that wraps an entrypoint function.

    This executor runs the provided function according to the defined schedule.
    It is a concrete implementation of AbstractScheduleExecutor.
    """

    async def execute(self, schedule: models.Schedule, context: models.ScheduleContext) -> None:
        """
        Execute the job using the wrapped function.

        This method calls the provided asynchronous function when the job is triggered.
        """
        if self.parameters.accepts_context:
            await cast(AsyncContextCrontab, self.parameters.func)(schedule, context)
        else:
            await cast(AsyncCrontab, self.parameters.func)(schedule)
