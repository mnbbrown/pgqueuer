from __future__ import annotations

import hashlib
import textwrap

from pgqueuer.domain.schema.model import Function, Trigger
from pgqueuer.domain.settings import DBSettings
from pgqueuer.domain.types import FunctionName, TableName, TriggerName


def suffixed_name(name: str, suffix: str) -> str:
    full = f"{name}_{suffix}"
    if len(full) <= 63:
        return full
    digest = hashlib.sha256(full.encode()).hexdigest()[:16]
    return f"{full[:46]}_{digest}"


def notify_function_body(channel: str) -> str:
    """Keep legacy statement triggers compatible while routing triggers are installed."""
    return textwrap.dedent(
        f"""
        BEGIN
            IF TG_OP IN ('INSERT', 'DELETE', 'TRUNCATE') THEN
                PERFORM pg_notify(
                    '{channel}',
                    json_build_object(
                        'channel', '{channel}',
                        'operation', lower(TG_OP),
                        'sent_at', NOW(),
                        'table', TG_TABLE_NAME,
                        'type', 'table_changed_event'
                    )::text
                );
            END IF;
            RETURN NULL;
        END;
        """
    )


def notify_function(settings: DBSettings) -> Function:
    return Function(
        name=FunctionName(settings.function),
        body=notify_function_body(settings.channel),
    )


def routed_notify_function(settings: DBSettings) -> Function:
    return Function(
        name=FunctionName(suffixed_name(settings.function, "routed")),
        body=textwrap.dedent(
            f"""
            DECLARE
                entrypoints text[];
                payload text;
            BEGIN
                SELECT array_agg(entrypoint ORDER BY entrypoint) INTO entrypoints
                FROM (SELECT DISTINCT entrypoint FROM pgqueuer_changed_rows) AS changed;
                IF entrypoints IS NULL THEN
                    RETURN NULL;
                END IF;
                payload := json_build_object(
                    'channel', '{settings.channel}',
                    'operation', lower(TG_OP),
                    'sent_at', NOW(),
                    'table', TG_TABLE_NAME,
                    'type', 'table_changed_event',
                    'entrypoints', entrypoints
                )::text;
                IF octet_length(payload) >= 7900 THEN
                    payload := json_build_object(
                        'channel', '{settings.channel}',
                        'operation', lower(TG_OP),
                        'sent_at', NOW(),
                        'table', TG_TABLE_NAME,
                        'type', 'table_changed_event'
                    )::text;
                END IF;
                PERFORM pg_notify('{settings.channel}', payload);
                RETURN NULL;
            END;
            """
        ),
    )


def notify_trigger(settings: DBSettings) -> Trigger:
    return Trigger(
        name=TriggerName(settings.trigger),
        table=TableName(settings.queue_table),
        function=FunctionName(settings.function),
        events=("TRUNCATE",),
    )


def routed_notify_triggers(settings: DBSettings) -> tuple[Trigger, ...]:
    function = routed_notify_function(settings).name
    return (
        Trigger(
            name=TriggerName(suffixed_name(settings.trigger, "insert")),
            table=TableName(settings.queue_table),
            function=function,
            events=("INSERT",),
            new_table="pgqueuer_changed_rows",
        ),
        Trigger(
            name=TriggerName(suffixed_name(settings.trigger, "delete")),
            table=TableName(settings.queue_table),
            function=function,
            events=("DELETE",),
            old_table="pgqueuer_changed_rows",
        ),
    )
