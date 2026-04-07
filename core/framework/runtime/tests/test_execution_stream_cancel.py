"""Regression tests for forced-cancel behavior in ExecutionStream."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.runtime.execution_stream import (
    EntryPointSpec,
    ExecutionAlreadyRunningError,
    ExecutionContext,
    ExecutionStream,
)
from framework.runtime.shared_state import IsolationLevel, SharedBufferManager


@pytest.mark.asyncio
async def test_execute_does_not_start_second_execution_when_cancel_times_out(monkeypatch):
    """A timed-out cancel must not unlock the stream for a second execution."""
    stream = ExecutionStream(
        stream_id="api",
        entry_spec=EntryPointSpec(
            id="api",
            name="API",
            entry_node="start",
            trigger_type="api",
            max_concurrent=1,
        ),
        graph=SimpleNamespace(id="g1"),
        goal=SimpleNamespace(id="goal-1", description="goal"),
        state_manager=SharedBufferManager(),
        storage=SimpleNamespace(base_path=Path(".")),
        outcome_aggregator=SimpleNamespace(),
    )
    stream._running = True

    release = asyncio.Event()

    async def ignores_cancel():
        while True:
            try:
                await release.wait()
                return
            except asyncio.CancelledError:
                # Simulate non-cooperative execution that ignores cancellation.
                continue

    stuck_task = asyncio.create_task(ignores_cancel())
    execution_id = "exec-stuck"
    stream._execution_tasks[execution_id] = stuck_task
    stream._active_executions[execution_id] = ExecutionContext(
        id=execution_id,
        correlation_id=execution_id,
        stream_id=stream.stream_id,
        entry_point=stream.entry_spec.id,
        input_data={},
        isolation_level=IsolationLevel.SHARED,
    )
    stream._completion_events[execution_id] = asyncio.Event()

    async def fake_wait(tasks, timeout=None):
        return set(), set(tasks)

    monkeypatch.setattr(asyncio, "wait", fake_wait)

    try:
        with pytest.raises(ExecutionAlreadyRunningError):
            await stream.execute({"input": "new run"})

        assert execution_id in stream._execution_tasks
        assert not stream._execution_tasks[execution_id].done()
        assert stream._active_executions[execution_id].status == "cancelling"
    finally:
        release.set()
        try:
            await asyncio.sleep(0)
            await asyncio.wait_for(stuck_task, timeout=1.0)
        except (asyncio.CancelledError, TimeoutError):
            pass

    # Once the stuck task actually exits, deferred cleanup should unlock the stream.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert execution_id not in stream._execution_tasks
    assert execution_id not in stream._active_executions
    assert execution_id not in stream._completion_events
