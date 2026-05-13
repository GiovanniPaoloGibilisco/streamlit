# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2026)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ParallelFragmentCoordinator.

The coordinator owns a ThreadPoolExecutor plus the bookkeeping that lets the
script thread block on outstanding worker fragments and surface
worker-initiated rerun/stop intent. These tests exercise the public surface
in isolation; integration with ScriptRunContext / ScriptRunner lives in
``script_runner_test.py`` and ``script_run_context_test.py``.
"""

from __future__ import annotations

import threading
import time
import unittest

import pytest

from streamlit.runtime.fragment import ParallelFragmentCoordinator
from streamlit.runtime.scriptrunner_utils.exceptions import (
    RerunException,
    StopException,
)
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData


def _wait_for_outstanding_zero(
    c: ParallelFragmentCoordinator, timeout: float = 1.0
) -> None:
    """Spin-wait until the coordinator's outstanding counter drains to zero.

    Used by tests that submit a worker but don't drive ``join()`` themselves;
    the executor's ``finally`` block runs asynchronously so we need to poll.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with c._outstanding_lock:
            if c._outstanding == 0:
                return
        time.sleep(0.01)
    raise AssertionError(
        f"outstanding never reached 0 within {timeout}s (last value: {c._outstanding})"
    )


class ParallelFragmentCoordinatorTest(unittest.TestCase):
    def test_construction_defaults(self):
        """A freshly constructed coordinator is idle: no outstanding work,
        no captured worker exception, no stop requested."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        try:
            assert c._outstanding == 0
            assert c.worker_exception is None
            assert c.should_stop() is False
        finally:
            c.drain()

    def test_submit_counter_round_trip(self):
        """submit() must restore _outstanding to zero even when the worker
        raises, otherwise join() would hang forever on a worker error."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        try:

            def explodes() -> None:
                raise ValueError("boom")

            c.submit(explodes)
            _wait_for_outstanding_zero(c)
            assert c._outstanding == 0
        finally:
            c.drain()

    def test_join_returns_immediately_when_idle(self):
        """If no work was submitted, join() must not invoke yield_check.

        Calling yield_check with no outstanding work could surface external
        cancellation requests at the wrong time relative to the script
        runner's existing yield logic.
        """
        yields: list[int] = []
        c = ParallelFragmentCoordinator(yield_check=lambda: yields.append(1))
        c.join()
        assert yields == []

    def test_join_waits_and_yields(self):
        """While work is outstanding, join() polls and calls yield_check
        each interval so the script thread stays responsive."""
        yields: list[int] = []
        c = ParallelFragmentCoordinator(
            yield_check=lambda: yields.append(1),
            poll_interval=0.01,
        )
        gate = threading.Event()
        c.submit(lambda: gate.wait(timeout=2.0))
        threading.Timer(0.05, gate.set).start()
        c.join()
        assert len(yields) >= 1

    def test_join_raises_stored_rerun_exception_with_data(self):
        """A worker's RerunException must surface from join() with its
        original RerunData attached so the script runner's rerun loop sees
        the correct request."""
        rerun_data = RerunData()
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        c.request_rerun(RerunException(rerun_data))
        with pytest.raises(RerunException) as excinfo:
            c.join()
        assert excinfo.value.rerun_data is rerun_data

    def test_join_raises_stored_stop_exception(self):
        """request_stop() stores a StopException that join() re-raises."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        c.request_stop()
        with pytest.raises(StopException):
            c.join()

    def test_first_writer_wins_and_should_stop(self):
        """The first request_rerun/request_stop wins; later calls are
        ignored. Both unconditionally set the stop event."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        try:
            assert not c.should_stop()
            rerun_exc = RerunException(RerunData())
            c.request_rerun(rerun_exc)
            c.request_stop()
            assert c.worker_exception is rerun_exc
            assert c.should_stop() is True
        finally:
            c.drain()

    def test_first_writer_wins_stop_then_rerun(self):
        """Symmetric to test_first_writer_wins_and_should_stop: first
        request_stop wins over a later request_rerun."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        try:
            c.request_stop()
            assert isinstance(c.worker_exception, StopException)
            c.request_rerun(RerunException(RerunData()))
            assert isinstance(c.worker_exception, StopException)
            assert c.should_stop() is True
        finally:
            c.drain()

    def test_drain_silent_and_synchronous(self):
        """drain() must not call yield_check (it's invoked from except
        blocks where re-raising would shadow the original exception) and
        must wait synchronously for in-flight workers to exit."""
        yields: list[int] = []
        c = ParallelFragmentCoordinator(yield_check=lambda: yields.append(1))
        gate = threading.Event()
        worker_done = threading.Event()

        def worker() -> None:
            gate.wait(timeout=2.0)
            worker_done.set()

        c.submit(worker)
        threading.Timer(0.05, gate.set).start()
        c.drain()
        assert worker_done.is_set()
        assert yields == []

    def test_drain_sets_stop_event(self):
        """drain() must set the stop event so any worker that reaches a
        yield point during drain notices and exits cooperatively."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        assert not c.should_stop()
        c.drain()
        assert c.should_stop()

    def test_nested_submit_counter(self):
        """A worker that calls submit() must increment the counter before
        the parent's tracked() finally runs, so join() waits for the
        grandchild rather than returning when the parent finishes."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None, poll_interval=0.01)
        child_done = threading.Event()

        def parent() -> None:
            c.submit(lambda: child_done.set())

        c.submit(parent)
        c.join()
        assert child_done.is_set()

    def test_join_propagates_yield_check_exception(self):
        """If yield_check raises (e.g. an external RERUN arrived while
        join() was polling), the exception must propagate so the caller's
        try/except can run drain()."""

        def yield_raises() -> None:
            raise RerunException(RerunData())

        c = ParallelFragmentCoordinator(yield_check=yield_raises, poll_interval=0.01)
        gate = threading.Event()
        c.submit(lambda: gate.wait(timeout=2.0))
        try:
            with pytest.raises(RerunException):
                c.join()
        finally:
            gate.set()
            c.drain()

    def test_submit_passes_args(self):
        """submit() forwards positional args to the worker function so
        callers can capture per-fragment context (the future
        ``_dispatch_parallel_fragment`` will rely on this)."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        try:
            captured: list[tuple[int, str]] = []
            done = threading.Event()

            def worker(value: int, label: str) -> None:
                captured.append((value, label))
                done.set()

            c.submit(worker, 42, "hello")
            assert done.wait(timeout=1.0)
            _wait_for_outstanding_zero(c)
            assert captured == [(42, "hello")]
        finally:
            c.drain()

    def test_join_after_drain_is_safe(self):
        """drain() shuts the executor down; calling join() afterwards on
        an idle coordinator must not crash."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        c.drain()
        c.join()

    def test_submit_after_shutdown_rolls_back_outstanding(self):
        """If ``submit()`` races with a concurrent ``drain()`` and the
        executor is already shut down, the outstanding counter must be
        rolled back; otherwise a future ``join()`` on the (single-use,
        but defensively coded) coordinator would hang forever."""
        c = ParallelFragmentCoordinator(yield_check=lambda: None)
        c.drain()
        with pytest.raises(RuntimeError):
            c.submit(lambda: None)
        assert c._outstanding == 0
