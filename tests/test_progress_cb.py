"""Unit tests: HermesRunner.run_for_user wires progress_cb to tool steps.

FR-404 / TS-405 — verifies that progress_cb is called once per tool step
(tool.started event) when run_for_user executes.
"""

from unittest.mock import MagicMock, patch


def _make_patched_runner(progress_cb):
    """Instantiate HermesRunner and call run_for_user with mocked AIAgent.

    AIAgent.__init__ is mocked to a no-op so no network/file I/O occurs.
    AIAgent.run_conversation is mocked to fire tool_progress_callback twice
    (simulating two tool steps) before returning a minimal result dict.
    """
    from run_agent import HermesRunner

    def _fake_run_conversation(self_, **kwargs):  # noqa: N803 — positional self
        # Simulate two tool steps firing the progress callback.
        for step in ("step_1", "step_2"):
            if self_.tool_progress_callback:
                self_.tool_progress_callback("tool.started", step, step, {})
        return {"response": "done", "messages": []}

    with (
        patch("run_agent.AIAgent.__init__", return_value=None),
        patch("run_agent.AIAgent.run_conversation", _fake_run_conversation),
    ):
        runner = HermesRunner()
        # Set the attribute that AIAgent.__init__ would normally set.
        # We do this by patching it on the constructed instance inside
        # run_for_user — but since __init__ is a no-op, we rely on
        # run_for_user creating the agent and passing tool_progress_callback
        # as a kwarg.  _fake_run_conversation reads it from self_.
        # To let the fake read it, we need AIAgent.__init__ to set the attr.
        # Override: patch __init__ to only set the one attr we need.
        pass

    # Re-run with an __init__ that sets tool_progress_callback.
    def _fake_init(self_, tool_progress_callback=None, **kw):
        self_.tool_progress_callback = tool_progress_callback

    with (
        patch("run_agent.AIAgent.__init__", _fake_init),
        patch("run_agent.AIAgent.run_conversation", _fake_run_conversation),
    ):
        runner = HermesRunner()
        result = runner.run_for_user(
            user_id="u1",
            prompt="hello",
            progress_cb=progress_cb,
        )

    return result


def test_progress_cb_called_per_tool_step():
    """progress_cb must be invoked once per tool step (TS-405)."""
    calls = []

    def cb(event, name, preview, args):
        calls.append((event, name))

    result = _make_patched_runner(cb)

    assert result == {"response": "done", "messages": []}
    assert len(calls) == 2, f"expected 2 calls, got {calls}"
    assert calls[0] == ("tool.started", "step_1")
    assert calls[1] == ("tool.started", "step_2")


def test_progress_cb_none_does_not_raise():
    """run_for_user with progress_cb=None must complete without error."""
    result = _make_patched_runner(None)
    assert result == {"response": "done", "messages": []}


def test_progress_cb_wired_as_tool_progress_callback():
    """HermesRunner must pass progress_cb as AIAgent.tool_progress_callback."""
    received = {}

    def _capture_init(self_, tool_progress_callback=None, **kw):
        self_.tool_progress_callback = tool_progress_callback
        received["cb"] = tool_progress_callback

    def _noop_run(self_, **kw):
        return {"response": "", "messages": []}

    sentinel = MagicMock()

    from run_agent import HermesRunner

    with (
        patch("run_agent.AIAgent.__init__", _capture_init),
        patch("run_agent.AIAgent.run_conversation", _noop_run),
    ):
        HermesRunner().run_for_user(user_id="u1", prompt="x", progress_cb=sentinel)

    assert received["cb"] is sentinel, (
        "progress_cb must be forwarded to AIAgent as tool_progress_callback"
    )
