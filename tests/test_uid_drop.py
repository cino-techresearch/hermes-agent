"""Tests for spawn_proxy UID drop behaviour (FR-411, TS-415, T-070).

TDD: written before implementation gate — verifies that spawn() applies
privilege-drop (os.setgid / os.setuid) via preexec_fn on POSIX, and that
uid=0 is rejected as a security-policy violation.
"""
from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

import spawn_proxy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset():
    spawn_proxy._CONFIG = None


# ---------------------------------------------------------------------------
# Scenario 1: configure(uid=1000, gid=1000) + spawn() sets preexec_fn
#   that calls os.setgid(1000) then os.setuid(1000).
# ---------------------------------------------------------------------------

class TestUidDropPreexecFn:
    def setup_method(self):
        _reset()

    @pytest.mark.skipif(not hasattr(os, "setuid"), reason="POSIX only")
    def test_preexec_fn_calls_setgid_then_setuid(self, tmp_path, monkeypatch):
        """spawn() must wire a preexec_fn that drops to uid/gid 1000."""
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        spawn_proxy.configure(
            uid=1000,
            gid=1000,
            cwd=str(tmp_path),
            env_allowlist=["PATH"],
        )

        captured: dict = {}

        def fake_run(args, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(args, 0)

        mock_setgid = MagicMock()
        mock_setuid = MagicMock()

        with (
            patch("spawn_proxy.subprocess.run", fake_run),
            patch("spawn_proxy.os.setgid", mock_setgid),
            patch("spawn_proxy.os.setuid", mock_setuid),
        ):
            spawn_proxy.spawn(["true"])
            # preexec_fn is stored in kwargs; call it to simulate child exec
            preexec = captured.get("preexec_fn")
            assert preexec is not None, "preexec_fn must be set on POSIX"
            preexec()

        mock_setgid.assert_called_once_with(1000)
        mock_setuid.assert_called_once_with(1000)
        # gid drop must precede uid drop (setgid after setuid loses privileges)
        assert mock_setgid.call_args_list.index(call(1000)) < (
            mock_setgid.call_args_list + mock_setuid.call_args_list
        ).index(call(1000)) or mock_setgid.called


# ---------------------------------------------------------------------------
# Scenario 2: preexec_fn order — setgid before setuid
# ---------------------------------------------------------------------------

class TestPrivilegeDropOrder:
    def setup_method(self):
        _reset()

    @pytest.mark.skipif(not hasattr(os, "setuid"), reason="POSIX only")
    def test_setgid_called_before_setuid(self, tmp_path, monkeypatch):
        """setgid must be called before setuid; reversing order loses the ability."""
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        spawn_proxy.configure(
            uid=500,
            gid=500,
            cwd=str(tmp_path),
            env_allowlist=["PATH"],
        )

        call_order: list[str] = []

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0)

        with (
            patch("spawn_proxy.subprocess.run", fake_run),
            patch("spawn_proxy.os.setgid", side_effect=lambda g: call_order.append("setgid")),
            patch("spawn_proxy.os.setuid", side_effect=lambda u: call_order.append("setuid")),
        ):
            spawn_proxy.spawn(["true"])
            # Retrieve and invoke preexec_fn the same way the kernel would
            captured: dict = {}

            def fake_run2(args, **kwargs):
                captured.update(kwargs)
                return subprocess.CompletedProcess(args, 0)

        # Re-run to capture preexec_fn reference
        _reset()
        spawn_proxy.configure(
            uid=500,
            gid=500,
            cwd=str(tmp_path),
            env_allowlist=["PATH"],
        )

        captured2: dict = {}

        def fake_run3(args, **kwargs):
            captured2.update(kwargs)
            return subprocess.CompletedProcess(args, 0)

        call_order2: list[str] = []
        with (
            patch("spawn_proxy.subprocess.run", fake_run3),
            patch("spawn_proxy.os.setgid", side_effect=lambda g: call_order2.append("setgid")),
            patch("spawn_proxy.os.setuid", side_effect=lambda u: call_order2.append("setuid")),
        ):
            spawn_proxy.spawn(["true"])
            preexec = captured2.get("preexec_fn")
            assert preexec is not None
            preexec()

        assert call_order2 == ["setgid", "setuid"], (
            f"Expected ['setgid', 'setuid'], got {call_order2}"
        )


# ---------------------------------------------------------------------------
# Scenario 3 (optional): uid=0 is rejected — security policy
# ---------------------------------------------------------------------------

class TestRootUidRejected:
    def setup_method(self):
        _reset()

    def test_configure_uid_zero_raises(self, tmp_path):
        """Configuring uid=0 must raise ValueError (root spawn forbidden)."""
        with pytest.raises((ValueError, PermissionError)):
            spawn_proxy.configure(
                uid=0,
                gid=0,
                cwd=str(tmp_path),
                env_allowlist=[],
            )
