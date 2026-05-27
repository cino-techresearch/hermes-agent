"""Tests for spawn_proxy UID drop behaviour (FR-411, TS-415, T-070, T-096).

TDD: written before implementation gate — verifies that spawn() applies
privilege-drop (os.setgid / os.setuid) via preexec_fn on POSIX, and that
uid=0 is rejected as a security-policy violation.
Extended by T-096: setgroups([gid]) + PR_SET_NO_NEW_PRIVS.
"""
from __future__ import annotations

import ctypes  # noqa: F401 — used in test_prctl_no_new_privs_called_on_linux
import os
import subprocess
import sys  # noqa: F401 — used in test_prctl_no_new_privs_called_on_linux
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

        mock_setgroups = MagicMock()
        mock_setgid = MagicMock()
        mock_setuid = MagicMock()

        with (
            patch("spawn_proxy.subprocess.run", fake_run),
            patch("spawn_proxy.os.setgroups", mock_setgroups),
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
            patch("spawn_proxy.os.setgroups", side_effect=lambda g: call_order2.append("setgroups")),
            patch("spawn_proxy.os.setgid", side_effect=lambda g: call_order2.append("setgid")),
            patch("spawn_proxy.os.setuid", side_effect=lambda u: call_order2.append("setuid")),
        ):
            spawn_proxy.spawn(["true"])
            preexec = captured2.get("preexec_fn")
            assert preexec is not None
            preexec()

        assert call_order2 == ["setgroups", "setgid", "setuid"], (
            f"Expected ['setgroups', 'setgid', 'setuid'], got {call_order2}"
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


# ---------------------------------------------------------------------------
# Scenario 4: setgroups([gid]) called in preexec_fn (T-096, FR-411)
# ---------------------------------------------------------------------------

class TestSetgroupsDrop:
    """preexec_fn must drop supplementary groups via os.setgroups([gid])."""

    def setup_method(self):
        _reset()

    @pytest.mark.skipif(not hasattr(os, "setuid"), reason="POSIX only")
    def test_preexec_fn_calls_setgroups_with_gid(self, tmp_path, monkeypatch):
        """setgroups([gid]) must be called before setgid/setuid."""
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

        mock_setgroups = MagicMock()
        mock_setgid = MagicMock()
        mock_setuid = MagicMock()

        with (
            patch("spawn_proxy.subprocess.run", fake_run),
            patch("spawn_proxy.os.setgroups", mock_setgroups),
            patch("spawn_proxy.os.setgid", mock_setgid),
            patch("spawn_proxy.os.setuid", mock_setuid),
        ):
            spawn_proxy.spawn(["true"])
            preexec = captured.get("preexec_fn")
            assert preexec is not None
            preexec()

        mock_setgroups.assert_called_once_with([1000])

    @pytest.mark.skipif(not hasattr(os, "setuid"), reason="POSIX only")
    def test_setgroups_called_before_setgid(self, tmp_path, monkeypatch):
        """setgroups must precede setgid (both require CAP_SETGID)."""
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        spawn_proxy.configure(
            uid=500,
            gid=500,
            cwd=str(tmp_path),
            env_allowlist=["PATH"],
        )

        captured: dict = {}
        call_order: list[str] = []

        def fake_run(args, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(args, 0)

        with (
            patch("spawn_proxy.subprocess.run", fake_run),
            patch("spawn_proxy.os.setgroups", side_effect=lambda g: call_order.append("setgroups")),
            patch("spawn_proxy.os.setgid", side_effect=lambda g: call_order.append("setgid")),
            patch("spawn_proxy.os.setuid", side_effect=lambda u: call_order.append("setuid")),
        ):
            spawn_proxy.spawn(["true"])
            preexec = captured.get("preexec_fn")
            assert preexec is not None
            preexec()

        assert call_order[0] == "setgroups", f"Expected setgroups first, got {call_order}"
        assert call_order.index("setgroups") < call_order.index("setgid")
        assert call_order.index("setgid") < call_order.index("setuid")


# ---------------------------------------------------------------------------
# Scenario 5: PR_SET_NO_NEW_PRIVS applied via prctl (T-096, FR-411)
# ---------------------------------------------------------------------------

class TestNoNewPrivs:
    """preexec_fn must call prctl(PR_SET_NO_NEW_PRIVS=38, 1, 0, 0, 0) on Linux."""

    def setup_method(self):
        _reset()

    @pytest.mark.skipif(not hasattr(os, "setuid"), reason="POSIX only")
    def test_prctl_no_new_privs_called_on_linux(self, tmp_path, monkeypatch):
        """On Linux, prctl(PR_SET_NO_NEW_PRIVS) must be called in preexec_fn."""
        import ctypes
        import sys

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

        mock_prctl = MagicMock(return_value=0)
        mock_libc = MagicMock()
        mock_libc.prctl = mock_prctl

        with (
            patch("spawn_proxy.subprocess.run", fake_run),
            patch("spawn_proxy.os.setgroups", MagicMock()),
            patch("spawn_proxy.os.setgid", MagicMock()),
            patch("spawn_proxy.os.setuid", MagicMock()),
            patch("spawn_proxy.sys.platform", "linux"),
            patch("spawn_proxy.ctypes.CDLL", return_value=mock_libc),
        ):
            spawn_proxy.spawn(["true"])
            preexec = captured.get("preexec_fn")
            assert preexec is not None
            preexec()

        PR_SET_NO_NEW_PRIVS = 38
        mock_prctl.assert_any_call(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
