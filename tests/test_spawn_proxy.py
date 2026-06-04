"""Tests for spawn_proxy — UID drop, cwd restriction, env allowlist.

TDD: written before implementation (T-063, FR-409/TS-411).
Extended by T-069/T-070.
"""
from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

import spawn_proxy


# ---------------------------------------------------------------------------
# configure() not called — RuntimeError
# ---------------------------------------------------------------------------

class TestSpawnWithoutConfigure:
    def setup_method(self):
        # Reset module-level config before each test
        spawn_proxy._CONFIG = None

    def test_spawn_raises_before_configure(self):
        with pytest.raises(RuntimeError, match="spawn_proxy.configure\\(\\) must be called first"):
            spawn_proxy.spawn(["echo", "hello"])


# ---------------------------------------------------------------------------
# configure() + spawn() happy path
# ---------------------------------------------------------------------------

class TestSpawnAfterConfigure:
    def setup_method(self):
        spawn_proxy._CONFIG = None

    def test_configure_sets_config(self, tmp_path):
        spawn_proxy.configure(
            uid=1000,
            gid=1000,
            cwd=str(tmp_path),
            env_allowlist=["PATH", "HOME"],
        )
        assert spawn_proxy._CONFIG is not None
        assert spawn_proxy._CONFIG["uid"] == 1000
        assert spawn_proxy._CONFIG["gid"] == 1000
        assert spawn_proxy._CONFIG["cwd"] == str(tmp_path)
        assert spawn_proxy._CONFIG["env_allowlist"] == ["PATH", "HOME"]

    def test_spawn_passes_cwd(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("HOME", "/tmp")
        spawn_proxy.configure(
            uid=os.getuid(),
            gid=os.getgid(),
            cwd=str(tmp_path),
            env_allowlist=["PATH", "HOME"],
        )

        captured: dict = {}

        def fake_run(args, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(args, 0)

        with patch("spawn_proxy.subprocess.run", fake_run):
            spawn_proxy.spawn(["true"])

        assert captured["cwd"] == str(tmp_path)

    def test_spawn_filters_env_by_allowlist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("HOME", "/tmp")
        monkeypatch.setenv("SECRET_TOKEN", "should-be-filtered")
        spawn_proxy.configure(
            uid=1000,
            gid=1000,
            cwd=str(tmp_path),
            env_allowlist=["PATH", "HOME"],
        )

        captured: dict = {}

        def fake_run(args, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(args, 0)

        with patch("spawn_proxy.subprocess.run", fake_run):
            spawn_proxy.spawn(["true"])

        env = captured["env"]
        assert "PATH" in env
        assert "HOME" in env
        assert "SECRET_TOKEN" not in env

    def test_spawn_allows_only_listed_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", "/bin")
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("EXTRA", "noise")
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

        with patch("spawn_proxy.subprocess.run", fake_run):
            spawn_proxy.spawn(["true"])

        assert set(captured["env"].keys()) <= {"PATH"}
