"""Subprocess proxy for HermesRunner tools.

Wraps subprocess.run / Popen with UID drop, cwd restriction, env allowlist.
Called once via configure() at HermesRunner boot.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

_CONFIG: dict[str, Any] | None = None


def configure(*, uid: int, gid: int, cwd: str, env_allowlist: list[str]) -> None:
    """Set process-wide spawn configuration. Must be called before spawn()."""
    if uid == 0 or gid == 0:
        raise ValueError(
            f"spawn_proxy: root uid/gid forbidden (uid={uid}, gid={gid})"
        )
    global _CONFIG
    _CONFIG = {"uid": uid, "gid": gid, "cwd": cwd, "env_allowlist": env_allowlist}


def _build_spawn_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply UID drop, cwd restriction, env allowlist to kwargs."""
    if _CONFIG is None:
        raise RuntimeError("spawn_proxy.configure() must be called first")

    filtered_env = {
        k: v for k, v in os.environ.items() if k in _CONFIG["env_allowlist"]
    }

    uid = _CONFIG["uid"]
    gid = _CONFIG["gid"]

    def _drop_privileges() -> None:
        os.setgid(gid)
        os.setuid(uid)

    kwargs.setdefault("cwd", _CONFIG["cwd"])
    kwargs.setdefault("env", filtered_env)
    # Only apply preexec_fn on POSIX; skip on Windows (no os.setuid).
    if hasattr(os, "setuid"):
        kwargs.setdefault("preexec_fn", _drop_privileges)

    return kwargs


def spawn(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run args as a subprocess with UID drop, cwd restriction, env allowlist."""
    return subprocess.run(args, **_build_spawn_kwargs(kwargs))


def spawn_popen(args: list[str], **kwargs: Any) -> subprocess.Popen:
    """Open args as a long-running subprocess with UID drop, cwd restriction, env allowlist."""
    return subprocess.Popen(args, **_build_spawn_kwargs(kwargs))
