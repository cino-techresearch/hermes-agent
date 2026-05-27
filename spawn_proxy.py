"""Subprocess proxy for HermesRunner tools.

Wraps subprocess.run / Popen with UID drop, cwd restriction, env allowlist.
Called once via configure() at HermesRunner boot.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from typing import Any

PR_SET_NO_NEW_PRIVS = 38

_CONFIG: dict[str, Any] | None = None


def configure(*, uid: int, gid: int, cwd: str, env_allowlist: list[str]) -> None:
    """Set process-wide spawn configuration. Must be called before spawn()."""
    if uid == 0 or gid == 0:
        raise ValueError(
            f"spawn_proxy: root uid/gid forbidden (uid={uid}, gid={gid})"
        )
    global _CONFIG
    _CONFIG = {"uid": uid, "gid": gid, "cwd": cwd, "env_allowlist": env_allowlist}


def spawn(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run args as a subprocess with UID drop, cwd restriction, env allowlist."""
    if _CONFIG is None:
        raise RuntimeError("spawn_proxy.configure() must be called first")

    filtered_env = {
        k: v for k, v in os.environ.items() if k in _CONFIG["env_allowlist"]
    }

    uid = _CONFIG["uid"]
    gid = _CONFIG["gid"]

    def _drop_privileges() -> None:
        # Block setuid-binary privilege escalation before dropping root.
        # Best-effort: Linux only; skip silently on macOS/other platforms.
        if sys.platform == "linux":
            try:
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
            except OSError:
                pass
        # Drop supplementary groups before setgid/setuid (requires CAP_SETGID).
        os.setgroups([gid])
        os.setgid(gid)
        os.setuid(uid)

    kwargs.setdefault("cwd", _CONFIG["cwd"])
    kwargs.setdefault("env", filtered_env)
    # Only apply preexec_fn on POSIX; skip on Windows (no os.setuid).
    if hasattr(os, "setuid"):
        kwargs.setdefault("preexec_fn", _drop_privileges)

    return subprocess.run(args, **kwargs)
