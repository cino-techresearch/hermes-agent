#!/usr/bin/env python3
"""Bridge between Hermes OAuth token and gws CLI.

Refreshes the token if expired, then executes gws with the valid access token.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import get_hermes_home


def get_token_path() -> Path:
    return get_hermes_home() / "google_token.json"


def _normalize_authorized_user_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def _maybe_read_stdin_token() -> dict | None:
    """Read token JSON from stdin if HERMES_TOKEN_STDIN=1. Fail-closed."""
    if os.environ.get("HERMES_TOKEN_STDIN") != "1":
        return None
    # bounded read (max 64KB)
    line = sys.stdin.buffer.read(64 * 1024)
    if not line:
        sys.exit(1)  # empty stdin
    try:
        payload = json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        sys.exit(1)  # malformed — fail-closed
    required = {"refresh_token", "client_id", "client_secret", "token_uri"}
    if not required.issubset(payload):
        sys.exit(1)  # missing required keys
    return payload


def _emit_refresh_envelope(token: dict) -> None:
    """Emit HERMES_REFRESH_V1 envelope to stderr for backend to capture."""
    import base64
    import hashlib
    import hmac
    import secrets
    import time
    key = os.environ.get("HERMES_REFRESH_HMAC_KEY", "").encode()
    if not key:
        return
    request_id = os.environ.get("HERMES_REQUEST_ID", "")
    payload = {
        "nonce": secrets.token_hex(32),
        "request_id": request_id,
        "user_id": os.environ.get("HERMES_USER_ID", ""),
        "token": token,
        "ts": int(time.time()),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    payload_b64 = base64.b64encode(payload_bytes).decode()
    sig = hmac.new(key, payload_bytes, hashlib.sha256).hexdigest()
    print(f"HERMES_REFRESH_V1 {sig} {payload_b64}", file=sys.stderr, flush=True)


def refresh_token(token_data: dict) -> dict:
    """Refresh the access token using the refresh token."""
    import urllib.error
    import urllib.parse
    import urllib.request

    required_keys = ["client_id", "client_secret", "refresh_token", "token_uri"]
    missing = [k for k in required_keys if k not in token_data]
    if missing:
        print(f"ERROR: google_token.json is missing required fields: {', '.join(missing)}", file=sys.stderr)
        print("Please re-authenticate by running the Google Workspace setup script.", file=sys.stderr)
        sys.exit(1)

    params = urllib.parse.urlencode({
        "client_id": token_data["client_id"],
        "client_secret": token_data["client_secret"],
        "refresh_token": token_data["refresh_token"],
        "grant_type": "refresh_token",
    }).encode()

    req = urllib.request.Request(token_data["token_uri"], data=params)
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: Token refresh failed (HTTP {e.code}): {body}", file=sys.stderr)
        print("Re-run setup.py to re-authenticate.", file=sys.stderr)
        sys.exit(1)

    token_data["token"] = result["access_token"]
    token_data["expiry"] = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + result["expires_in"],
        tz=timezone.utc,
    ).isoformat()

    get_token_path().write_text(
        json.dumps(_normalize_authorized_user_payload(token_data), indent=2)
    )
    return token_data


def get_valid_token() -> str:
    """Return a valid access token, refreshing if needed.

    Prefers stdin pipe token (HERMES_TOKEN_STDIN=1) over file fallback.
    On refresh, emits HERMES_REFRESH_V1 envelope to stderr.
    """
    stdin_token = _maybe_read_stdin_token()
    if stdin_token is not None:
        token_data = stdin_token
        # gws_bridge refresh_token() uses urllib directly — reuse it
        expiry = token_data.get("expiry", "")
        if expiry:
            exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if now >= exp_dt:
                # refresh_token() writes to file; we capture the result and emit envelope
                # but skip the write when in stdin-pipe mode
                token_data = _refresh_token_no_write(token_data)
                _emit_refresh_envelope(token_data)
        if "token" not in token_data:
            print("ERROR: Token payload missing 'token' field.", file=sys.stderr)
            sys.exit(1)
        return token_data["token"]

    # Fallback: file-based token (CLI standalone use)
    token_path = get_token_path()
    if not token_path.exists():
        print("ERROR: No Google token found. Run setup.py --auth-url first.", file=sys.stderr)
        sys.exit(1)

    token_data = json.loads(token_path.read_text())

    expiry = token_data.get("expiry", "")
    if expiry:
        exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if now >= exp_dt:
            token_data = refresh_token(token_data)

    return token_data["token"]


def _refresh_token_no_write(token_data: dict) -> dict:
    """Refresh access token without writing to disk (stdin-pipe mode)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode({
        "client_id": token_data["client_id"],
        "client_secret": token_data["client_secret"],
        "refresh_token": token_data["refresh_token"],
        "grant_type": "refresh_token",
    }).encode()

    req = urllib.request.Request(token_data["token_uri"], data=params)
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: Token refresh failed (HTTP {e.code}): {body}", file=sys.stderr)
        sys.exit(1)

    updated = dict(token_data)
    updated["token"] = result["access_token"]
    updated["expiry"] = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + result["expires_in"],
        tz=timezone.utc,
    ).isoformat()
    return updated


def main():
    """Refresh token if needed, then exec gws with remaining args."""
    if len(sys.argv) < 2:
        print("Usage: gws_bridge.py <gws args...>", file=sys.stderr)
        sys.exit(1)

    access_token = get_valid_token()
    env = os.environ.copy()
    env["GOOGLE_WORKSPACE_CLI_TOKEN"] = access_token

    result = subprocess.run(["gws"] + sys.argv[1:], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
