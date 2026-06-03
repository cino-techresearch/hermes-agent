"""Regression tests for terminal command handling."""

import json

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_actual_sudo_after_leading_env_assignment_is_rewritten(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("DEBUG=1 sudo whoami")

    assert transformed == "DEBUG=1 sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_cached_sudo_password_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool._set_cached_sudo_password("cached-pass")

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("echo ok && sudo whoami")

    assert transformed == "echo ok && sudo -S -p '' whoami"
    assert sudo_stdin == "cached-pass\n"


def test_cached_sudo_password_isolated_by_session_key(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    terminal_tool._set_cached_sudo_password("alpha-pass")

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-b")
    assert terminal_tool._get_cached_sudo_password() == ""

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    assert terminal_tool._get_cached_sudo_password() == "alpha-pass"


def test_validate_workdir_allows_windows_drive_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project") is None
    assert terminal_tool._validate_workdir("C:/Users/Alice/project") is None


def test_validate_workdir_allows_windows_unc_paths():
    assert terminal_tool._validate_workdir(r"\\server\share\project") is None


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_terminal_subprocess_receives_gateway_session_context(monkeypatch):
    """Terminal-launched CLI commands need origin metadata for cron delivery."""
    from gateway.session_context import clear_session_vars, set_session_vars

    class DummyEnvironment:
        def __init__(self) -> None:
            self.env: dict[str, str] = {}

        def execute(self, command, **kwargs):  # noqa: ANN001, ANN201
            keys = (
                "HERMES_SESSION_PLATFORM",
                "HERMES_SESSION_CHAT_ID",
                "HERMES_SESSION_CHAT_NAME",
                "HERMES_SESSION_THREAD_ID",
                "HERMES_SESSION_USER_ID",
                "HERMES_SESSION_USER_NAME",
                "HERMES_SESSION_KEY",
            )
            return {
                "output": json.dumps({key: self.env.get(key) for key in keys}),
                "returncode": 0,
            }

    dummy = DummyEnvironment()
    config = {
        "env_type": "local",
        "cwd": ".",
        "timeout": 30,
        "lifetime_seconds": 300,
    }
    tokens = set_session_vars(
        platform="telegram",
        chat_id="-100123",
        chat_name="Ops",
        thread_id="456",
        user_id="789",
        user_name="ram",
        session_key="telegram:-100123:456",
    )
    try:
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda *_args, **_kwargs: {"approved": True},
        )
        monkeypatch.setitem(terminal_tool._active_environments, "default", dummy)
        monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)

        result = json.loads(terminal_tool.terminal_tool(command="hermes cron create 1m 'ping'"))
        seen_env = json.loads(result["output"])

        assert seen_env == {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "-100123",
            "HERMES_SESSION_CHAT_NAME": "Ops",
            "HERMES_SESSION_THREAD_ID": "456",
            "HERMES_SESSION_USER_ID": "789",
            "HERMES_SESSION_USER_NAME": "ram",
            "HERMES_SESSION_KEY": "telegram:-100123:456",
        }
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)
        clear_session_vars(tokens)
