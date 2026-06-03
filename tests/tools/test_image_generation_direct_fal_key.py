"""Direct (non-gateway) FAL submit path resolves FAL_KEY per-tenant.

Regression coverage for the in-process multi-tenant key-isolation fix: the
direct ``_submit_fal_request`` branch must resolve ``FAL_KEY`` via
``get_env_value`` (os.environ → per-tenant ``get_hermes_home()/.env``) and bind
it to an explicit ``fal_client.SyncClient(key=...)`` instead of relying on the
process-global ``os.environ["FAL_KEY"]`` read inside ``fal_client.submit`` —
which would otherwise leak keys across concurrent tenant runs.
"""

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


def _load_tool_module(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, TOOLS_DIR / filename)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_tool_and_agent_modules():
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "tools"
        or name.startswith("tools.")
        or name in {"fal_client"}
    }
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "tools" or name.startswith("tools.") or name in {"fal_client"}:
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def _install_fake_tools_package():
    tools_package = types.ModuleType("tools")
    tools_package.__path__ = [str(TOOLS_DIR)]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_package
    sys.modules["tools.debug_helpers"] = types.SimpleNamespace(
        DebugSession=lambda *args, **kwargs: types.SimpleNamespace(
            active=False,
            session_id="debug-session",
            log_call=lambda *a, **k: None,
            save=lambda: None,
            get_session_info=lambda: {},
        )
    )
    sys.modules["tools.managed_tool_gateway"] = _load_tool_module(
        "tools.managed_tool_gateway",
        "managed_tool_gateway.py",
    )


def _install_fake_fal_client(captured):
    def submit(model, arguments=None, headers=None):
        # The direct path must NOT fall back to the process-global module-level
        # submit when a key is resolvable — that read ignores the tenant .env.
        captured["module_submit_called"] = True
        raise AssertionError("direct path with a resolvable key must use SyncClient")

    class SyncClient:
        def __init__(self, key=None, default_timeout=120.0):
            captured.setdefault("client_keys", []).append(key)
            self.key = key
            self.default_timeout = default_timeout
            self._client = object()

        def submit(self, model, arguments=None, *, headers=None, **kwargs):
            captured.setdefault("submits", []).append(
                {
                    "key": self.key,
                    "model": model,
                    "arguments": arguments,
                    "headers": headers,
                }
            )
            return types.SimpleNamespace(request_id="req-direct")

    fal_client_module = types.SimpleNamespace(submit=submit, SyncClient=SyncClient)
    sys.modules["fal_client"] = fal_client_module
    return fal_client_module


def test_direct_fal_submit_uses_tenant_resolved_key(monkeypatch, tmp_path):
    captured: dict = {}
    _install_fake_tools_package()
    _install_fake_fal_client(captured)
    # os.environ has NO key; the key lives only in the tenant .env (simulated
    # via get_env_value). The old code path would silently ignore this.
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name: "tenant-a-key" if name == "FAL_KEY" else None,
    )

    image_generation_tool = _load_tool_module(
        "tools.image_generation_tool",
        "image_generation_tool.py",
    )
    monkeypatch.setattr(image_generation_tool, "_resolve_managed_fal_gateway", lambda: None)
    monkeypatch.setattr(image_generation_tool.uuid, "uuid4", lambda: "direct-123")

    image_generation_tool._submit_fal_request("fal-ai/flux-2-pro", {"prompt": "hi"})

    assert captured.get("module_submit_called") is not True
    assert captured["client_keys"] == ["tenant-a-key"]
    assert captured["submits"] == [
        {
            "key": "tenant-a-key",
            "model": "fal-ai/flux-2-pro",
            "arguments": {"prompt": "hi"},
            "headers": {"x-idempotency-key": "direct-123"},
        }
    ]


def test_direct_fal_clients_isolated_per_tenant(monkeypatch, tmp_path):
    """Two tenants with distinct homes+keys never share a SyncClient, so a
    concurrent run cannot submit with another tenant's credentials."""
    captured: dict = {}
    _install_fake_tools_package()
    _install_fake_fal_client(captured)
    monkeypatch.delenv("FAL_KEY", raising=False)

    image_generation_tool = _load_tool_module(
        "tools.image_generation_tool",
        "image_generation_tool.py",
    )
    monkeypatch.setattr(image_generation_tool, "_resolve_managed_fal_gateway", lambda: None)

    import hermes_constants as hc

    home_a = tmp_path / "tenant_a"
    home_b = tmp_path / "tenant_b"

    def _run_as(home, key):
        monkeypatch.setattr(hc, "get_hermes_home", lambda: home)
        monkeypatch.setattr(
            "hermes_cli.config.get_env_value",
            lambda name, _k=key: _k if name == "FAL_KEY" else None,
        )
        image_generation_tool._submit_fal_request("fal-ai/flux-2-pro", {"prompt": "x"})

    _run_as(home_a, "key-a")
    _run_as(home_b, "key-b")
    _run_as(home_a, "key-a")  # tenant A again → cached, no new client

    # Two distinct clients created (one per tenant), each bound to its own key.
    assert captured["client_keys"] == ["key-a", "key-b"]
    submit_keys = [s["key"] for s in captured["submits"]]
    assert submit_keys == ["key-a", "key-b", "key-a"]


def test_direct_fal_submit_falls_back_when_no_key(monkeypatch):
    """No key anywhere → preserve the library's own credential resolution and
    error surface for the single-user/CLI path."""
    captured: dict = {}
    _install_fake_tools_package()
    fake = _install_fake_fal_client(captured)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda name: None)

    # Replace the raising module-level submit with a recording one for this case.
    def submit(model, arguments=None, headers=None):
        captured["module_submit_called"] = True
        return types.SimpleNamespace(request_id="req-fallback")

    fake.submit = submit

    image_generation_tool = _load_tool_module(
        "tools.image_generation_tool",
        "image_generation_tool.py",
    )
    monkeypatch.setattr(image_generation_tool, "_resolve_managed_fal_gateway", lambda: None)

    image_generation_tool._submit_fal_request("fal-ai/flux-2-pro", {"prompt": "hi"})

    assert captured["module_submit_called"] is True
    assert "client_keys" not in captured
