"""Shared OpenRouter API client for Hermes tools.

Provides a tenant-keyed lazy-initialized AsyncOpenAI client that all tool
modules can share.  Routes through the centralized provider router in
agent/auxiliary_client.py so auth, headers, and API format are handled
consistently.

Each unique HERMES_HOME path (= tenant) gets its own client instance, cached
via functools.lru_cache(maxsize=128) so connections are reused within a tenant
but isolated across tenants (NFR-6, Phase 0-C, T-007).
"""

import functools
import os


@functools.lru_cache(maxsize=128)
def _build_client(tenant_id: str):
    """Build and cache an AsyncOpenAI-compatible client for *tenant_id*.

    *tenant_id* is the string representation of the HERMES_HOME path so that
    each tenant directory gets an isolated client with its own connection pool.
    Raises ValueError if OPENROUTER_API_KEY is not set.
    """
    from agent.auxiliary_client import resolve_provider_client
    client, _model = resolve_provider_client("openrouter", async_mode=True)
    if client is None:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return client


def get_client():
    """Return a tenant-keyed async OpenAI-compatible client for OpenRouter.

    The client is created lazily on first call per tenant and reused thereafter.
    Tenant identity is derived from hermes_constants.get_hermes_home().
    """
    import hermes_constants as hc
    tenant_id = str(hc.get_hermes_home())
    return _build_client(tenant_id)


def get_async_client():
    """Backward-compatible alias for get_client()."""
    return get_client()


def check_api_key() -> bool:
    """Check whether the OpenRouter API key is present."""
    return bool(os.getenv("OPENROUTER_API_KEY"))
