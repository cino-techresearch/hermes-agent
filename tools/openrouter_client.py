"""Shared OpenRouter API client for Hermes tools.

Provides a single lazy-initialized AsyncOpenAI client that all tool modules
can share.  Routes through the centralized provider router in
agent/auxiliary_client.py so auth, headers, and API format are handled
consistently.
"""

import functools
import os

from hermes_constants import get_hermes_home


@functools.lru_cache(maxsize=128)
def _build_async_client(tenant_id: str):
    """Build (and cache) an async client for *tenant_id*.

    Multitenant (model 2 / library-import): keying by ``str(get_hermes_home())``
    isolates clients across tenants that share one process with different
    per-task HERMES_HOME values, instead of a single process-global ``_client``
    that would leak one tenant's connection/auth to another.
    """
    from agent.auxiliary_client import resolve_provider_client
    client, _model = resolve_provider_client("openrouter", async_mode=True)
    if client is None:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return client


def get_async_client():
    """Return a per-tenant async OpenAI-compatible client for OpenRouter.

    Lazily built and cached per active HERMES_HOME (tenant). Uses the centralized
    provider router for auth and client construction.
    Raises ValueError if OPENROUTER_API_KEY is not set.
    """
    return _build_async_client(str(get_hermes_home()))


def check_api_key() -> bool:
    """Check whether the OpenRouter API key is present."""
    return bool(os.getenv("OPENROUTER_API_KEY"))
