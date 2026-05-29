"""Multitenant (model 2 / library-import) cross-tenant leak regression.

Mirrors saas2 PoC v2: a module-level ``DEFAULT_DB_PATH`` frozen at import time
leaks across tenants when ``HERMES_HOME`` changes per request. The PEP 562
``__getattr__`` lazy resolution + ``SessionDB.__init__`` calling
``get_hermes_home()`` at instantiation must make each tenant resolve to its own
state.db path.

The per-task home override primitive (``_HERMES_HOME_OVERRIDE`` ContextVar) is
provided by upstream ``hermes_constants``; this test guards the fork's residual
(lazy path) on top of it.
"""
import hermes_constants
import hermes_state


def test_default_db_path_lazy_per_tenant(tmp_path):
    a = tmp_path / "tenantA"
    b = tmp_path / "tenantB"

    tok = hermes_constants.set_hermes_home_override(str(a))
    try:
        assert hermes_state.DEFAULT_DB_PATH == a / "state.db"
    finally:
        hermes_constants.reset_hermes_home_override(tok)

    tok = hermes_constants.set_hermes_home_override(str(b))
    try:
        assert hermes_state.DEFAULT_DB_PATH == b / "state.db"
    finally:
        hermes_constants.reset_hermes_home_override(tok)


def test_sessiondb_default_honors_active_override(tmp_path):
    a = tmp_path / "tenantA"
    tok = hermes_constants.set_hermes_home_override(str(a))
    try:
        db = hermes_state.SessionDB()
        assert db.db_path == a / "state.db"
    finally:
        hermes_constants.reset_hermes_home_override(tok)


def test_from_import_binding_is_not_used_in_repo():
    """check_session_search_requirements must use attribute access, not a frozen
    ``from hermes_state import DEFAULT_DB_PATH`` binding (which would not track
    the per-task override)."""
    import inspect

    from tools import session_search_tool

    src = inspect.getsource(session_search_tool.check_session_search_requirements)
    assert "from hermes_state import DEFAULT_DB_PATH" not in src
    assert "hermes_state.DEFAULT_DB_PATH" in src
