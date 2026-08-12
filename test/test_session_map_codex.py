"""SessionMap must treat a codex mapping like a claude one: its transcripts live
in the adapter's own SDK store, not the kiro-cli sessions directory, so get()
must return the sid without a kiro-file check and prune() must not sweep it.

Regression: the resume/prune paths originally special-cased only "claude_code",
so a codex session (labelled "codex" by _provider_map_label) fell through to the
kiro-dir existence check, was never found, and was silently removed — breaking
codex session resume on every reopen and pruning it on every periodic sweep.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kiro_crew.session_map import SessionMap


@pytest.fixture()
def session_map(tmp_path):
    with patch("kiro_crew.session_map.config_dir", return_value=tmp_path):
        yield SessionMap()


class TestCodexIsSdkManaged:
    def test_get_returns_codex_sid_without_a_kiro_file(self, session_map):
        # No kiro-cli transcript exists (codex stores its own) — get() must still
        # resolve the mapping rather than delete it.
        session_map.set("dash:1", "sid-codex", provider="codex")
        assert session_map.get("dash:1") == "sid-codex"
        # And the entry survives the lookup (was not pruned as stale).
        assert session_map.get_provider("dash:1") == "codex"

    def test_prune_keeps_codex_entries(self, session_map):
        session_map.set("dash:codex", "sid-codex", provider="codex")
        session_map.set("dash:claude", "sid-claude", provider="claude_code")
        removed = session_map.prune()
        assert removed == 0
        assert session_map.get("dash:codex") == "sid-codex"
        assert session_map.get("dash:claude") == "sid-claude"

    def test_kiro_entry_without_file_is_still_pruned(self, session_map):
        # The kiro path is unchanged: an "acp" mapping with no transcript on disk
        # is genuinely stale and must still be swept.
        session_map.set("dash:kiro", "sid-kiro", provider="acp")
        assert session_map.prune() == 1
        assert session_map.get("dash:kiro") is None


class TestBackgroundSessionBackendDetection:
    """_bg_provider_is_kiro keys off agent.acp_backend, not the fixed-enum
    agent.provider — so codex routes background one-liners off the kiro-only
    AcpRuntime (which would raise on a codex-only host with no kiro-cli)."""

    def _mgr(self, acp_backend: str):
        from kiro_crew.config import KiroCrewConfig
        from kiro_crew.session import SessionManager

        cfg = KiroCrewConfig()
        cfg.agent.acp_backend = acp_backend
        return SessionManager(cfg, provider_factory=lambda **_: None)

    def test_kiro_backend_is_kiro(self):
        assert self._mgr("")._bg_provider_is_kiro() is True

    def test_codex_backend_is_not_kiro(self):
        assert self._mgr("codex")._bg_provider_is_kiro() is False
