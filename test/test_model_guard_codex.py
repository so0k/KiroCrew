"""Regression test: ``_model_rejected_reason`` needs no codex exemption.

Verifies (see the task-spec investigation) that the canonical-key registry
(``model_registry.json``) holds only Claude/Fable-shaped keys plus ``auto`` —
no ``gpt-*``/openai-shaped entry — so ``model_registry.is_canonical_key``
never fires on a codex-advertised model id such as ``gpt-5.6-sol`` or its
effort-suffixed siblings. Effort travels on its own channel — the ``effort``
config option live, ``effort.effort_settings_key`` on disk — so no
effort-bearing spelling is a registry key either, including codex-acp's
``<model>[<effort>]`` display form (which ``codex_base_model_id`` splits before
the wire; see test_acp_backend_codex). Because ``agent.provider`` is pinned to
``"acp"`` on this fork regardless of ``agent.acp_backend`` (kiro vs. codex), the guard's
existing ``claude_code`` exemption is dormant for codex and none is needed:
a real codex model id is never a registry key, so it is never rejected.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kiro_crew import model_registry
from kiro_crew.dashboard import chat_handlers


def _mock_cfg(*, provider: str = "acp", acp_backend: str = "codex") -> MagicMock:
    cfg = MagicMock()
    cfg.agent = SimpleNamespace(provider=provider, acp_backend=acp_backend)
    return cfg


class TestIsCanonicalKeyHasNoCodexOverlap:
    """Ground truth for the verdict: enumerate the canonical key set."""

    def test_registry_has_no_gpt_or_openai_shaped_key(self) -> None:
        # model_registry.is_canonical_key(name) is `name in _REGISTRY`, and
        # _REGISTRY is loaded verbatim from model_registry.json's top-level
        # keys. None of them look like a codex-advertised id.
        for key in model_registry._REGISTRY:
            assert not key.startswith("gpt-"), key
            assert "openai" not in key, key

    @pytest.mark.parametrize(
        "codex_id",
        [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            # Effort is set via a separate cli.json key (effort.py), never
            # folded into the model id — but pin the shape stays non-canonical
            # even if that ever changed.
            "gpt-5.6-sol-high",
        ],
    )
    def test_codex_style_ids_are_not_canonical_keys(self, codex_id: str) -> None:
        assert model_registry.is_canonical_key(codex_id) is False


class TestModelRejectedReasonCodex:
    """``_model_rejected_reason`` under the fork's codex backend."""

    @pytest.mark.parametrize("codex_id", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    def test_codex_model_ids_pass_the_guard(self, monkeypatch, codex_id: str) -> None:
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load",
            lambda: _mock_cfg(provider="acp", acp_backend="codex"),
        )
        assert chat_handlers._model_rejected_reason(codex_id) is None

    def test_canonical_key_is_still_rejected_under_codex_backend(self, monkeypatch) -> None:
        # A display-only canonical key (e.g. from the dropdown fallback) must
        # still be rejected even when the configured backend is codex —
        # `agent.provider` (not `acp_backend`) drives the exemption, and this
        # fork pins provider to "acp", so canonical keys stay rejected.
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load",
            lambda: _mock_cfg(provider="acp", acp_backend="codex"),
        )
        reason = chat_handlers._model_rejected_reason("opus-4.8-1m")
        assert reason is not None
        assert "opus-4.8-1m" in reason

    def test_auto_and_empty_always_pass_under_codex_backend(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load",
            lambda: _mock_cfg(provider="acp", acp_backend="codex"),
        )
        assert chat_handlers._model_rejected_reason("") is None
        assert chat_handlers._model_rejected_reason("auto") is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
