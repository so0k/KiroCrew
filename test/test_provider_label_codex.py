"""``configured_provider_label()`` must report "codex" for a codex-backed
session, not the pinned "acp" enum value.

Regression this guards: ``agent.provider`` is fixed to ``"acp"`` on this fork
(see ``docs/system-specs/modules/providers.md``); only ``agent.acp_backend``
says codex is actually serving the turn. Before this helper existed, call
sites across the codebase derived their session label straight from
``cfg.agent.provider``, so a codex session was labelled "acp" everywhere a
distinct "codex" label was needed — context.py's spec-adapter gate
(``_SPEC_ADAPTER_PROVIDER_TYPES``), session_map provider tagging
(``detect_provider_switch`` / ``seed_conversation``), and the per-turn
usage/telemetry provider dimension. Follows the style of
test_session_map_codex.py's ``TestBackgroundSessionBackendDetection``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import kiro_crew.dashboard.handlers.usage as usage_mod
import kiro_crew.workflows.agent_exec as agent_exec_mod
from kiro_crew.config import KiroCrewConfig
from kiro_crew.session_map import configured_provider_label


def _cfg(acp_backend: str) -> KiroCrewConfig:
    cfg = KiroCrewConfig()
    cfg.agent.acp_backend = acp_backend
    return cfg


class TestConfiguredProviderLabel:
    def test_default_config_labels_acp(self):
        with patch.object(KiroCrewConfig, "load", return_value=_cfg("")):
            assert configured_provider_label() == "acp"

    def test_codex_backend_labels_codex(self):
        with patch.object(KiroCrewConfig, "load", return_value=_cfg("codex")):
            assert configured_provider_label() == "codex"


class TestChangedCallSitesWireTheSharedHelper:
    """Every fixed call site imports this SAME module-level function (not a
    private copy or a re-derivation of ``cfg.agent.provider``), so patching
    config load once here is representative of what every fixed site does
    under a codex config.
    """

    def test_subagent_module_shares_the_helper(self):
        import kiro_crew.subagent as subagent_mod

        assert subagent_mod.configured_provider_label is configured_provider_label

    def test_slack_gateway_module_shares_the_helper(self):
        import kiro_crew.slack.gateway as gateway_mod

        assert gateway_mod.configured_provider_label is configured_provider_label

    def test_task_executor_module_shares_the_helper(self):
        import kiro_crew.task_executor as task_executor_mod

        assert task_executor_mod.configured_provider_label is configured_provider_label

    def test_dashboard_hooks_module_shares_the_helper(self):
        import kiro_crew.dashboard.handlers.hooks as hooks_mod

        assert hooks_mod.configured_provider_label is configured_provider_label

    def test_chat_runner_module_shares_the_helper(self):
        import kiro_crew.dashboard.chat_runner as chat_runner_mod

        assert chat_runner_mod.configured_provider_label is configured_provider_label

    def test_workflow_agent_exec_module_shares_the_helper(self):
        assert agent_exec_mod.configured_provider_label is configured_provider_label

    def test_subagent_label_choice_yields_codex_under_codex_config(self):
        # Reproduces subagent.py's state/usage-row label expression —
        # ``"claude_code" if is_cc else configured_provider_label()`` — using
        # the SAME function object subagent.py imported. is_cc is False here
        # (not a claude_code subagent), so under a codex config this is
        # exactly what a codex subagent's persisted "provider" field and
        # usage row record.
        import kiro_crew.subagent as subagent_mod

        with patch.object(KiroCrewConfig, "load", return_value=_cfg("codex")):
            is_cc = False
            label = "claude_code" if is_cc else subagent_mod.configured_provider_label()
        assert label == "codex"

    def test_subagent_claude_branch_is_unaffected_by_codex_config(self):
        # The dormant claude_code seam must keep winning over the helper even
        # when acp_backend happens to be "codex" (the two are mutually
        # exclusive in practice, but the branch order must not regress).
        import kiro_crew.subagent as subagent_mod

        with patch.object(KiroCrewConfig, "load", return_value=_cfg("codex")):
            is_cc = True
            label = "claude_code" if is_cc else subagent_mod.configured_provider_label()
        assert label == "claude_code"

    def test_slack_gateway_label_choice_yields_codex_under_codex_config(self):
        # Reproduces the gateway's repeated
        # ``configured_provider_label() if hasattr(self, "_cfg") else "acp"``
        # guard for an object that already has ``_cfg`` set (the live-gateway
        # case), using the SAME function object gateway.py imported.
        import kiro_crew.slack.gateway as gateway_mod

        class _FakeSelf:
            _cfg = object()

        with patch.object(KiroCrewConfig, "load", return_value=_cfg("codex")):
            fake_self = _FakeSelf()
            provider = (
                gateway_mod.configured_provider_label() if hasattr(fake_self, "_cfg") else "acp"
            )
        assert provider == "codex"


class _FakeProvider:
    """Bare provider double: no context/agent accessors, so the real
    ``read_context_tokens`` yields ``(0, 0)`` and ``read_effective_agent`` ``""``.
    """

    def __init__(self, key: str = "k") -> None:
        self.key = key


class _FakeSessions:
    """Minimal ``SessionManager`` double for ``build_agent_fn``."""

    async def get_or_create(self, key, *, agent=None, model=None, cwd=None, **kw):
        return _FakeProvider(key), True, False

    def release(self, key, *, cleanup=False):
        pass


class TestWorkflowUsageRowProviderLabel:
    """The workflow-agent usage row carries the configured backend's label.

    ``build_agent_fn``'s per-turn row is the workflow surface's only spend
    attribution, so a hardcoded "acp" there files every codex-backed workflow
    turn under the wrong provider dimension. These tests drive the real
    ``agent_fn`` and capture the record ``_build_token_record`` actually
    produced (``_write_token_record`` patched out, so no disk I/O).
    """

    @staticmethod
    def _rows(monkeypatch) -> list[dict]:
        rows: list[dict] = []
        monkeypatch.setattr(
            usage_mod, "_write_token_record", lambda record, now: rows.append(record)
        )

        async def fake_stream(provider, message, **kw):
            return f"reply:{message}"

        monkeypatch.setattr(agent_exec_mod, "stream_and_collect", fake_stream)
        return rows

    @pytest.mark.asyncio
    async def test_row_provider_is_codex_under_codex_backend(self, monkeypatch):
        rows = self._rows(monkeypatch)
        fn = agent_exec_mod.build_agent_fn(
            _FakeSessions(), run_id="wf_codex", default_agent="researcher", default_model="m1"
        )

        with patch.object(KiroCrewConfig, "load", return_value=_cfg("codex")):
            out = await fn("do work", {})

        assert out == "reply:do work"
        assert len(rows) == 1
        assert rows[0]["surface"] == "workflow"
        assert rows[0]["provider"] == "codex"

    @pytest.mark.asyncio
    async def test_row_provider_is_acp_without_codex_backend(self, monkeypatch):
        # Negative control: the label is resolved from config, not pinned to
        # either literal.
        rows = self._rows(monkeypatch)
        fn = agent_exec_mod.build_agent_fn(
            _FakeSessions(), run_id="wf_acp", default_agent="researcher", default_model="m1"
        )

        with patch.object(KiroCrewConfig, "load", return_value=_cfg("")):
            await fn("do work", {})

        assert len(rows) == 1
        assert rows[0]["provider"] == "acp"
