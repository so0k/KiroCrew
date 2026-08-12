"""Agent-profile parity gap: what a kiro custom agent delivers vs. what a spec
adapter (claude, codex) can natively apply.

``session/set_mode`` is the ONLY mechanism that applies a custom kiro agent's
``tools`` allowlist — see acp-client.md "Custom Agent Support" and the
governance.md scope note ("the underlying kiro-cli agent config
(``~/.kiro/agents/*.json``) is out of scope" for Kiro Crew's own policy/profile
ceiling). A spec adapter never sends ``set_mode`` and reads no kiro agent
config at all (acp-client.md "Backend Selection" — codex dialect), so an agent
whose config deliberately narrows ``tools`` below the adapter's own built-in
capability set has no equivalent restriction applied on that backend.

The persona/prompt, steering resources, skills, model, and MCP servers are
all delivered by other backend-agnostic mechanisms (context.py injection,
``_apply_startup_model``, ``_codex_session_mcp_servers`` /
``_claude_acp_mcp_servers``) and are therefore NOT re-verified here — this
file is scoped to the one residual gap: a custom agent's ``tools`` allowlist,
which Kiro Crew's own hooks gate deliberately does not read either (verified
below), leaving no enforcement point at all for it besides the kiro-only
``set_mode`` guard. ``_spec_adapter_shell_restriction`` /
``AcpClient._assert_spec_adapter_agent_permitted`` polyfill the one concrete,
positively-verifiable case: an agent config that omits ``execute_bash`` from
``tools`` (e.g. the shipped ``auto-improvement-pr-author``, ``"tools": []``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kiro_crew.acp.client as client_mod
import kiro_crew.hooks as hooks_mod
from kiro_crew.acp.client import CLIENT_NAME, AcpClient, AcpError, _spec_adapter_shell_restriction
from kiro_crew.acp.types import ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX
from kiro_crew.agent_files import OWNED_KIRO_AGENT_FILES

_PR_AUTHOR_AGENT = "auto-improvement-pr-author"
_SHIPPED_PR_AUTHOR = (
    Path(__file__).parent.parent
    / "src"
    / "kiro_crew"
    / "apps"
    / "builtins"
    / "auto_improvement"
    / "agents"
    / "pr-author.json"
)


def _write_agent(tmp_path: Path, name: str, data: dict) -> None:
    (tmp_path / f"{name}.json").write_text(json.dumps(data))


class TestSpecAdapterShellRestrictionPureFunction:
    """Unit coverage of the detector itself, independent of AcpClient."""

    def test_default_kirocrew_agent_is_never_restricted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        assert _spec_adapter_shell_restriction(CLIENT_NAME) is None

    def test_empty_agent_name_is_never_restricted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        assert _spec_adapter_shell_restriction("") is None

    def test_missing_agent_config_fails_open_not_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Cannot verify a restriction that isn't there to read -> unchanged
        # (today's) behavior, not a new refusal for an unrelated reason.
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        assert _spec_adapter_shell_restriction("no-such-agent") is None

    def test_malformed_json_fails_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        (tmp_path / "broken.json").write_text("{not json")
        assert _spec_adapter_shell_restriction("broken") is None

    def test_non_dict_json_fails_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        (tmp_path / "listy.json").write_text("[]")
        assert _spec_adapter_shell_restriction("listy") is None

    def test_agent_with_no_tools_key_fails_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "no-tools-key", {"name": "no-tools-key", "model": "auto"})
        assert _spec_adapter_shell_restriction("no-tools-key") is None

    def test_agent_granting_execute_bash_is_not_restricted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(
            tmp_path,
            "discovery-like",
            {"tools": ["fs_read", "fs_write", "execute_bash"], "allowedTools": []},
        )
        assert _spec_adapter_shell_restriction("discovery-like") is None

    def test_agent_omitting_execute_bash_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "pr-author", {"tools": []})
        reason = _spec_adapter_shell_restriction("pr-author")
        assert reason is not None
        assert "execute_bash" in reason

    def test_shipped_pr_author_agent_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Grounds the check in the one concrete shipped example, not a fixture."""
        data = json.loads(_SHIPPED_PR_AUTHOR.read_text())
        assert data.get("tools") == [], "fixture drifted from the shipped pr-author.json"
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, _PR_AUTHOR_AGENT, data)
        assert _spec_adapter_shell_restriction(_PR_AUTHOR_AGENT) is not None

    def test_kirocrew_managed_agents_are_exempt_even_when_shell_less(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Every OWNED_KIRO_AGENT_FILES agent passes despite a tools list that
        omits execute_bash: their shell-less profiles are Kiro Crew's own
        scope/cost choice, and refusing kirocrew-lite would brick the
        background session on every spec-adapter host."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        for filename in OWNED_KIRO_AGENT_FILES:
            name = filename.removesuffix(".json")
            _write_agent(tmp_path, name, {"name": name, "tools": []})
            assert _spec_adapter_shell_restriction(name) is None, name


class TestAssertSpecAdapterAgentPermitted:
    """Wiring: the AcpClient method raises AcpError at the set_mode-equivalent step."""

    @staticmethod
    def _client(agent: str, backend: str) -> AcpClient:
        return AcpClient(agent=agent, acp_backend=backend)

    def test_codex_refuses_a_restricted_custom_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "readonly-agent", {"tools": ["fs_read", "grep"]})
        client = self._client("readonly-agent", ACP_BACKEND_CODEX)
        with pytest.raises(AcpError, match="readonly-agent"):
            client._assert_spec_adapter_agent_permitted()

    def test_claude_also_refuses_a_restricted_custom_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The gap is structural to "no set_mode", not codex-specific: claude
        # skips set_mode too and reads no kiro agent config either.
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "readonly-agent", {"tools": ["fs_read"]})
        client = self._client("readonly-agent", ACP_BACKEND_CLAUDE)
        with pytest.raises(AcpError):
            client._assert_spec_adapter_agent_permitted()

    def test_codex_permits_the_default_kirocrew_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        client = self._client(CLIENT_NAME, ACP_BACKEND_CODEX)
        client._assert_spec_adapter_agent_permitted()  # must not raise

    def test_codex_permits_a_custom_agent_that_grants_execute_bash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "broad-agent", {"tools": ["execute_bash", "fs_read"]})
        client = self._client("broad-agent", ACP_BACKEND_CODEX)
        client._assert_spec_adapter_agent_permitted()  # must not raise

    def test_codex_permits_an_unmaterialized_custom_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No config to read -> cannot verify a restriction -> unchanged
        # (today's, pre-patch) behavior rather than a new blanket refusal.
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        client = self._client("never-materialized", ACP_BACKEND_CODEX)
        client._assert_spec_adapter_agent_permitted()  # must not raise


class TestHooksGateDoesNotReadAgentProfileToolRules:
    """Pins the verify-first finding backing the polyfill's placement.

    Kiro Crew's own PreToolUse gate governs POLICY ∩ PROFILE
    (``kiro_crew.platform.governance``) and never reads
    ``~/.kiro/agents/*.json`` — governance.md is explicit that the kiro agent
    config is out of scope for that ceiling. So a profile-denied tool is NOT
    "already refused" for a codex session by the hooks gate reading the
    agent's own tools list; the gap can only be closed at the acp client
    (this module), which is what ``_assert_spec_adapter_agent_permitted``
    does.
    """

    def test_hook_manager_source_never_references_the_kiro_agents_dir(self) -> None:
        import inspect

        src = inspect.getsource(hooks_mod)
        assert "kiro_agents_dir" not in src
        assert ".kiro/agents" not in src
