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
``_claude_session_mcp_servers``) and are therefore NOT re-verified here — this
file is scoped to the one residual gap: a custom agent's ``tools`` allowlist,
which Kiro Crew's own hooks gate deliberately does not read either (verified
below), leaving no enforcement point at all for it besides the kiro-only
``set_mode`` guard. ``_spec_adapter_shell_restriction`` /
``AcpClient._assert_spec_adapter_agent_permitted`` polyfill the one concrete,
positively-verifiable case: an agent config whose ``tools`` list withholds the
builtin shell tool under every name kiro-cli grants it by (``execute_bash``,
``shell``, and the ``"*"`` wildcard) — e.g. the shipped
``auto-improvement-pr-author``, ``"tools": []``. The config is resolved the way
kiro-cli resolves ``--agent``: ``<project>/.kiro/agents`` shadows the user-level
agents dir and BOTH scopes match on the name a spec is dispatchable under (its
declared ``name`` field, not its filename) — so both are covered below,
including the cases where the resolution has no single answer to mirror: two
specs declaring one name (fail-closed on either twin) and a spec Kiro Crew's own
reader refuses while kiro-cli would still activate it, over the read cap or
through a sensitive symlink target (fail-closed as unverifiable).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import kiro_crew.acp.client as client_mod
import kiro_crew.agent_discovery as agent_discovery
import kiro_crew.hooks as hooks_mod
from kiro_crew.acp.client import (
    CLIENT_NAME,
    AcpClient,
    AcpError,
    _agent_spec_over_read_cap,
    _agent_spec_sensitive_target,
    _spec_adapter_shell_restriction,
)
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


def _write_project_agent(project_dir: Path, filename: str, data: dict) -> None:
    """Write a project-local spec into the dir kiro-cli resolves ``--agent`` against."""
    agents_dir = project_dir / ".kiro" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{filename}.json").write_text(json.dumps(data))


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

    def test_agent_granting_the_shell_alias_is_not_restricted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``shell`` is the other name kiro-cli grants the same builtin under."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "alias-agent", {"tools": ["fs_read", "shell"]})
        assert _spec_adapter_shell_restriction("alias-agent") is None

    def test_agent_granting_the_wildcard_is_not_restricted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``"*"`` grants every tool, shell included — nothing is withheld."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "wildcard-agent", {"tools": ["*"]})
        assert _spec_adapter_shell_restriction("wildcard-agent") is None

    def test_non_string_tools_entries_do_not_grant_shell(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A shell-shaped non-string entry is not a grant: it stays a refusal."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "odd-agent", {"tools": [{"shell": True}, ["*"], None]})
        assert _spec_adapter_shell_restriction("odd-agent") is not None

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


class TestSpecAdapterShellRestrictionProjectScope:
    """The detector consults every dir kiro-cli resolves ``--agent`` against.

    kiro-cli searches ``<cwd>/.kiro/agents`` before the user-level dir
    (``config.paths.project_agents_dir``), and Kiro Crew spawns the backend with
    the session's project dir as cwd — so a project-local spec is dispatchable
    and shadows a same-named user-level one.
    """

    @staticmethod
    def _project(tmp_path: Path) -> Path:
        project = tmp_path / "project"
        project.mkdir()
        return project

    def test_project_local_shell_withholding_agent_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = self._project(tmp_path)
        _write_project_agent(project, "local-readonly", {"tools": ["fs_read"]})
        reason = _spec_adapter_shell_restriction("local-readonly", project)
        assert reason is not None
        assert "execute_bash" in reason

    def test_project_local_agent_granting_shell_is_not_flagged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = self._project(tmp_path)
        _write_project_agent(project, "local-broad", {"tools": ["shell", "fs_read"]})
        assert _spec_adapter_shell_restriction("local-broad", project) is None

    def test_project_local_agent_is_matched_on_its_declared_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``--agent`` accepts the declared ``name``, which wins over the filename."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = self._project(tmp_path)
        _write_project_agent(project, "file-stem", {"name": "declared-name", "tools": []})
        assert _spec_adapter_shell_restriction("declared-name", project) is not None

    def test_project_local_spec_shadows_a_permissive_user_level_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        project = self._project(tmp_path)
        _write_agent(user_dir, "shadowed", {"tools": ["execute_bash"]})
        _write_project_agent(project, "shadowed", {"tools": []})
        assert _spec_adapter_shell_restriction("shadowed", project) is not None

    def test_user_level_spec_is_still_read_when_the_project_has_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        project = self._project(tmp_path)
        _write_agent(user_dir, "user-only", {"tools": []})
        assert _spec_adapter_shell_restriction("user-only", project) is not None
        _write_agent(user_dir, "user-broad", {"tools": ["execute_bash"]})
        assert _spec_adapter_shell_restriction("user-broad", project) is None

    def test_a_project_without_any_agents_dir_is_harmless(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        assert _spec_adapter_shell_restriction("nobody", self._project(tmp_path)) is None

    @pytest.mark.parametrize("body", ["{not json", "[]", '"a string"'])
    def test_an_undispatchable_project_file_does_not_shadow_the_user_level_spec(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
    ) -> None:
        """A project file the loader cannot dispatch must not suppress the check.

        ``agent_discovery`` builds the project dispatch set only from specs that
        parse as a JSON object, so kiro-cli cannot activate a malformed or
        non-object file and ``--agent`` still resolves to the user-level spec.
        Treating the broken file as the resolved config would return "nothing to
        verify" and permit a session the user-level profile refuses.
        """
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        project = self._project(tmp_path)
        _write_agent(user_dir, "pr-author", {"tools": []})
        agents_dir = project / ".kiro" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "pr-author.json").write_text(body)
        reason = _spec_adapter_shell_restriction("pr-author", project)
        assert reason is not None
        assert str(user_dir) in reason

    def test_an_unreadable_project_file_does_not_shadow_the_user_level_spec(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Same for a file the hardened reader cannot read at all.

        A spec the reader rejects is absent from ``agent_discovery``'s project
        dispatch set, so kiro-cli cannot activate it and ``--agent`` resolves to
        the user-level spec — which the guard must then verify rather than
        reporting nothing to check.
        """
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        project = self._project(tmp_path)
        _write_agent(user_dir, "pr-author", {"tools": []})
        _write_project_agent(project, "pr-author", {"tools": ["*"]})
        real_read = agent_discovery.safe_read_file_bytes

        def unreadable(raw: str) -> bytes | None:
            return None if str(project) in raw else real_read(raw)

        monkeypatch.setattr(agent_discovery, "safe_read_file_bytes", unreadable)
        reason = _spec_adapter_shell_restriction("pr-author", project)
        assert reason is not None
        assert str(user_dir) in reason

    def test_duplicate_declared_names_refuse_on_the_restrictive_twin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two same-dir specs declaring one name: any withholding twin refuses.

        Within the project scope ``agent_discovery.list_agents`` overwrites
        ``seen[name]`` as it walks the stem-sorted files, so its winner is
        whichever file happens to sort last — an artifact of the sort, not a
        resolution rule kiro-cli documents. Both files are dispatchable under
        the name, so a permissive twin must not mask the restrictive one
        whichever way the stems sort.
        """
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        for permissive_stem, restrictive_stem in (("aaa", "zzz"), ("zzz", "aaa")):
            project = tmp_path / f"proj-{permissive_stem}"
            project.mkdir()
            _write_project_agent(project, permissive_stem, {"name": "dup", "tools": ["*"]})
            _write_project_agent(project, restrictive_stem, {"name": "dup", "tools": []})
            reason = _spec_adapter_shell_restriction("dup", project)
            assert reason is not None, permissive_stem
            assert restrictive_stem in reason

    def test_duplicate_declared_names_all_granting_shell_are_permitted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The tie-break is fail-closed, not a ban on duplicate names."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = self._project(tmp_path)
        _write_project_agent(project, "aaa", {"name": "dup", "tools": ["*"]})
        _write_project_agent(project, "zzz", {"name": "dup", "tools": ["shell"]})
        assert _spec_adapter_shell_restriction("dup", project) is None

    def test_project_specs_are_read_through_the_hardened_reader(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No unbounded second read: the parsed spec comes from that reader.

        ``project_dir`` derives from a caller-supplied session field and this
        runs on the event loop during the handshake, so the guard must not
        re-read the file outside the size / sensitive-symlink-target gate.
        """
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = self._project(tmp_path)
        _write_project_agent(project, "local-readonly", {"tools": ["fs_read"]})
        reads: list[str] = []
        real_read = agent_discovery.safe_read_file_bytes

        def counting(raw: str) -> bytes | None:
            reads.append(raw)
            return real_read(raw)

        monkeypatch.setattr(agent_discovery, "safe_read_file_bytes", counting)
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda *a, **kw: pytest.fail("guard re-read a spec outside the hardened reader"),
        )
        assert _spec_adapter_shell_restriction("local-readonly", project) is not None
        assert len(reads) == 1


class TestSpecAdapterShellRestrictionUserScopeNameField:
    """The user scope resolves by the JSON ``name`` field, not by the filename.

    kiro-cli dispatches a user-level agent by its declared ``name``
    (``agent_discovery._global_agent_info`` names every row
    ``spec_str(data, "name", f.stem)``), while the filenames in that dir are
    namespaced by whoever materialized them: an app writes
    ``<app>--<agent>.json`` (``apps.bridges._safe_link_name``) and a package
    installs ``<Pkg>-<name>.json``. A stem-only lookup therefore reads every
    app- and package-installed restricted profile as absent — a silent shell
    grant on a spec adapter for exactly the agents an operator did not
    hand-write.
    """

    def test_app_namespaced_spec_is_matched_on_its_declared_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``myapp--restricted.json`` declaring ``restricted`` is dispatchable as it."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "myapp--restricted", {"name": "restricted", "tools": []})
        reason = _spec_adapter_shell_restriction("restricted")
        assert reason is not None
        assert "myapp--restricted.json" in reason

    def test_package_namespaced_spec_is_matched_on_its_declared_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "SomePkg-gpu-dev", {"name": "gpu-dev", "tools": ["fs_read"]})
        assert _spec_adapter_shell_restriction("gpu-dev") is not None

    def test_a_namespaced_spec_declaring_another_name_is_not_matched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A coincidental filename suffix must not refuse an unrelated agent.

        The ``name`` field is authoritative, so a file whose filename ends in
        the requested name while declaring a different one is not dispatchable
        under it and carries no restriction to honor.
        """
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "myapp--restricted", {"name": "other-agent", "tools": []})
        assert _spec_adapter_shell_restriction("restricted") is None

    def test_a_withholding_namespaced_twin_refuses_past_a_permissive_bare_spec(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two files, one name: the fail-closed tie-break spans the whole dir.

        ``list_agents`` picks between them on package preference and first-seen
        order, neither of which kiro-cli documents as a resolution rule, so a
        permissive bare ``<agent>.json`` must not mask a restricted app twin.
        """
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "restricted", {"tools": ["*"]})
        _write_agent(tmp_path, "myapp--restricted", {"name": "restricted", "tools": []})
        reason = _spec_adapter_shell_restriction("restricted")
        assert reason is not None
        assert "myapp--restricted.json" in reason

    def test_an_unparseable_namespaced_candidate_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """kiro-cli cannot activate it either, so there is no restriction to honor."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        (tmp_path / "myapp--restricted.json").write_text("{not json")
        assert _spec_adapter_shell_restriction("restricted") is None
        _write_agent(tmp_path, "restricted", {"tools": ["execute_bash"]})
        assert _spec_adapter_shell_restriction("restricted") is None

    def test_a_project_spec_still_shadows_a_namespaced_user_level_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Scope order is unchanged by the name-field match in the user dir."""
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        project = tmp_path / "project"
        project.mkdir()
        _write_agent(user_dir, "myapp--shadowed", {"name": "shadowed", "tools": []})
        _write_project_agent(project, "shadowed", {"tools": ["execute_bash"]})
        assert _spec_adapter_shell_restriction("shadowed", project) is None

    def test_an_agent_name_carrying_glob_metacharacters_still_resolves(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The exact filename is inspected directly, not only through the scan.

        A name with glob metacharacters cannot be matched literally by the
        namespaced scan (and makes an invalid pattern on some Python versions),
        so the exact ``<agent>.json`` lookup is what keeps it verifiable.
        """
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        try:
            _write_agent(tmp_path, "**", {"tools": []})
        except OSError:
            pytest.skip("filesystem rejects glob metacharacters in a filename")
        assert _spec_adapter_shell_restriction("**") is not None


class TestSpecAdapterShellRestrictionUnverifiableSpecs:
    """A reader refusal kiro-cli does NOT share refuses instead of permitting.

    ``agent_discovery._read_agent_spec`` rejections split two ways. Most of them
    also stop kiro-cli from turning the file into a mode (unparseable JSON, a
    broken link), so skipping the file costs no evidence. Two are Kiro Crew's own
    reader policy while kiro-cli would still activate the spec and honor its
    ``tools`` list: the ``hooks.MAX_FILE_BYTES`` cap (kiro-cli caps nothing) and
    a symlink whose resolved target is sensitive (kiro-cli filters nothing).
    Those refuse as unverifiable.
    """

    def test_over_cap_predicate_is_stat_based(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Asking the question must not perform the read the cap prevents."""
        spec = tmp_path / "big.json"
        spec.write_text(json.dumps({"tools": []}))
        monkeypatch.setattr(
            Path, "read_text", lambda *a, **kw: pytest.fail("size check read the file")
        )
        monkeypatch.setattr(client_mod, "MAX_FILE_BYTES", 2)
        assert _agent_spec_over_read_cap(spec) is True
        monkeypatch.setattr(client_mod, "MAX_FILE_BYTES", 1024 * 1024)
        assert _agent_spec_over_read_cap(spec) is False
        assert _agent_spec_over_read_cap(tmp_path / "absent.json") is False

    def test_over_cap_user_level_spec_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "huge", {"tools": ["execute_bash"]})
        monkeypatch.setattr(client_mod, "MAX_FILE_BYTES", 2)
        reason = _spec_adapter_shell_restriction("huge")
        assert reason is not None
        assert "cannot be verified" in reason

    def test_over_cap_project_spec_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The declared name lives in the body the cap forbids reading."""
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        project = tmp_path / "project"
        project.mkdir()
        _write_project_agent(project, "huge-local", {"tools": ["execute_bash"]})
        monkeypatch.setattr(
            client_mod, "_agent_spec_over_read_cap", lambda p: str(project) in str(p)
        )
        reason = _spec_adapter_shell_restriction("huge-local", project)
        assert reason is not None
        assert "cannot be verified" in reason

    def test_over_cap_project_spec_refuses_whatever_its_filename(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An unrelated-LOOKING oversized project file still refuses.

        Every file in the project agents dir is a candidate (dispatch is on the
        declared ``name``), and the cap forbids reading the body that would say
        which name this one claims — so it may be dispatchable as the requested
        agent and refuses rather than being dropped on its stem.
        """
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        project = tmp_path / "project"
        project.mkdir()
        _write_project_agent(project, "unrelated", {"tools": []})
        _write_agent(user_dir, "wanted", {"tools": ["execute_bash"]})
        monkeypatch.setattr(
            client_mod, "_agent_spec_over_read_cap", lambda p: p.name == "unrelated.json"
        )
        reason = _spec_adapter_shell_restriction("wanted", project)
        assert reason is not None
        assert "unrelated.json" in reason
        assert "cannot be verified" in reason

    def test_over_cap_user_spec_outside_the_narrowed_scan_is_not_a_candidate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The user scope refuses only files it actually inspects.

        Its candidate set is ``<agent>.json`` plus the ``*<agent>.json`` matches,
        so a filename that cannot be dispatchable under the requested name is
        never stat-ed for the cap and cannot refuse.
        """
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        _write_agent(user_dir, "zebra", {"tools": []})
        _write_agent(user_dir, "wanted", {"tools": ["execute_bash"]})
        monkeypatch.setattr(
            client_mod, "_agent_spec_over_read_cap", lambda p: p.name == "zebra.json"
        )
        assert _spec_adapter_shell_restriction("wanted") is None

    def test_over_cap_namespaced_user_spec_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An app-namespaced spec over the cap cannot be dropped on its stem.

        ``app--foo.json`` is dispatchable as ``foo``, and the cap forbids reading
        the ``name`` field that says so — matching on the stem would silently
        drop it and grant the adapter's shell.
        """
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "app--foo", {"name": "foo", "tools": []})
        monkeypatch.setattr(client_mod, "MAX_FILE_BYTES", 2)
        reason = _spec_adapter_shell_restriction("foo")
        assert reason is not None
        assert "app--foo.json" in reason
        assert "cannot be verified" in reason

    @staticmethod
    def _plant_sensitive_symlink(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agents_dir: Path, filename: str
    ) -> Path:
        """Link ``agents_dir/filename`` at a shell-withholding spec under a sensitive tree.

        The sensitive tree is faked the way ``test_agent_discovery`` fakes it —
        by pointing the reader's own ``is_sensitive_path`` at a tmp dir — so the
        test exercises the real resolve-then-classify path without needing a
        writable ``~/.aws``.
        """
        secret_dir = tmp_path / "fake-sensitive"
        secret_dir.mkdir(exist_ok=True)
        target = secret_dir / "planted.json"
        target.write_text(json.dumps({"tools": []}))
        agents_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(target, agents_dir / filename)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform")
        monkeypatch.setattr(
            agent_discovery,
            "is_sensitive_path",
            lambda p: str(p).startswith(str(secret_dir)),
        )
        return secret_dir

    def test_sensitive_target_predicate_resolves_without_reading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Classifying the link must not open the protected file."""
        agents_dir = tmp_path / "agents"
        self._plant_sensitive_symlink(monkeypatch, tmp_path, agents_dir, "evil.json")
        monkeypatch.setattr(
            Path, "read_text", lambda *a, **kw: pytest.fail("sensitive check read the target")
        )
        assert _agent_spec_sensitive_target(agents_dir / "evil.json") is True
        assert _agent_spec_sensitive_target(agents_dir / "absent.json") is False

    def test_sensitive_target_project_spec_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """kiro-cli applies no sensitive-path filter, so the spec still binds there."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = tmp_path / "project"
        secret_dir = self._plant_sensitive_symlink(
            monkeypatch, tmp_path, project / ".kiro" / "agents", "planted-agent.json"
        )
        real_read = agent_discovery.safe_read_file_bytes

        def guarded(raw: str) -> bytes | None:
            assert str(secret_dir) not in raw, "guard read a sensitive agent-spec target"
            return real_read(raw)

        monkeypatch.setattr(agent_discovery, "safe_read_file_bytes", guarded)
        reason = _spec_adapter_shell_restriction("planted-agent", project)
        assert reason is not None
        assert "sensitive" in reason
        assert "cannot be verified" in reason

    def test_sensitive_target_user_level_spec_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        user_dir = tmp_path / "user"
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        self._plant_sensitive_symlink(monkeypatch, tmp_path, user_dir, "planted-agent.json")
        reason = _spec_adapter_shell_restriction("planted-agent")
        assert reason is not None
        assert "cannot be verified" in reason

    def test_sensitive_target_project_spec_refuses_whatever_its_filename(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An unrelated-LOOKING planted link in the project dir still refuses.

        The link's stem says nothing about the ``name`` it is dispatchable
        under, and reading that name is exactly what the sensitive target
        forbids — so a planted link cannot buy a permit by being renamed.
        """
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        project = tmp_path / "project"
        self._plant_sensitive_symlink(
            monkeypatch, tmp_path, project / ".kiro" / "agents", "unrelated.json"
        )
        _write_agent(user_dir, "wanted", {"tools": ["execute_bash"]})
        reason = _spec_adapter_shell_restriction("wanted", project)
        assert reason is not None
        assert "unrelated.json" in reason
        assert "sensitive" in reason

    def test_sensitive_target_user_spec_outside_the_narrowed_scan_is_not_a_candidate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A planted link the user scope never inspects cannot refuse."""
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        self._plant_sensitive_symlink(monkeypatch, tmp_path, user_dir, "zebra.json")
        _write_agent(user_dir, "wanted", {"tools": ["execute_bash"]})
        assert _spec_adapter_shell_restriction("wanted") is None

    def test_sensitive_target_namespaced_user_spec_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``app--foo.json`` linked at a sensitive spec refuses agent ``foo``.

        The planted target withholds shell, so kiro-cli would activate it and
        honor that restriction while this reader refuses to read it — the file is
        unverifiable rather than absent, and its stem is no reason to drop it.
        """
        user_dir = tmp_path / "user"
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: user_dir)
        self._plant_sensitive_symlink(monkeypatch, tmp_path, user_dir, "app--foo.json")
        reason = _spec_adapter_shell_restriction("foo")
        assert reason is not None
        assert "app--foo.json" in reason
        assert "cannot be verified" in reason


class TestAssertSpecAdapterAgentPermitted:
    """Wiring: the AcpClient method raises AcpError at the set_mode-equivalent step."""

    @staticmethod
    def _client(agent: str, backend: str, work_dir: Path | None = None) -> AcpClient:
        return AcpClient(agent=agent, acp_backend=backend, work_dir=work_dir)

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

    def test_codex_permits_a_custom_agent_granting_the_wildcard(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        _write_agent(tmp_path, "wildcard-agent", {"tools": ["*"]})
        client = self._client("wildcard-agent", ACP_BACKEND_CODEX)
        client._assert_spec_adapter_agent_permitted()  # must not raise

    def test_codex_refuses_a_project_local_restricted_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The session's work dir is the cwd kiro-cli would resolve ``--agent`` in."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = tmp_path / "project"
        project.mkdir()
        _write_project_agent(project, "local-readonly", {"tools": ["fs_read"]})
        client = self._client("local-readonly", ACP_BACKEND_CODEX, work_dir=project)
        with pytest.raises(AcpError, match="local-readonly"):
            client._assert_spec_adapter_agent_permitted()

    def test_codex_permits_an_unmaterialized_custom_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No config to read -> cannot verify a restriction -> unchanged
        # (today's, pre-patch) behavior rather than a new blanket refusal.
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path)
        client = self._client("never-materialized", ACP_BACKEND_CODEX)
        client._assert_spec_adapter_agent_permitted()  # must not raise


class TestGuardRunsOffTheEventLoop:
    """The handshake awaits the guard through ``asyncio.to_thread``.

    The guard is blocking filesystem work — the project scope reads every spec
    in ``<work_dir>/.kiro/agents`` and each read goes through
    ``hooks.safe_read_file_bytes``, whose per-file cap still admits a large read
    (and whose ``open`` blocks outright on a planted FIFO). Running it inline in
    ``_initialize_session`` would put that on the event loop, stalling every
    other session's frames; offloading keeps the cost on this handshake alone
    while the ``AcpError`` still propagates.
    """

    @staticmethod
    def _handshake_client(agent: str, work_dir: Path) -> AcpClient:
        """A codex client wired to reach step 4 of the handshake and stop there."""
        client = AcpClient(work_dir=work_dir, agent=agent, acp_backend=ACP_BACKEND_CODEX)
        proc = MagicMock()
        proc.returncode = None
        proc.stdin = MagicMock()
        proc.stdin.drain = AsyncMock()
        client._process = proc
        client._next_req_id = MagicMock(side_effect=range(1, 100))
        client._wait_for_response = AsyncMock(
            return_value={"protocolVersion": 1, "agentCapabilities": {}}
        )
        client._new_session_following_substitution = AsyncMock(
            return_value={"sessionId": "sess-guard"}
        )
        client._apply_startup_model = AsyncMock()
        client._drain_notifications = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_handshake_refusal_surfaces_through_the_offload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The worker thread must not swallow the refusal."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = tmp_path / "project"
        project.mkdir()
        _write_project_agent(project, "local-readonly", {"tools": ["fs_read"]})
        client = self._handshake_client("local-readonly", project)
        with pytest.raises(AcpError, match="local-readonly"):
            await client._initialize_session()

    @pytest.mark.asyncio
    async def test_handshake_reads_no_agent_spec_on_the_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Every spec read the guard performs happens in a worker thread."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        project = tmp_path / "project"
        project.mkdir()
        _write_project_agent(project, "broad-local", {"tools": ["execute_bash"]})
        threads: list[str] = []
        real_read = agent_discovery.safe_read_file_bytes

        def recording(raw: str) -> bytes | None:
            threads.append(threading.current_thread().name)
            return real_read(raw)

        monkeypatch.setattr(agent_discovery, "safe_read_file_bytes", recording)
        client = self._handshake_client("broad-local", project)
        await client._initialize_session()
        assert threads, "the guard read no spec, so the offload is unproven"
        main = threading.main_thread().name
        assert all(name != main for name in threads), threads

    @pytest.mark.asyncio
    async def test_the_guard_itself_is_what_gets_offloaded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pins the seam, so an inlined call is caught even with no spec to read."""
        monkeypatch.setattr(client_mod, "kiro_agents_dir", lambda: tmp_path / "user")
        offloaded: list[str] = []
        real_to_thread = asyncio.to_thread

        async def recording(func, /, *args, **kwargs):
            offloaded.append(getattr(func, "__name__", ""))
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", recording)
        client = self._handshake_client("never-materialized", tmp_path / "project")
        await client._initialize_session()
        assert "_assert_spec_adapter_agent_permitted" in offloaded


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
