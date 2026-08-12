"""Tests for the codex ACP backend's context/steering/skills injection parity.

The codex backend (``provider_type == "codex"``) runs the npm ``codex-acp``
adapter instead of kiro-cli, so — exactly like the claude_code seam — it never
gets an agent's steering/skill ``resources`` loaded natively via
``kiro-cli --agent``. ``ContextBuilder`` must inject them itself for codex,
the same way it already does for claude_code, while a kiro session
(``provider_type`` "acp" or "") keeps deferring to kiro-cli's native load.

Mirrors ``test/test_context.py::TestLoadSteeringResources`` (steering) and
``test/test_agent_template_skills.py::TestSessionContextGate`` (mapped skill
globs), extended to cover the codex provider_type alongside claude_code/acp.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kiro_crew.context import ContextBuilder, _skills_injection_plan
from kiro_crew.learn import LessonStore
from kiro_crew.memory import MemoryStore
from kiro_crew.skills import SkillsLoader


def _builder(tmp_path: Path, skills_root: Path | None = None) -> ContextBuilder:
    return ContextBuilder(
        memory=MemoryStore(workspace=tmp_path / "ws"),
        skills=SkillsLoader(skills_path=skills_root or tmp_path / "skills", install_builtins=False),
        lessons=LessonStore(base_dir=tmp_path),
    )


def _make_skill(root: Path, name: str, *, desc: str = "") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    md = d / "SKILL.md"
    md.write_text(
        f"---\nname: {name}\ndescription: {desc or name + ' skill'}\n---\n\nBody of {name}\n",
        encoding="utf-8",
    )
    return md


class TestSkillsInjectionPlanCodex:
    """Unit coverage of the shared gate helper itself."""

    def test_mapped_agent_injects_only_for_spec_adapters(self):
        # A mapped agent (non-empty globs) is spec-adapter-gated: codex and
        # claude_code inject, kiro (is_spec_adapter=False) defers to the
        # native --agent load.
        with patch("kiro_crew.context.agent_skill_globs", return_value=["*/SKILL.md"]):
            inject_spec, _ = _skills_injection_plan("specialist", is_spec_adapter=True)
            inject_kiro, _ = _skills_injection_plan("specialist", is_spec_adapter=False)
        assert inject_spec is True
        assert inject_kiro is False

    def test_unmapped_kirocrew_ignores_the_spec_adapter_flag(self):
        # No mapping (agent=None) -> the whole catalog regardless of provider.
        inject, globs = _skills_injection_plan(None, is_spec_adapter=False)
        assert inject is True
        assert globs == []


class TestSteeringInjectedForCodex:
    """Mirrors TestLoadSteeringResources.test_steering_injected_for_cc_but_not_acp."""

    def _write_steering(self, tmp_path: Path) -> None:
        steering_dir = tmp_path / ".kiro" / "steering"
        steering_dir.mkdir(parents=True)
        (steering_dir / "rules.md").write_text("# My Rules\nSTEERING_MARKER_XYZ")
        agents_dir = tmp_path / ".kiro" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "kirocrew.json").write_text(
            json.dumps({"resources": ["file://.kiro/steering/**/*.md"]})
        )

    def test_steering_injected_for_codex(self, tmp_path):
        """codex-acp does not read agent ``resources`` either, so it needs the
        same explicit steering load as claude_code."""
        self._write_steering(tmp_path)
        builder = _builder(tmp_path)

        with patch("pathlib.Path.home", return_value=tmp_path):
            codex_ctx = builder.build_session_context(provider_type="codex")

        assert "STEERING_MARKER_XYZ" in codex_ctx

    @pytest.mark.parametrize("provider_type", ["acp", ""])
    def test_steering_not_reinjected_for_kiro(self, tmp_path, provider_type):
        """kiro (provider_type "acp"/"") is unchanged: kiro-cli --agent loads
        the same resources natively, so re-injecting here would duplicate them."""
        self._write_steering(tmp_path)
        builder = _builder(tmp_path)

        with patch("pathlib.Path.home", return_value=tmp_path):
            kiro_ctx = builder.build_session_context(provider_type=provider_type)

        assert "STEERING_MARKER_XYZ" not in kiro_ctx

    def test_codex_and_claude_code_get_the_same_steering_treatment(self, tmp_path):
        self._write_steering(tmp_path)
        builder = _builder(tmp_path)

        with patch("pathlib.Path.home", return_value=tmp_path):
            codex_ctx = builder.build_session_context(provider_type="codex")
            cc_ctx = builder.build_session_context(provider_type="claude_code")

        assert "STEERING_MARKER_XYZ" in codex_ctx
        assert "STEERING_MARKER_XYZ" in cc_ctx


class TestMappedSkillsOnCodex:
    """Mirrors test_agent_template_skills.py::TestSessionContextGate for codex."""

    @pytest.fixture(autouse=True)
    def _fake_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(
            "kiro_crew.agent_discovery._KIRO_AGENTS_DIR", tmp_path / ".kiro" / "agents"
        )
        return tmp_path

    def _agents_dir(self, home: Path) -> Path:
        d = home / ".kiro" / "agents"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_mapped_custom_agent_gets_its_skills_on_codex(self, tmp_path):
        """A custom agent with a skill:// mapping gets exactly the mapped set
        on codex, the same way it already does on claude_code."""
        skills_root = tmp_path / "skills"
        _make_skill(skills_root, "alpha")
        _make_skill(skills_root, "beta")
        d = self._agents_dir(tmp_path)
        (d / "specialist.json").write_text(
            json.dumps(
                {
                    "name": "specialist",
                    "resources": [f"skill://{(skills_root / 'alpha' / 'SKILL.md').as_posix()}"],
                }
            ),
            encoding="utf-8",
        )

        ctx = _builder(tmp_path, skills_root).build_session_context(
            agent="specialist", provider_type="codex"
        )
        assert "alpha" in ctx
        assert "beta" not in ctx

    def test_mapped_agent_on_kiro_still_defers_to_native_load(self, tmp_path):
        """Unchanged kiro behavior: kiro-cli loads the mapped skill:// files
        itself, so Kiro Crew must not inject them a second time."""
        skills_root = tmp_path / "skills"
        _make_skill(skills_root, "alpha")
        d = self._agents_dir(tmp_path)
        (d / "specialist.json").write_text(
            json.dumps(
                {
                    "name": "specialist",
                    "resources": [f"skill://{(skills_root / 'alpha' / 'SKILL.md').as_posix()}"],
                }
            ),
            encoding="utf-8",
        )

        ctx = _builder(tmp_path, skills_root).build_session_context(
            agent="specialist", provider_type="acp"
        )
        assert "[Skills:]" not in ctx

    def test_unmapped_kirocrew_still_gets_everything_on_codex(self, tmp_path):
        skills_root = tmp_path / "skills"
        _make_skill(skills_root, "alpha")
        _make_skill(skills_root, "beta")
        self._agents_dir(tmp_path)

        ctx = _builder(tmp_path, skills_root).build_session_context(
            agent="kirocrew", provider_type="codex"
        )
        assert "alpha" in ctx and "beta" in ctx

    def test_unmapped_custom_agent_still_gets_nothing_on_codex(self, tmp_path):
        skills_root = tmp_path / "skills"
        _make_skill(skills_root, "alpha")
        d = self._agents_dir(tmp_path)
        (d / "plain.json").write_text(json.dumps({"name": "plain"}), encoding="utf-8")

        ctx = _builder(tmp_path, skills_root).build_session_context(
            agent="plain", provider_type="codex"
        )
        assert "[Skills:]" not in ctx


class TestReinjectionOnCodex:
    """The post-compaction skills re-injection uses the same gate helper as
    session-start, so it must extend to codex the same way."""

    def _builder_with_skill(self, tmp_path: Path) -> ContextBuilder:
        skills_dir = tmp_path / "skills" / "widget-maker"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: widget-maker\ndescription: Build a widget.\n---\n# WidgetMaker\nBody.",
            encoding="utf-8",
        )
        return _builder(tmp_path)

    def test_reinjection_fires_for_codex_default_agent(self, tmp_path):
        builder = self._builder_with_skill(tmp_path)
        msg, _ = builder.build_message(
            "carry on",
            is_new_session=False,
            needs_reinjection=True,
            provider_type="codex",
        )
        assert "[REINJECTED AFTER COMPACTION" in msg
        assert "widget-maker" in msg

    def test_reinjection_still_fires_for_kiro(self, tmp_path):
        """Unmapped default agent on kiro is unaffected by the widened gate:
        the "no mapping" branch never depended on provider_type."""
        builder = self._builder_with_skill(tmp_path)
        msg, _ = builder.build_message(
            "carry on",
            is_new_session=False,
            needs_reinjection=True,
            provider_type="acp",
        )
        assert "[REINJECTED AFTER COMPACTION" in msg
        assert "widget-maker" in msg
