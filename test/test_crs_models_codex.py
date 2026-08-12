"""The Code Review Sage settings dropdown follows the CONFIGURED ACP backend.

The review-model allowlist is built from ``model_registry``'s ``claude_code``
canonical keys, which only the kiro-cli backend serves. The codex backend rejects
those ids on the wire, and the registry describes none of its own models, so on
codex the settings endpoint must enumerate NOTHING — the UI is left with its own
'Default (agent config)' entry, which inherits the agent config's model and
resolves to whatever codex actually serves. Validation must then fall back to its
safe-token-only rule so a live codex model id is still persistable, with codex
itself rejecting an invalid pick.

The backend is a config value the operator can change under a running gateway, so
these also pin that the lookup happens per call rather than at import.
"""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import make_mocked_request

from kiro_crew import model_registry
from kiro_crew.apps.builtins.code_review_sage.backend import routes


@pytest.fixture(autouse=True)
def _isolate_model_cache(monkeypatch):
    """Give each test its own copy of the per-backend allowlist cache; the real
    dict is module state shared with every other test in the process."""
    monkeypatch.setattr(routes, "_KNOWN_MODELS_BY_BACKEND", dict(routes._KNOWN_MODELS_BY_BACKEND))


def _set_backend(monkeypatch, backend: str) -> None:
    monkeypatch.setattr(routes, "configured_acp_backend", lambda: backend)


async def _settings_get() -> dict:
    resp = await routes._handle_settings(
        make_mocked_request("GET", "/api/apps/code-review-sage/settings")
    )
    return json.loads(resp.body)


def test_registry_describes_no_codex_models():
    """The premise of the empty dropdown: the registry has no codex entries."""
    assert model_registry.display_list("codex") == []


def test_known_models_empty_on_codex(monkeypatch):
    _set_backend(monkeypatch, "codex")
    assert routes._known_models() == []


def test_known_models_served_on_kiro_backend(monkeypatch):
    _set_backend(monkeypatch, "")
    expected = [row["model_name"] for row in model_registry.display_list("claude_code")]
    assert expected  # the registry ships Anthropic entries
    assert routes._known_models() == expected


@pytest.mark.asyncio
async def test_settings_models_empty_on_codex(monkeypatch):
    _set_backend(monkeypatch, "codex")
    assert (await _settings_get())["models"] == []


@pytest.mark.asyncio
async def test_settings_models_unchanged_on_kiro_backend(monkeypatch):
    _set_backend(monkeypatch, "")
    expected = [row["model_name"] for row in model_registry.display_list("claude_code")]
    assert (await _settings_get())["models"] == expected


def test_valid_model_allows_safe_token_on_codex(monkeypatch):
    _set_backend(monkeypatch, "codex")
    for token in ("gpt-5.2-codex", "auto", "some_model.v2"):
        assert routes._valid_model(token) is True


@pytest.mark.parametrize(
    "bad",
    ["", "../../etc/passwd", "gpt 5 codex", "model;rm -rf /", "global.anthropic.x[1m]", "m" * 65],
)
def test_valid_model_rejects_unsafe_token_on_codex(monkeypatch, bad):
    """The safe-token half of validation is unconditional: the value becomes a
    cli.json overlay key for the review worker subprocess."""
    _set_backend(monkeypatch, "codex")
    assert routes._valid_model(bad) is False


def test_valid_model_still_allowlists_on_kiro_backend(monkeypatch):
    _set_backend(monkeypatch, "")
    known = [row["model_name"] for row in model_registry.display_list("claude_code")]
    assert routes._valid_model(known[0]) is True
    assert routes._valid_model("evil-model-9000") is False


def test_backend_is_read_per_call(monkeypatch):
    """A backend switch is picked up without re-importing the module."""
    backend = {"id": ""}
    monkeypatch.setattr(routes, "configured_acp_backend", lambda: backend["id"])
    assert routes._known_models()
    backend["id"] = "codex"
    assert routes._known_models() == []
    backend["id"] = ""
    assert routes._known_models()
