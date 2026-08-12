"""Tests for the codex-backend fallback in kiro_crew.dashboard.handlers.usage.

_parse_sessions() normally reads kiro-cli's own transcript directory
(``_sessions_dir()``), which never exists on a codex-only host (codex-acp
keeps no session transcripts of its own). These tests pin the SAME-SHAPE
contract on that fallback path (:func:`_sessions_from_own_records`, reached
through :func:`_parse_sessions`), the unchanged kiro-path error contract, and
:func:`_cached_parse_sessions` agreeing with both.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import kiro_crew.dashboard.handlers.usage as usage_mod
from kiro_crew.acp.types import ACP_BACKEND_CODEX
from kiro_crew.dashboard.handlers.usage import _cached_parse_sessions, _parse_sessions

# The exact key set _parse_sessions() returns on the healthy kiro path — the
# contract the codex fallback must match byte-for-byte at the top level.
_EXPECTED_TOP_KEYS = {
    "total_sessions",
    "total_messages",
    "total_tool_calls",
    "all_time_sessions",
    "daily_history",
    "today",
    "this_week",
    "this_month",
    "avg_msgs_per_session",
    "avg_tools_per_session",
}
_EXPECTED_PERIOD_KEYS = {"sessions", "messages", "tool_calls"}
_EXPECTED_DAILY_ENTRY_KEYS = {"date", "sessions", "messages", "tool_calls"}


def _write_token_shard(shard_dir: Path, day: str, records: list[dict]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"{day}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def _token_record(slot: str, ts: datetime) -> dict:
    return {
        "_type": "tokens",
        "ts": ts.isoformat(),
        "slot": slot,
        "provider": "codex",
        "model": "gpt-5",
        "input": 100,
        "output": 50,
        "cache_create": 0,
        "cache_read": 0,
        "cost": 0.0,
        "credits": 0.0,
        "turns": 1,
        "duration_ms": 500,
    }


@pytest.fixture()
def isolated_shard_dir(tmp_path, monkeypatch):
    """Point the token-usage shard dir at an isolated tmp directory and reset
    every process-global cache this module touches, mirroring test_usage.py's
    ``_patch_shard_layout`` / ``_reset_token_cache`` / ``_reset_sessions_cache``.
    """
    shard_dir = tmp_path / "tokens"
    shard_dir.mkdir()
    monkeypatch.setattr(usage_mod, "_TOKEN_USAGE_DIR", shard_dir)
    usage_mod._TOKEN_CACHE = {}
    usage_mod._TOKEN_CACHE_KEY = None
    usage_mod._TOKEN_CACHE_TS = 0.0
    usage_mod._SESSIONS_CACHE = None
    usage_mod._SESSIONS_CACHE_TS = 0.0
    yield shard_dir
    usage_mod._SESSIONS_CACHE = None
    usage_mod._SESSIONS_CACHE_TS = 0.0


class TestCodexFallbackShape:
    """(a) codex backend + no kiro sessions dir + own records -> same-shape
    dict, no "error" key, today/this_week/this_month present with int values.
    """

    def test_builds_same_shape_dict_no_error(self, tmp_path, isolated_shard_dir):
        now = datetime.now().astimezone()
        _write_token_shard(
            isolated_shard_dir,
            now.strftime("%Y-%m-%d"),
            [_token_record("dashboard:chat-1-111", now)],
        )
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            result = _parse_sessions()

        assert "error" not in result
        assert result["total_sessions"] == 1
        assert result["total_messages"] == 1
        assert result["total_tool_calls"] == 0

        for period in ("today", "this_week", "this_month"):
            assert period in result
            for field in ("sessions", "messages", "tool_calls"):
                value = result[period][field]
                assert isinstance(value, int)

        assert result["today"]["sessions"] == 1
        assert result["today"]["messages"] == 1

    def test_distinct_slots_counted_as_distinct_sessions(self, tmp_path, isolated_shard_dir):
        now = datetime.now().astimezone()
        _write_token_shard(
            isolated_shard_dir,
            now.strftime("%Y-%m-%d"),
            [
                _token_record("dashboard:chat-1-111", now),
                _token_record("dashboard:chat-2-222", now),
                _token_record("dashboard:chat-1-111", now + timedelta(minutes=5)),
            ],
        )
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            result = _parse_sessions()

        # Two distinct slots -> two sessions; three rows total -> three messages.
        assert result["total_sessions"] == 2
        assert result["total_messages"] == 3
        assert result["all_time_sessions"] == 2

    def test_long_lived_session_bucketed_once_by_earliest_day(self, tmp_path, isolated_shard_dir):
        # The same slot posts turns on two different days within the window;
        # it must be counted as ONE session, attributed to its earliest day,
        # not once per active day.
        now = datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0)
        yesterday = now - timedelta(days=1)
        _write_token_shard(
            isolated_shard_dir,
            yesterday.strftime("%Y-%m-%d"),
            [_token_record("dashboard:chat-9-999", yesterday)],
        )
        _write_token_shard(
            isolated_shard_dir,
            now.strftime("%Y-%m-%d"),
            [_token_record("dashboard:chat-9-999", now)],
        )
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            result = _parse_sessions()

        assert result["total_sessions"] == 1
        assert result["total_messages"] == 2
        # Bucketed under the earliest (yesterday) day, not today.
        dates = [h["date"] for h in result["daily_history"]]
        assert dates == [yesterday.strftime("%Y-%m-%d")]

    def test_no_own_records_yields_empty_but_valid_shape(self, tmp_path, isolated_shard_dir):
        # No shard files at all — still same shape, all zeros, no "error".
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            result = _parse_sessions()

        assert "error" not in result
        assert result["total_sessions"] == 0
        assert result["total_messages"] == 0
        assert result["total_tool_calls"] == 0
        assert result["daily_history"] == []
        assert result["avg_msgs_per_session"] == 0.0
        assert result["avg_tools_per_session"] == 0.0


class TestKiroBackendUnchanged:
    """(b) kiro backend + no dir => existing {"error": ...} contract unchanged."""

    def test_kiro_backend_missing_dir_still_errors(self, tmp_path):
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "nope"),
            patch.object(usage_mod, "configured_acp_backend", return_value=""),
        ):
            assert _parse_sessions() == {"error": "No sessions directory"}

    def test_default_config_missing_dir_still_errors(self, tmp_path):
        # No patch on configured_acp_backend at all: the default config has
        # agent.acp_backend == "" (kiro-cli), so behavior must be identical
        # to before this change existed.
        with patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "nope"):
            assert _parse_sessions() == {"error": "No sessions directory"}


class TestShapeKeysMatchKiroPath:
    """(c) shape keys match the kiro-path shape exactly (pin the key set)."""

    def test_top_level_and_nested_keys_match(self, tmp_path, isolated_shard_dir):
        # Build the healthy kiro-path result from a real kiro session file.
        kiro_dir = tmp_path / "cli"
        kiro_dir.mkdir()
        session_file = kiro_dir / "s1.jsonl"
        session_file.write_text(
            json.dumps({"kind": "Prompt"}) + "\n" + json.dumps({"kind": "ToolResults"}) + "\n"
        )
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", kiro_dir),
            patch.object(usage_mod, "validate_file_path", return_value=str(session_file)),
        ):
            kiro_result = _parse_sessions()
        assert "error" not in kiro_result
        assert set(kiro_result.keys()) == _EXPECTED_TOP_KEYS

        # Build the codex fallback result from Kiro Crew's own records.
        now = datetime.now().astimezone()
        _write_token_shard(
            isolated_shard_dir,
            now.strftime("%Y-%m-%d"),
            [_token_record("dashboard:chat-1-111", now)],
        )
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            codex_result = _parse_sessions()
        assert "error" not in codex_result

        # Exact same top-level key set on both paths.
        assert set(codex_result.keys()) == set(kiro_result.keys()) == _EXPECTED_TOP_KEYS

        for period in ("today", "this_week", "this_month"):
            assert set(kiro_result[period].keys()) == _EXPECTED_PERIOD_KEYS
            assert set(codex_result[period].keys()) == _EXPECTED_PERIOD_KEYS

        assert set(kiro_result["daily_history"][0].keys()) == _EXPECTED_DAILY_ENTRY_KEYS
        assert set(codex_result["daily_history"][0].keys()) == _EXPECTED_DAILY_ENTRY_KEYS


class TestCachedParseSessionsAgreement:
    """_cached_parse_sessions must agree with _parse_sessions on both paths:
    {} shortcut only for kiro-cli, real fallback data for a non-kiro backend.
    """

    @pytest.mark.asyncio
    async def test_kiro_backend_missing_dir_returns_empty(self, tmp_path, isolated_shard_dir):
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "nope"),
            patch.object(usage_mod, "configured_acp_backend", return_value=""),
        ):
            assert await _cached_parse_sessions() == {}

    @pytest.mark.asyncio
    async def test_codex_backend_missing_dir_returns_fallback_data(
        self, tmp_path, isolated_shard_dir
    ):
        now = datetime.now().astimezone()
        _write_token_shard(
            isolated_shard_dir,
            now.strftime("%Y-%m-%d"),
            [_token_record("dashboard:chat-1-111", now)],
        )
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            result = await _cached_parse_sessions()

        assert result != {}
        assert "error" not in result
        assert result["total_sessions"] == 1
        # A healthy (non-error) result must be cacheable.
        assert usage_mod._SESSIONS_CACHE is not None

    @pytest.mark.asyncio
    async def test_codex_backend_cache_hit_skips_reparse(self, tmp_path, isolated_shard_dir):
        now = datetime.now().astimezone()
        _write_token_shard(
            isolated_shard_dir,
            now.strftime("%Y-%m-%d"),
            [_token_record("dashboard:chat-1-111", now)],
        )
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            first = await _cached_parse_sessions()
            # Remove the shard so a re-parse would see different data; the
            # cached result must still be served within the TTL.
            for f in isolated_shard_dir.iterdir():
                f.unlink()
            second = await _cached_parse_sessions()

        assert first == second
        assert second["total_sessions"] == 1


class TestOwnRecordsFallbackDirect:
    """Exercise _sessions_from_own_records() directly for the tool_calls-is-
    unavailable contract and the credits/turns-window scoping it documents.
    """

    def test_tool_calls_always_zero(self, tmp_path, isolated_shard_dir):
        now = datetime.now().astimezone()
        _write_token_shard(
            isolated_shard_dir,
            now.strftime("%Y-%m-%d"),
            [_token_record("dashboard:chat-1-111", now)],
        )
        result = usage_mod._sessions_from_own_records()
        assert result["total_tool_calls"] == 0
        assert result["avg_tools_per_session"] == 0.0
        assert all(h["tool_calls"] == 0 for h in result["daily_history"])

    def test_ignores_non_token_records(self, tmp_path, isolated_shard_dir):
        now = datetime.now().astimezone()
        _write_token_shard(
            isolated_shard_dir,
            now.strftime("%Y-%m-%d"),
            [
                {"_type": "metadata", "created_at": now.isoformat()},
                {"role": "user", "content": "hello"},
                _token_record("dashboard:chat-1-111", now),
            ],
        )
        result = usage_mod._sessions_from_own_records()
        assert result["total_sessions"] == 1
        assert result["total_messages"] == 1

    def test_rows_missing_slot_are_skipped(self, tmp_path, isolated_shard_dir):
        now = datetime.now().astimezone()
        record = _token_record("", now)
        _write_token_shard(isolated_shard_dir, now.strftime("%Y-%m-%d"), [record])
        result = usage_mod._sessions_from_own_records()
        assert result["total_sessions"] == 0
        assert result["total_messages"] == 0

    def test_malformed_json_line_skipped(self, tmp_path, isolated_shard_dir):
        shard_path = (
            isolated_shard_dir / f"{datetime.now().astimezone().strftime('%Y-%m-%d')}.jsonl"
        )
        shard_path.write_text('{"bad json"\n')
        result = usage_mod._sessions_from_own_records()
        assert result["total_sessions"] == 0

    def test_old_shard_outside_window_excluded(self, tmp_path, isolated_shard_dir):
        old_day = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        old_ts = datetime.now().astimezone() - timedelta(days=60)
        _write_token_shard(
            isolated_shard_dir, old_day, [_token_record("dashboard:chat-old-1", old_ts)]
        )
        result = usage_mod._sessions_from_own_records()
        assert result["total_sessions"] == 0
