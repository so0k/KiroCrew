"""Tests for the codex rollout-log side channel (:mod:`kiro_crew.codex_usage`)
and the ``billing`` block it feeds into GET /api/usage/kiro.

Every fixture is a synthetic rollout store under ``tmp_path`` with ``CODEX_HOME``
pointed at it — no network, and never the developer's real ``~/.codex``.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import kiro_crew.codex_usage as codex_usage
import kiro_crew.dashboard.handlers.usage as usage_mod
from kiro_crew.acp.types import ACP_BACKEND_CODEX
from kiro_crew.codex_usage import (
    codex_sessions_root,
    latest_rate_limits,
    session_token_usage,
)
from kiro_crew.dashboard.handlers.usage import _codex_billing_block, api_kiro_usage

# Real session uuids from a live rollout store; the filename component equals
# the ACP session id, which is what the per-session join keys on.
_SID = "019ff55b-2775-7cc0-90ed-b0de45dd132a"
_OTHER_SID = "019ff403-394b-7392-9b64-b2e50871b89c"

# Seconds a read of a hostile store may take before the test calls it hung. Any
# real read of these fixtures is sub-millisecond; the point is that a regression
# FAILS the test instead of hanging the suite, so the guard runs on a daemon
# thread and the value only has to be far above real work.
_HANG_GUARD_S = 10.0

_TOTALS = {
    "input_tokens": 649509,
    "cached_input_tokens": 514432,
    "cache_write_input_tokens": 0,
    "output_tokens": 2724,
    "reasoning_output_tokens": 177,
    "total_tokens": 652233,
}


def _rate_limits(
    *,
    used_percent: float = 13.0,
    secondary: dict | None = None,
    plan_type: str = "plus",
    credits: dict | None = None,
) -> dict:
    """A ``rate_limits`` block shaped exactly like codex's own."""
    return {
        "limit_id": "codex",
        "limit_name": None,
        "primary": {
            "used_percent": used_percent,
            "window_minutes": 10080,
            "resets_at": 1787011261,
        },
        "secondary": secondary,
        "credits": (
            credits
            if credits is not None
            else {"has_credits": False, "unlimited": False, "balance": "0"}
        ),
        "individual_limit": None,
        "spend_control_reached": None,
        "plan_type": plan_type,
        "rate_limit_reached_type": None,
    }


def _token_count_line(
    ts: str,
    *,
    rate_limits: dict | None = None,
    totals: dict | None = None,
    context_window: int = 258400,
) -> str:
    info: dict = {}
    if totals is not None:
        info["total_token_usage"] = totals
        info["last_token_usage"] = totals
        info["model_context_window"] = context_window
    payload = {"type": "token_count", "info": info, "rate_limits": rate_limits}
    return json.dumps({"timestamp": ts, "type": "event_msg", "payload": payload})


def _other_line(text: str = "hello") -> str:
    """A non-token_count rollout line, which most lines in a real file are."""
    return json.dumps(
        {
            "timestamp": "2026-08-12T09:43:40.000Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": text},
        }
    )


def _filler_line(size: int) -> str:
    """A non-token_count line of EXACTLY ``size`` bytes including its newline."""
    overhead = len('{"pad":""}') + 1
    return '{"pad":"' + "x" * max(size - overhead, 1) + '"}'


def _straddling_lines(event: str) -> list[str]:
    """Lines placing ``event`` across the tail window's leading edge.

    The tail read discards its first (partial) line, so an event cut by that
    edge is invisible to it — this is the layout that forces the bounded
    full-file scan. Sizes are derived from ``_TAIL_READ_BYTES`` so the fixture
    cannot silently stop straddling if that bound changes.
    """
    tail = codex_usage._TAIL_READ_BYTES
    event_len = len(event) + 1
    # size - tail then lands mid-event: past its "token_count" marker, before
    # its end.
    return [_filler_line(200), event, _filler_line(tail - event_len // 2)]


def _decoy_sid(n: int) -> str:
    """A distinct session uuid, for filling the candidate set with noise."""
    return f"019ff55b-2775-7cc0-90ed-b0de45dd13{n:02x}"


def _write_rollout(
    root: Path,
    lines: list[str],
    *,
    sid: str = _SID,
    day: tuple[str, str, str] = ("2026", "08", "12"),
    started: str = "2026-08-12T09-43-39",
    mtime: float | None = None,
) -> Path:
    """Materialise one rollout file in codex's dated layout."""
    day_dir = root.joinpath(*day)
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"rollout-{started}-{sid}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _without_hanging(call):
    """Run ``call`` on a daemon thread, FAILING rather than hanging if it blocks.

    ``open(2)`` on a FIFO with no writer blocks uninterruptibly, so a regression
    cannot be caught in-line: the call is abandoned on a daemon thread that dies
    with the interpreter, and the assertion fires on the join timeout.
    """
    done: list[tuple[str, object]] = []

    def _run() -> None:
        try:
            done.append(("ok", call()))
        except Exception as exc:  # re-raised below, on the calling thread
            done.append(("raised", exc))

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(_HANG_GUARD_S)
    assert not worker.is_alive(), "the read blocked — a non-regular file in the store hung open(2)"
    kind, value = done[0]
    if kind == "raised":
        raise value  # type: ignore[misc]
    return value


@pytest.fixture()
def codex_root(tmp_path, monkeypatch) -> Path:
    """Point ``CODEX_HOME`` at an isolated tree and reset the TTL cache."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    codex_usage.clear_cache()
    yield tmp_path / "codex" / "sessions"
    codex_usage.clear_cache()


class TestLatestRateLimits:
    def test_reads_the_newest_event_in_the_file(self, codex_root):
        _write_rollout(
            codex_root,
            [
                _other_line(),
                _token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits()),
                _other_line(),
                _token_count_line(
                    "2026-08-12T10:39:00.000Z",
                    rate_limits=_rate_limits(used_percent=21.5),
                ),
            ],
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.plan_type == "plus"
        assert limits.captured_at == "2026-08-12T10:39:00.000Z"
        assert limits.primary is not None
        assert limits.primary.used_percent == 21.5
        assert limits.primary.window_minutes == 10080
        assert limits.primary.resets_at == 1787011261
        assert limits.credits is not None
        assert limits.credits.has_credits is False
        assert limits.credits.balance == "0"

    def test_secondary_null_stays_none(self, codex_root):
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.secondary is None

    def test_secondary_window_parsed_when_present(self, codex_root):
        _write_rollout(
            codex_root,
            [
                _token_count_line(
                    "2026-08-12T09:44:00.000Z",
                    rate_limits=_rate_limits(
                        secondary={
                            "used_percent": 4.5,
                            "window_minutes": 300,
                            "resets_at": 1786000000,
                        }
                    ),
                )
            ],
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.secondary is not None
        assert limits.secondary.window_minutes == 300
        assert limits.secondary.used_percent == 4.5
        assert limits.secondary.resets_at == 1786000000

    def test_missing_reset_time_is_none_not_zero(self, codex_root):
        limits_raw = _rate_limits()
        limits_raw["primary"]["resets_at"] = None
        _write_rollout(
            codex_root, [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=limits_raw)]
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.resets_at is None

    def test_garbage_lines_are_skipped(self, codex_root):
        _write_rollout(
            codex_root,
            [
                '{"bad json',
                "",
                "not json at all",
                json.dumps({"payload": "token_count is a string here"}),
                _token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits()),
                '{"truncated": ',
            ],
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.used_percent == 13.0

    def test_missing_store_returns_none(self, codex_root):
        assert not codex_root.exists()
        assert latest_rate_limits() is None

    def test_store_with_no_rollout_files_returns_none(self, codex_root):
        codex_root.joinpath("2026", "08", "12").mkdir(parents=True)
        assert latest_rate_limits() is None

    def test_non_date_directories_are_ignored(self, codex_root):
        codex_root.mkdir(parents=True)
        (codex_root / "not-a-year").mkdir()
        (codex_root / "not-a-year" / "rollout-x-y.jsonl").write_text("{}\n")
        assert latest_rate_limits() is None

    def test_event_without_rate_limits_falls_through_to_older_file(self, codex_root):
        # The most recently written session completed a turn but codex reported
        # no rate limits on it; the snapshot must come from the next candidate
        # rather than being reported as absent.
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
            sid=_OTHER_SID,
            started="2026-08-12T09-40-15",
            mtime=1_800_000_000,
        )
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T10:00:00.000Z", rate_limits=None)],
            sid=_SID,
            mtime=1_800_000_500,
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.captured_at == "2026-08-12T09:44:00.000Z"

    def test_resumed_session_in_an_older_day_dir_wins_on_mtime(self, codex_root):
        # A resumed session keeps appending to the file named for the day it
        # STARTED, so the freshest event can live under an older date.
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T01:00:00.000Z", rate_limits=_rate_limits())],
            sid=_SID,
            day=("2026", "08", "12"),
            mtime=1_800_000_000,
        )
        _write_rollout(
            codex_root,
            [
                _token_count_line(
                    "2026-08-12T11:00:00.000Z", rate_limits=_rate_limits(used_percent=44.0)
                )
            ],
            sid=_OTHER_SID,
            day=("2026", "08", "11"),
            started="2026-08-11T22-00-00",
            mtime=1_800_000_900,
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.used_percent == 44.0

    def test_last_complete_event_inside_the_tail_window_wins(self, codex_root):
        # A long transcript: the tail window must find the LAST event without
        # reading the head.
        filler = _other_line("x" * (codex_usage._TAIL_READ_BYTES * 2))
        _write_rollout(
            codex_root,
            [
                filler,
                _token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits()),
                _token_count_line(
                    "2026-08-12T10:39:00.000Z", rate_limits=_rate_limits(used_percent=31.0)
                ),
            ],
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.used_percent == 31.0

    def test_event_straddling_the_tail_boundary_is_still_found(self, codex_root):
        # The only token_count event is cut by the tail window's leading edge,
        # so the discarded partial line IS the event. A bounded full scan is
        # what keeps it readable.
        event = _token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())
        path = _write_rollout(codex_root, _straddling_lines(event))
        # Pin the premise: the tail alone cannot see this event.
        handle = codex_usage._open_rollout(path, codex_usage._resolved_root(codex_root))
        assert handle is not None
        with handle as fh:
            tail_lines, complete = codex_usage._tail_lines(fh)
        assert not complete
        assert not any(codex_usage._TOKEN_COUNT_MARKER in line for line in tail_lines)

        limits = latest_rate_limits()
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.used_percent == 13.0

    def test_oversized_file_is_not_fully_scanned(self, codex_root, monkeypatch):
        # Same straddling layout, but the file is over the full-scan bound: the
        # answer is "unknown", never an unbounded read.
        monkeypatch.setattr(codex_usage, "_FULL_SCAN_MAX_BYTES", 1024)
        event = _token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())
        _write_rollout(codex_root, _straddling_lines(event))
        assert latest_rate_limits() is None

    def test_snapshot_anywhere_in_the_weekly_window_is_found(self, codex_root):
        # A host whose last completed turn was days ago still has a meaningful
        # weekly percentage; the day-dir walk covers the whole weekly window,
        # so the panel must not go blank after three quiet days.
        days = [f"{day:02d}" for day in range(12, 4, -1)]
        for offset, day in enumerate(days):
            _write_rollout(
                codex_root,
                [
                    _token_count_line(
                        f"2026-08-{day}T09:44:00.000Z",
                        # Only the OLDEST day dir carries a rate-limited event;
                        # every fresher one has to be skipped, not treated as
                        # the answer.
                        rate_limits=_rate_limits(used_percent=66.0) if day == days[-1] else None,
                    )
                ],
                sid=_decoy_sid(offset),
                day=("2026", "08", day),
                started=f"2026-08-{day}T09-43-39",
            )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.used_percent == 66.0

    def test_total_miss_takes_one_wider_pass(self, codex_root):
        # Nothing inside the weekly window: a stale snapshot carrying its own
        # captured_at still beats an empty panel, so ONE wider pass runs before
        # the negative result is cached.
        for offset, day in enumerate(f"{day:02d}" for day in range(12, 0, -1)):
            _write_rollout(
                codex_root,
                [_token_count_line(f"2026-08-{day}T09:44:00.000Z", rate_limits=None)],
                sid=_decoy_sid(offset),
                day=("2026", "08", day),
                started=f"2026-08-{day}T09-43-39",
            )
        _write_rollout(
            codex_root,
            [_token_count_line("2026-07-20T09:44:00.000Z", rate_limits=_rate_limits())],
            sid=_OTHER_SID,
            day=("2026", "07", "20"),
            started="2026-07-20T09-43-39",
        )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.captured_at == "2026-07-20T09:44:00.000Z"

    def test_the_wider_pass_is_still_bounded(self, codex_root, monkeypatch):
        # The fallback widens the walk, it does not remove the bound.
        monkeypatch.setattr(codex_usage, "_LATEST_DAY_DIRS", 1)
        monkeypatch.setattr(codex_usage, "_WIDE_DAY_DIRS", 2)
        for offset, day in enumerate(["12", "11", "10"]):
            _write_rollout(
                codex_root,
                [
                    _token_count_line(
                        f"2026-08-{day}T09:44:00.000Z",
                        rate_limits=_rate_limits() if day == "10" else None,
                    )
                ],
                sid=_decoy_sid(offset),
                day=("2026", "08", day),
                started=f"2026-08-{day}T09-43-39",
            )
        assert latest_rate_limits() is None

    def test_a_run_of_turnless_sessions_does_not_hide_the_snapshot(self, codex_root):
        # Sessions started but never completing a turn carry no token_count
        # event at all and sort NEWEST on mtime, so the candidate bound has to
        # leave room for a whole run of them in front of the real snapshot.
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
            sid=_SID,
            mtime=1_800_000_000,
        )
        for offset in range(12):
            _write_rollout(
                codex_root,
                [_other_line()],
                sid=_decoy_sid(offset),
                started=f"2026-08-12T10-00-{offset:02d}",
                mtime=1_800_000_100 + offset,
            )
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.used_percent == 13.0

    def test_the_candidate_bound_is_still_enforced(self, codex_root, monkeypatch):
        monkeypatch.setattr(codex_usage, "_LATEST_CANDIDATE_FILES", 2)
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
            sid=_SID,
            mtime=1_800_000_000,
        )
        for offset in range(3):
            _write_rollout(
                codex_root,
                [_other_line()],
                sid=_decoy_sid(offset),
                started=f"2026-08-12T10-00-{offset:02d}",
                mtime=1_800_000_100 + offset,
            )
        assert latest_rate_limits() is None

    def test_result_is_ttl_cached(self, codex_root):
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
        )
        first = latest_rate_limits()
        assert first is not None
        # Delete the whole store: a cache hit must still answer.
        for path in codex_root.rglob("*.jsonl"):
            path.unlink()
        assert latest_rate_limits() == first
        codex_usage.clear_cache()
        assert latest_rate_limits() is None

    def test_cache_is_keyed_on_the_resolved_store_root(self, tmp_path, monkeypatch, codex_root):
        # Two stores, no clear_cache() between them: a re-pointed CODEX_HOME
        # must MISS rather than serve the other store's snapshot.
        for name, used_percent in (("a", 13.0), ("b", 77.0)):
            _write_rollout(
                tmp_path / name / "sessions",
                [
                    _token_count_line(
                        "2026-08-12T09:44:00.000Z",
                        rate_limits=_rate_limits(used_percent=used_percent),
                    )
                ],
            )

        def _percent(name: str) -> float:
            monkeypatch.setenv("CODEX_HOME", str(tmp_path / name))
            limits = latest_rate_limits()
            assert limits is not None and limits.primary is not None
            return limits.primary.used_percent

        assert _percent("a") == 13.0
        assert _percent("b") == 77.0
        assert _percent("a") == 13.0

    def test_negative_result_is_cached_too(self, codex_root):
        assert latest_rate_limits() is None
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
        )
        assert latest_rate_limits() is None
        codex_usage.clear_cache()
        assert latest_rate_limits() is not None

    def test_expired_cache_re_reads(self, codex_root, monkeypatch):
        monkeypatch.setattr(codex_usage, "RATE_LIMIT_CACHE_TTL_S", 0.0)
        assert latest_rate_limits() is None
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
        )
        assert latest_rate_limits() is not None

    def test_unreadable_store_is_not_fatal(self, codex_root):
        with patch.object(codex_usage, "_day_dirs_newest_first", side_effect=OSError("boom")):
            assert latest_rate_limits() is None

    def test_sessions_root_follows_codex_home(self, codex_root):
        assert codex_sessions_root() == codex_root


class TestSessionTokenUsage:
    def test_joins_on_the_session_id(self, codex_root):
        _write_rollout(
            codex_root,
            [
                _token_count_line(
                    "2026-08-12T09:44:00.000Z", rate_limits=_rate_limits(), totals=_TOTALS
                )
            ],
            sid=_SID,
        )
        # A decoy session whose numbers must never be returned for _SID.
        _write_rollout(
            codex_root,
            [
                _token_count_line(
                    "2026-08-12T09:44:00.000Z",
                    totals={**_TOTALS, "total_tokens": 11},
                )
            ],
            sid=_OTHER_SID,
            started="2026-08-12T03-27-59",
        )
        tokens = session_token_usage(_SID)
        assert tokens is not None
        assert tokens.total_tokens == 652233
        assert tokens.input_tokens == 649509
        assert tokens.cached_input_tokens == 514432
        assert tokens.output_tokens == 2724
        assert tokens.reasoning_output_tokens == 177
        assert tokens.context_window == 258400

    def test_returns_the_cumulative_total_of_the_last_event(self, codex_root):
        _write_rollout(
            codex_root,
            [
                _token_count_line(
                    "2026-08-12T09:44:00.000Z", totals={**_TOTALS, "total_tokens": 100}
                ),
                _token_count_line(
                    "2026-08-12T10:39:00.000Z", totals={**_TOTALS, "total_tokens": 900}
                ),
            ],
        )
        tokens = session_token_usage(_SID)
        assert tokens is not None
        assert tokens.total_tokens == 900

    def test_unknown_session_returns_none(self, codex_root):
        _write_rollout(codex_root, [_token_count_line("2026-08-12T09:44:00.000Z", totals=_TOTALS)])
        assert session_token_usage(_OTHER_SID) is None

    def test_missing_store_returns_none(self, codex_root):
        assert session_token_usage(_SID) is None

    @pytest.mark.parametrize(
        "sid",
        ["", "*", "../../../etc/passwd", "not-a-uuid", _SID + "/x", "019ff55b_2775"],
    )
    def test_malformed_session_id_is_refused(self, codex_root, sid):
        _write_rollout(codex_root, [_token_count_line("2026-08-12T09:44:00.000Z", totals=_TOTALS)])
        assert session_token_usage(sid) is None

    def test_event_without_totals_returns_none(self, codex_root):
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
        )
        assert session_token_usage(_SID) is None

    def test_garbage_only_file_returns_none(self, codex_root):
        _write_rollout(codex_root, ['{"bad json', "nope"])
        assert session_token_usage(_SID) is None

    def test_missing_context_window_is_zero(self, codex_root):
        line = json.loads(_token_count_line("2026-08-12T09:44:00.000Z", totals=_TOTALS))
        del line["payload"]["info"]["model_context_window"]
        _write_rollout(codex_root, [json.dumps(line)])
        tokens = session_token_usage(_SID)
        assert tokens is not None
        assert tokens.context_window == 0

    def test_straddling_event_is_still_found(self, codex_root):
        event = _token_count_line("2026-08-12T09:44:00.000Z", totals=_TOTALS)
        _write_rollout(codex_root, _straddling_lines(event))
        tokens = session_token_usage(_SID)
        assert tokens is not None
        assert tokens.total_tokens == 652233


class TestStoreContainment:
    """Nothing outside the rollout store is ever opened, for any CODEX_HOME.

    ``hooks.is_sensitive_path`` is anchored on ``$HOME``, so with the store
    relocated outside the home directory — which every fixture here is — it
    does NOT cover ``$CODEX_HOME/auth.json``. Requiring the resolved candidate
    to stay under the resolved sessions root is what makes the module's
    "rollout logs and nothing else" guarantee true.
    """

    @pytest.fixture()
    def spy(self, monkeypatch) -> list[str]:
        """Record the canonical path of every file SUCCESSFULLY opened.

        ``os.open``, because the reader opens descriptors directly to pass
        ``O_NOFOLLOW``/``O_NONBLOCK`` and ``fstat`` the result. Only a
        descriptor that was actually handed back counts: a refused open (an
        ``O_NOFOLLOW`` ``ELOOP`` on a swapped final component) reaches no bytes,
        and recording the attempt would read as a leak that did not happen.
        """
        opened: list[str] = []
        real_open = os.open

        def _spy(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            opened.append(os.path.realpath(str(path)))
            return fd

        monkeypatch.setattr(os, "open", _spy)
        return opened

    @staticmethod
    def _decoy(codex_root: Path, *, sid: str = _OTHER_SID) -> None:
        """A real, in-store rollout, so the spy is provably live.

        Its token_count carries neither rate_limits nor totals, so neither
        reader can answer from it — only its being OPENED is the point.
        """
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=None)],
            sid=sid,
            started="2026-08-12T08-00-00",
        )

    @staticmethod
    def _auth_json(codex_root: Path) -> Path:
        """A stand-in for codex's live OAuth token pair, next to the store."""
        auth = codex_root.parent / "auth.json"
        auth.parent.mkdir(parents=True, exist_ok=True)
        auth.write_text(json.dumps({"tokens": {"access_token": "SECRET"}}) + "\n")
        return auth

    @staticmethod
    def _link_named_like_a_rollout(codex_root: Path, target: Path, *, sid: str = _SID) -> None:
        day_dir = codex_root / "2026" / "08" / "12"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / f"rollout-2026-08-12T09-43-39-{sid}.jsonl").symlink_to(target)

    def test_rate_limit_read_never_follows_a_link_out_of_the_store(self, codex_root, spy):
        auth = self._auth_json(codex_root)
        self._link_named_like_a_rollout(codex_root, auth)
        self._decoy(codex_root)
        # Materialising the fixture opened files itself; only the reader's own
        # opens are the subject here.
        spy.clear()
        assert latest_rate_limits() is None
        assert os.path.realpath(str(auth)) not in spy
        assert any(path.endswith(f"-{_OTHER_SID}.jsonl") for path in spy)

    def test_session_read_never_follows_a_link_out_of_the_store(self, codex_root, spy):
        # Both files glob as rollouts for _SID; only the real one is opened.
        auth = self._auth_json(codex_root)
        self._link_named_like_a_rollout(codex_root, auth)
        self._decoy(codex_root, sid=_SID)
        spy.clear()
        assert session_token_usage(_SID) is None
        assert os.path.realpath(str(auth)) not in spy
        assert any(path.endswith("T08-00-00-" + _SID + ".jsonl") for path in spy)

    def test_an_in_store_link_still_reads(self, codex_root):
        # The containment check is a containment check, not a symlink ban: a
        # link to another rollout INSIDE the store stays readable.
        target = _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
            sid=_OTHER_SID,
            started="2026-08-12T08-00-00",
        )
        self._link_named_like_a_rollout(codex_root, target)
        limits = latest_rate_limits()
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.used_percent == 13.0

    def test_an_in_store_link_is_read_through_to_its_target(self, codex_root, spy):
        # Same as above, from the open side: the hardened open follows the link
        # and the bytes come from the target's own inode.
        target = _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
            sid=_OTHER_SID,
            started="2026-08-12T08-00-00",
        )
        self._link_named_like_a_rollout(codex_root, target)
        spy.clear()
        assert latest_rate_limits() is not None
        assert os.path.realpath(str(target)) in spy

    def test_a_final_component_swapped_into_a_symlink_is_refused(
        self, codex_root, spy, monkeypatch
    ):
        # TOCTOU: containment is decided on a canonicalized path but the open
        # traverses the name again, so a rollout replaced by a symlink AFTER the
        # check would be followed without O_NOFOLLOW. The swap keeps the rollout
        # basename, so the name gate cannot catch this one.
        auth = self._auth_json(codex_root)
        target = _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
        )
        self._decoy(codex_root)
        real_contained = codex_usage._contained

        def _swap_after_the_check(path: Path, resolved_root: Path):
            resolved = real_contained(path, resolved_root)
            if resolved is not None and resolved == target:
                target.unlink()
                target.symlink_to(auth)
            return resolved

        monkeypatch.setattr(codex_usage, "_contained", _swap_after_the_check)
        spy.clear()
        assert latest_rate_limits() is None
        assert os.path.realpath(str(auth)) not in spy
        assert any(path.endswith(f"-{_OTHER_SID}.jsonl") for path in spy)

    @staticmethod
    def _store_root_linked_to(monkeypatch, home: Path, target: Path) -> Path:
        """Point ``$CODEX_HOME`` at a home whose ``sessions`` IS a symlink."""
        home.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        (home / codex_usage._SESSIONS_SUBDIR).symlink_to(target, target_is_directory=True)
        monkeypatch.setenv("CODEX_HOME", str(home))
        codex_usage.clear_cache()
        return home / codex_usage._SESSIONS_SUBDIR

    def test_a_symlinked_store_root_does_not_expose_its_neighbours(
        self, tmp_path, monkeypatch, spy
    ):
        # sessions -> a directory that ALSO holds auth.json. Containment is
        # measured against the RESOLVED root, so auth.json is "inside" the store
        # and the containment check alone admits it; the resolved BASENAME is
        # what refuses it.
        store = tmp_path / "elsewhere"
        root = self._store_root_linked_to(monkeypatch, tmp_path / "codex", store)
        auth = store / "auth.json"
        auth.write_text(json.dumps({"tokens": {"access_token": "SECRET"}}) + "\n")
        self._link_named_like_a_rollout(root, auth)
        self._decoy(root)
        spy.clear()
        assert latest_rate_limits() is None
        assert os.path.realpath(str(auth)) not in spy
        assert any(path.endswith(f"-{_OTHER_SID}.jsonl") for path in spy)

    def test_a_self_referential_store_root_does_not_expose_the_home(
        self, tmp_path, monkeypatch, spy
    ):
        # sessions -> $CODEX_HOME itself, so the store root resolves onto the
        # directory holding the live OAuth token pair.
        home = tmp_path / "codex"
        root = self._store_root_linked_to(monkeypatch, home, home)
        auth = home / "auth.json"
        auth.write_text(json.dumps({"tokens": {"access_token": "SECRET"}}) + "\n")
        self._link_named_like_a_rollout(root, auth)
        self._decoy(root)
        spy.clear()
        assert latest_rate_limits() is None
        assert os.path.realpath(str(auth)) not in spy
        assert any(path.endswith(f"-{_OTHER_SID}.jsonl") for path in spy)

    def test_a_session_read_refuses_a_link_onto_a_non_rollout_name(
        self, tmp_path, monkeypatch, spy
    ):
        # The per-session reader takes the same open path, so the name gate
        # covers it too. auth.json carries no token_count event either way, so
        # only the spy distinguishes "refused" from "read and found nothing".
        store = tmp_path / "elsewhere"
        root = self._store_root_linked_to(monkeypatch, tmp_path / "codex", store)
        auth = store / "auth.json"
        auth.write_text(json.dumps({"tokens": {"access_token": "SECRET"}}) + "\n")
        self._link_named_like_a_rollout(root, auth)
        self._decoy(root, sid=_SID)
        spy.clear()
        assert session_token_usage(_SID) is None
        assert os.path.realpath(str(auth)) not in spy
        assert any(path.endswith("T08-00-00-" + _SID + ".jsonl") for path in spy)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is POSIX-only")
class TestNonRegularFilesInTheStore:
    """A file in the store that is not a regular file must not wedge the reader.

    ``$CODEX_HOME/sessions`` is not a protected path and no denied-command rule
    covers ``mkfifo``, so a FIFO named like a rollout is reachable. Opening it
    blocks until a writer appears, and the read runs inside the usage handler's
    cache lock — one such file would take the whole endpoint down permanently.
    """

    @staticmethod
    def _fifo(codex_root: Path, *, sid: str = _SID) -> Path:
        day_dir = codex_root / "2026" / "08" / "12"
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"rollout-2026-08-12T09-43-39-{sid}.jsonl"
        os.mkfifo(path)
        return path

    def test_the_hardened_open_refuses_a_fifo_instead_of_blocking(self, codex_root):
        path = self._fifo(codex_root)
        root = codex_usage._resolved_root(codex_root)
        assert _without_hanging(lambda: codex_usage._open_rollout(path, root)) is None

    def test_a_fifo_does_not_wedge_the_rate_limit_read(self, codex_root):
        self._fifo(codex_root)
        assert _without_hanging(latest_rate_limits) is None

    def test_a_fifo_is_skipped_not_fatal(self, codex_root):
        # The candidate after it still answers: a hostile entry costs one skip.
        # The FIFO is given the NEWEST mtime so it is tried FIRST — otherwise the
        # real rollout answers before the FIFO is ever reached and the test
        # proves nothing.
        fifo = self._fifo(codex_root)
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
            sid=_OTHER_SID,
            started="2026-08-12T08-00-00",
            mtime=1_800_000_000,
        )
        os.utime(fifo, (1_800_000_500, 1_800_000_500))
        limits = _without_hanging(latest_rate_limits)
        assert limits is not None
        assert limits.primary is not None
        assert limits.primary.used_percent == 13.0

    def test_a_fifo_does_not_wedge_the_session_read(self, codex_root):
        self._fifo(codex_root)
        assert _without_hanging(lambda: session_token_usage(_SID)) is None

    def test_a_directory_named_like_a_rollout_is_refused(self, codex_root):
        day_dir = codex_root / "2026" / "08" / "12"
        day_dir.mkdir(parents=True)
        (day_dir / f"rollout-2026-08-12T09-43-39-{_SID}.jsonl").mkdir()
        assert latest_rate_limits() is None


class TestCodexBillingBlock:
    """The exact ``billing`` payload the frontend consumes on a codex host."""

    def test_exact_payload_shape(self, codex_root):
        _write_rollout(
            codex_root,
            [
                _token_count_line(
                    "2026-08-12T10:39:00.000Z",
                    rate_limits=_rate_limits(
                        secondary={
                            "used_percent": 4.5,
                            "window_minutes": 300,
                            "resets_at": 1786000000,
                        }
                    ),
                    totals=_TOTALS,
                )
            ],
        )
        assert _codex_billing_block() == {
            "provider": "codex",
            "plan_type": "plus",
            "primary": {
                "used_percent": 13.0,
                "window_minutes": 10080,
                "resets_at": 1787011261,
            },
            "secondary": {
                "used_percent": 4.5,
                "window_minutes": 300,
                "resets_at": 1786000000,
            },
            "credits": {"has_credits": False, "unlimited": False, "balance": "0"},
            "captured_at": "2026-08-12T10:39:00.000Z",
        }

    def test_null_secondary_is_null_not_zeroes(self, codex_root):
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T09:44:00.000Z", rate_limits=_rate_limits())],
        )
        assert _codex_billing_block()["secondary"] is None

    def test_no_store_yields_empty_block(self, codex_root):
        assert _codex_billing_block() == {}


class TestApiKiroUsageBilling:
    @pytest.fixture(autouse=True)
    def _reset_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(usage_mod, "_CACHE", {})
        monkeypatch.setattr(usage_mod, "_CACHE_TS", 0.0)
        monkeypatch.setattr(usage_mod, "_TOKEN_USAGE_DIR", tmp_path / "tokens")
        usage_mod._SESSIONS_CACHE = None
        usage_mod._SESSIONS_CACHE_TS = 0.0
        yield
        usage_mod._CACHE = {}
        usage_mod._CACHE_TS = 0.0
        usage_mod._SESSIONS_CACHE = None
        usage_mod._SESSIONS_CACHE_TS = 0.0

    async def _billing(self) -> dict:
        app = web.Application()
        app.router.add_get("/api/usage/kiro", api_kiro_usage)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/usage/kiro")
            assert resp.status == 200
            data = await resp.json()
        return data["billing"]

    @pytest.mark.asyncio
    async def test_codex_backend_serves_rate_limits(self, tmp_path, codex_root):
        _write_rollout(
            codex_root,
            [
                _token_count_line(
                    "2026-08-12T10:39:00.000Z", rate_limits=_rate_limits(), totals=_TOTALS
                )
            ],
        )
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            billing = await self._billing()
        assert billing["provider"] == "codex"
        assert billing["primary"]["used_percent"] == 13.0
        assert billing["captured_at"] == "2026-08-12T10:39:00.000Z"

    @pytest.mark.asyncio
    async def test_codex_backend_with_no_store_serves_empty_billing(self, tmp_path, codex_root):
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", tmp_path / "no-kiro-cli-here"),
            patch.object(usage_mod, "configured_acp_backend", return_value=ACP_BACKEND_CODEX),
        ):
            billing = await self._billing()
        assert billing == {}

    @pytest.mark.asyncio
    async def test_kiro_backend_keeps_the_credit_plan_block(self, tmp_path, codex_root):
        # A populated codex store must NOT leak into a kiro-cli host's billing.
        _write_rollout(
            codex_root,
            [_token_count_line("2026-08-12T10:39:00.000Z", rate_limits=_rate_limits())],
        )
        kiro_dir = tmp_path / "cli"
        kiro_dir.mkdir()
        session_file = kiro_dir / "s1.jsonl"
        session_file.write_text(json.dumps({"kind": "Prompt"}) + "\n")
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", kiro_dir),
            patch.object(usage_mod, "validate_file_path", return_value=str(session_file)),
            patch.object(usage_mod, "configured_acp_backend", return_value=""),
            patch.object(
                usage_mod,
                "get_usage_cache",
                return_value={"credits_used": 10, "credits_plan": 100, "plan": "Pro"},
            ),
        ):
            billing = await self._billing()
        assert billing["credits_used"] == 10
        assert billing["credits_plan"] == 100
        assert "provider" not in billing

    @pytest.mark.asyncio
    async def test_kiro_backend_without_a_plan_stays_empty(self, tmp_path, codex_root):
        kiro_dir = tmp_path / "cli"
        kiro_dir.mkdir()
        session_file = kiro_dir / "s1.jsonl"
        session_file.write_text(json.dumps({"kind": "Prompt"}) + "\n")
        with (
            patch.object(usage_mod, "_SESSIONS_DIR", kiro_dir),
            patch.object(usage_mod, "validate_file_path", return_value=str(session_file)),
            patch.object(usage_mod, "configured_acp_backend", return_value=""),
            patch.object(usage_mod, "get_usage_cache", return_value={"available": False}),
        ):
            billing = await self._billing()
        assert billing == {}
