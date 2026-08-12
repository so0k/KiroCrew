"""Read the Codex CLI's own rollout logs for usage and rate-limit data.

Codex writes one JSONL rollout per session under
``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ISO-TS>-<session-uuid>.jsonl`` and
appends a ``token_count`` event at the end of every turn carrying both the
session's token totals and the account's ChatGPT-subscription rate-limit
windows::

    {"timestamp": "…Z", "type": "event_msg", "payload": {
        "type": "token_count",
        "info": {"total_token_usage": {…}, "last_token_usage": {…},
                 "model_context_window": 258400},
        "rate_limits": {"primary": {"used_percent": 13.0,
                                    "window_minutes": 10080,
                                    "resets_at": 1787011261},
                        "secondary": null, "credits": {…},
                        "plan_type": "plus"}}}

This module is a SIDE CHANNEL, not the intended interface: codex-acp does not
forward any of it over ACP yet (upstream codex-acp issue #334), so the Usage
tab would otherwise have no billing source at all on a codex host. It is
deliberately read-only, self-contained, and depended on from exactly one place
(``dashboard/handlers/usage.py``) so it can be deleted wholesale once the data
arrives over the wire.

Two guarantees the callers rely on:

* **Only rollout logs are read.** Every byte comes from a descriptor opened by
  :func:`_open_rollout`, the module's one open path, which requires the
  candidate to canonicalize inside the resolved sessions root, to still be
  NAMED ``rollout-*.jsonl`` after canonicalization, and to ``fstat`` as a
  regular file behind an ``O_NOFOLLOW`` open. Containment alone is not enough:
  ``is_sensitive_path`` is anchored on ``$HOME``, so with the store relocated
  outside the home directory a symlink named ``rollout-…-<uuid>.jsonl`` would
  otherwise canonicalize onto ``$CODEX_HOME/auth.json`` — the live OAuth token
  pair — and a ``sessions`` root that is itself a symlink next to that file
  makes it resolve *inside* the root, where only its name gives it away. On a
  POSIX host that combination is what keeps a hostile name in the store from
  becoming a read primitive; on Windows, where ``O_NOFOLLOW`` does not exist,
  the name and containment gates still hold but a final-component swap is not
  closed (see :data:`_O_NOFOLLOW`).
* **Nothing raises.** Every public entry point returns ``None`` on a missing
  store, an unreadable file, or a malformed line, logging at debug. The data is
  a nice-to-have panel, never a reason to fail a dashboard request.

The filename ``<session-uuid>`` equals the ACP session id Kiro Crew stores in
``session_map``, so a per-session join needs no extra bookkeeping.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from kiro_crew.acp.client import codex_home_path
from kiro_crew.hooks import validate_file_path

logger = logging.getLogger(__name__)

# Rollout store layout: <root>/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl.
_SESSIONS_SUBDIR = "sessions"
_ROLLOUT_PREFIX = "rollout-"
_ROLLOUT_SUFFIX = ".jsonl"

# How long a resolved rate-limit snapshot is served without re-walking the
# store. The dashboard polls /api/usage/kiro on a timer and codex only appends
# a token_count event at the end of a turn, so a sub-minute TTL keeps the walk
# off the hot path while staying well inside the shortest (5h) rate-limit
# window's useful resolution.
RATE_LIMIT_CACHE_TTL_S = 30.0

# Tail window for locating the LAST token_count event. Rollout lines run ~1 KB
# and a token_count lands once per turn, so 64 KB covers the last few dozen
# turns of any session; a full read of a multi-MB transcript to answer one
# panel is what this bound exists to avoid.
_TAIL_READ_BYTES = 64 * 1024

# A file whose tail window yielded no token_count event is re-read line by line
# only below this size. That is the one case the tail cannot answer correctly:
# the sole token_count event can straddle the window's leading edge, whose
# partial first line must be discarded. Bounded so a pathological transcript
# still cannot turn one poll into an unbounded read.
_FULL_SCAN_MAX_BYTES = 4 * 1024 * 1024

# Bounds on the "newest rollout" search. Date directories are walked
# newest-first and only this many day directories are opened. The bound covers
# the WEEKLY rate-limit window (10080 minutes) with a day of slack, because a
# snapshot taken any time inside that window still says something true about
# the account's weekly quota — a host whose last codex turn was four days ago
# has a meaningful weekly percentage and must not report "no billing". Files
# are ranked by mtime across the whole set, since a *resumed* session keeps
# appending to the file under its ORIGINAL date, so the newest event does not
# always live in the newest day directory.
_LATEST_DAY_DIRS = 8

# Day directories examined by the ONE wider pass taken only when the bound
# above found nothing at all. A store whose newest turn predates the weekly
# window carries a stale snapshot, but a stale snapshot with its own
# ``captured_at`` beats an empty panel, and paying for the wider walk only on a
# total miss keeps it off the hot path.
_WIDE_DAY_DIRS = 30

# How many candidate files (newest mtime first) are opened per pass before
# giving up. A session that was started but never completed a turn carries no
# token_count event at all yet sorts newest on mtime, so the bound has to leave
# room for a run of such sessions to sit in front of the real snapshot: a
# handful of parallel or abandoned sessions would otherwise push it out of the
# candidate set entirely. Candidates that yield no event are skipped, not
# treated as the answer, until the bound is exhausted.
_LATEST_CANDIDATE_FILES = 24

# Session ids are UUIDs. Matched before interpolation into a glob pattern so a
# caller-supplied id can never carry a path separator or a wildcard. ``\Z``, not
# ``$``: ``$`` also matches before a trailing newline, so "<uuid>\n" would pass.
_SID_RE = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")

# Open flags for the one hardened open. ``O_NOFOLLOW`` refuses a final component
# that is a symlink, which on the ALREADY-canonical path can only mean a
# concurrent swap after the containment check (the mechanism
# ``hooks.safe_read_file`` uses for the same race). ``O_NONBLOCK`` is what makes
# ``open(2)`` of a FIFO return instead of blocking until a writer appears — a
# hostile ``rollout-…jsonl`` FIFO in the store would otherwise wedge the reader,
# and with it the usage handler holding its cache lock, forever. It is left set
# on the descriptor we keep: POSIX gives it no effect on a read of a REGULAR
# file, and the ``fstat`` below is what proves the file is one.
# Both flags are POSIX-only and resolve to 0 on Windows, where the containment,
# basename, and regular-file gates still run but a final-component symlink swap
# is not closed. Codex does not run there, and this module must stay importable.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)

_TOKEN_COUNT_MARKER = b'"token_count"'
_EVENT_TYPE = "token_count"


@dataclass(frozen=True)
class CodexRateLimitWindow:
    """One rate-limit window as codex reports it.

    ``window_minutes`` identifies the window (10080 = weekly, 300 = the
    shorter 5h window) and ``resets_at`` is epoch SECONDS, or ``None`` when
    the event carries no reset time.
    """

    used_percent: float
    window_minutes: int
    resets_at: int | None


@dataclass(frozen=True)
class CodexCredits:
    """Credit balance block. ``balance`` is a string in codex's own payload."""

    has_credits: bool
    unlimited: bool
    balance: str


@dataclass(frozen=True)
class CodexRateLimits:
    """The newest ``rate_limits`` snapshot found in the rollout store.

    ``secondary`` and ``credits`` are ``None`` when codex reported them as
    null, which it routinely does for ``secondary``. ``captured_at`` is the
    rollout event's own timestamp (ISO-8601, UTC) — the snapshot is as old as
    the last turn, so a consumer must be able to say how old.
    """

    plan_type: str
    primary: CodexRateLimitWindow | None
    secondary: CodexRateLimitWindow | None
    credits: CodexCredits | None
    captured_at: str


@dataclass(frozen=True)
class CodexSessionTokens:
    """Cumulative token usage for one session, plus its context window.

    Fields mirror codex's ``total_token_usage`` (cumulative over the session,
    not the last turn). ``context_window`` is ``model_context_window``, 0 when
    the event did not carry one.
    """

    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int
    context_window: int


# Cached rate-limit snapshot, keyed on the resolved store root it was read
# from. ``_rl_cache_ts == 0.0`` means cold; a non-zero timestamp with
# ``_rl_cache is None`` is a cached NEGATIVE result, so a host with no codex
# store does not re-walk it on every poll. The key makes a ``$CODEX_HOME``
# change (a test fixture, a re-pointed runtime) miss instead of serving another
# store's snapshot; one entry is enough, since a process reads one store at a
# time and a switch must invalidate rather than accumulate. Read and written
# from whichever executor thread serves the request; a lost race costs one
# extra walk, so no lock is taken.
_rl_cache: CodexRateLimits | None = None
_rl_cache_root: str = ""
_rl_cache_ts: float = 0.0


def clear_cache() -> None:
    """Drop the cached rate-limit snapshot.

    A test seam — no shipped code path calls it, since the TTL and the
    root-keyed entry already handle expiry and a re-pointed ``$CODEX_HOME``.
    """
    global _rl_cache, _rl_cache_root, _rl_cache_ts
    _rl_cache = None
    _rl_cache_root = ""
    _rl_cache_ts = 0.0


def codex_sessions_root() -> Path:
    """Root of the codex rollout store (``$CODEX_HOME/sessions``)."""
    return codex_home_path() / _SESSIONS_SUBDIR


def _resolved_root(root: Path) -> Path:
    """``root`` canonicalized, for both containment checks and the cache key.

    Non-strict: the store need not exist yet, and a missing directory must read
    as "no data", not as an error.
    """
    try:
        return root.resolve()
    except OSError:
        return root


def _numeric_subdirs_newest_first(parent: Path) -> list[Path]:
    """Date-component subdirectories of ``parent``, numerically descending.

    Non-numeric entries are ignored: the layout is YYYY/MM/DD, and anything
    else under it is not a date directory. Sorting on ``int`` rather than the
    name keeps an unpadded component (``8`` vs ``08``) in the right order.
    """
    try:
        entries = list(parent.iterdir())
    except OSError:
        return []
    dated: list[tuple[int, Path]] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        dated.append((int(entry.name), entry))
    return [path for _, path in sorted(dated, key=lambda pair: pair[0], reverse=True)]


def _day_dirs_newest_first(root: Path, limit: int) -> list[Path]:
    """Up to ``limit`` day directories under ``root``, newest date first."""
    out: list[Path] = []
    for year_dir in _numeric_subdirs_newest_first(root):
        for month_dir in _numeric_subdirs_newest_first(year_dir):
            for day_dir in _numeric_subdirs_newest_first(month_dir):
                out.append(day_dir)
                if len(out) >= limit:
                    return out
    return out


def _looks_like_a_rollout(name: str) -> bool:
    """Whether ``name`` is a rollout log's filename.

    Applied twice: to a directory entry while ranking candidates, and to the
    RESOLVED name in :func:`_open_rollout`, where it is a security gate.
    """
    return name.startswith(_ROLLOUT_PREFIX) and name.endswith(_ROLLOUT_SUFFIX)


def _rollout_files_by_mtime(day_dirs: list[Path]) -> list[Path]:
    """Rollout files across ``day_dirs``, most recently modified first.

    mtime rather than the filename timestamp: a resumed session appends to the
    file named for the day it STARTED, so the newest event is in the most
    recently written file, not the newest-named one.
    """
    stamped: list[tuple[float, Path]] = []
    for day_dir in day_dirs:
        try:
            entries = list(day_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not _looks_like_a_rollout(entry.name):
                continue
            try:
                stamped.append((entry.stat().st_mtime, entry))
            except OSError:
                continue
    return [path for _, path in sorted(stamped, key=lambda pair: pair[0], reverse=True)]


def _tail_lines(fh: BinaryIO) -> tuple[list[bytes], bool]:
    """Read the last ``_TAIL_READ_BYTES`` of the open rollout ``fh`` as lines.

    Returns ``(lines, complete)`` where ``complete`` is True only when the read
    covered the whole file. When it did not, the first element of the window is
    a partial line and is dropped — so a caller that finds nothing must decide
    whether the discarded head is worth a full scan.

    Takes an open descriptor, not a path: the tail read and the full-scan
    fallback share the ONE descriptor :func:`_open_rollout` vetted, so nothing
    is re-opened by name — and re-traversed — after its containment and file
    type were established.
    """
    fh.seek(0, os.SEEK_END)
    size = fh.tell()
    offset = max(0, size - _TAIL_READ_BYTES)
    fh.seek(offset)
    blob = fh.read()
    lines = blob.split(b"\n")
    if offset > 0:
        lines = lines[1:]
    return lines, offset == 0


def _parse_token_count(line: bytes) -> dict | None:
    """Parse one rollout line, returning it only if it is a token_count event."""
    if _TOKEN_COUNT_MARKER not in line:
        return None
    try:
        obj = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    payload = obj.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != _EVENT_TYPE:
        return None
    return obj


def _contained(path: Path, resolved_root: Path) -> Path | None:
    """``path`` canonicalized, or None when it escapes the rollout store.

    Two gates. ``validate_file_path`` refuses a path resolving onto a
    ``$HOME``-anchored sensitive location, but the store can be relocated
    outside ``$HOME``; requiring the RESOLVED path to stay under the resolved
    sessions root is what stops a symlink named like a rollout file inside a day
    directory from being a read primitive for arbitrary files,
    ``$CODEX_HOME/auth.json`` included.

    A containment check, not a symlink ban: a link to another rollout inside the
    store resolves inside the root and still reads. Containment is measured
    against the RESOLVED root, so it says nothing about a store root that is
    itself a link — :func:`_open_rollout` adds the two gates that do.
    """
    resolved_str = validate_file_path(str(path))
    if resolved_str is None:
        return None
    resolved = Path(resolved_str)
    if not resolved.is_relative_to(resolved_root):
        logger.debug("codex_usage: refusing %s, resolves outside %s", path, resolved_root)
        return None
    return resolved


def _open_rollout(path: Path, resolved_root: Path) -> BinaryIO | None:
    """A read-only handle on the rollout ``path``, or None if it is not one.

    The module's ONE open path: everything it reads comes from a descriptor this
    returns, so all four gates apply to every byte.

    1. :func:`_contained` — canonicalize, refuse a sensitive target, and require
       the result to stay inside the resolved sessions root.
    2. The RESOLVED name must still be a rollout name. Containment is measured
       against the resolved root, so a ``sessions`` root that is itself a symlink
       into a directory holding ``auth.json`` — or straight back onto
       ``$CODEX_HOME`` — makes that file resolve *inside* the root and pass gate
       1; its name is the only thing left that gives it away. Checking the
       resolved name rather than the candidate's own still admits a legitimate
       in-store rollout reached through a link.
    3. ``O_NOFOLLOW`` on the canonical path. Its final component is not a symlink
       by construction, so this rejects exactly one thing: a component swapped
       into a symlink between the check and the open (``ELOOP``/``EMLINK``).
    4. ``fstat`` + ``S_ISREG``, taken on the descriptor rather than the path, so
       a FIFO or directory named like a rollout is refused. ``O_NONBLOCK`` is
       what lets the FIFO case reach this check at all instead of hanging in
       ``open(2)``, and with it the whole usage endpoint.

    Never raises: a rejection — escaped, wrong name, non-regular, vanished,
    swapped — logs at debug and returns None, which every caller reads as
    "no data".
    """
    resolved = _contained(path, resolved_root)
    if resolved is None:
        return None
    if not _looks_like_a_rollout(resolved.name):
        logger.debug("codex_usage: refusing %s, resolves onto non-rollout %s", path, resolved)
        return None
    try:
        fd = os.open(resolved, os.O_RDONLY | _O_NONBLOCK | _O_NOFOLLOW)
    except OSError:
        logger.debug("codex_usage: cannot open %s", resolved, exc_info=True)
        return None
    try:
        is_regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        os.close(fd)
        logger.debug("codex_usage: cannot stat the open %s", resolved, exc_info=True)
        return None
    if not is_regular:
        os.close(fd)
        logger.debug("codex_usage: refusing %s, not a regular file", resolved)
        return None
    try:
        return os.fdopen(fd, "rb")
    except OSError:
        os.close(fd)
        logger.debug("codex_usage: cannot wrap the open %s", resolved, exc_info=True)
        return None


def _last_token_count(path: Path, resolved_root: Path, *, require_rate_limits: bool) -> dict | None:
    """The last token_count event in ``path``, or None.

    ``require_rate_limits`` skips events whose ``rate_limits`` is null, which a
    token_count event is allowed to be — the token totals are still there, so
    the two callers want different "last" events. ``resolved_root`` is the
    canonical store root the path must stay inside (see :func:`_open_rollout`).
    """

    def _pick(lines: list[bytes]) -> dict | None:
        for line in reversed(lines):
            obj = _parse_token_count(line)
            if obj is None:
                continue
            if require_rate_limits and not isinstance(obj["payload"].get("rate_limits"), dict):
                continue
            return obj
        return None

    handle = _open_rollout(path, resolved_root)
    if handle is None:
        return None
    try:
        with handle as fh:
            lines, complete = _tail_lines(fh)
            found = _pick(lines)
            if found is not None or complete:
                return found
            # The tail missed it. The event can straddle the window's leading
            # edge, so re-read line by line when the file is small enough to be
            # worth it — off the SAME descriptor, whose size comes from
            # ``fstat`` rather than a second lookup by name.
            if os.fstat(fh.fileno()).st_size > _FULL_SCAN_MAX_BYTES:
                return None
            fh.seek(0)
            return _pick(fh.readlines())
    except OSError:
        logger.debug("codex_usage: cannot read %s", path, exc_info=True)
        return None


def _as_int(value: object) -> int:
    """Int from a JSON scalar, 0 for null or anything non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _as_float(value: object) -> float:
    """Float from a JSON scalar, 0.0 for null or anything non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _as_opt_int(value: object) -> int | None:
    """Int from a JSON scalar, None for null or anything non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _window_from(raw: object) -> CodexRateLimitWindow | None:
    """One rate-limit window, or None when codex reported it as null."""
    if not isinstance(raw, dict):
        return None
    return CodexRateLimitWindow(
        used_percent=_as_float(raw.get("used_percent")),
        window_minutes=_as_int(raw.get("window_minutes")),
        resets_at=_as_opt_int(raw.get("resets_at")),
    )


def _credits_from(raw: object) -> CodexCredits | None:
    """The credits block, or None when codex reported it as null."""
    if not isinstance(raw, dict):
        return None
    balance = raw.get("balance")
    return CodexCredits(
        has_credits=bool(raw.get("has_credits")),
        unlimited=bool(raw.get("unlimited")),
        balance=str(balance) if balance is not None else "",
    )


def _rate_limits_from(event: dict) -> CodexRateLimits | None:
    """Build the snapshot from a token_count event, or None if it carries none."""
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    raw = payload.get("rate_limits")
    if not isinstance(raw, dict):
        return None
    primary = _window_from(raw.get("primary"))
    secondary = _window_from(raw.get("secondary"))
    credits = _credits_from(raw.get("credits"))
    # A snapshot with no window and no credits carries nothing to render.
    if primary is None and secondary is None and credits is None:
        return None
    plan_type = raw.get("plan_type")
    timestamp = event.get("timestamp")
    return CodexRateLimits(
        plan_type=str(plan_type) if isinstance(plan_type, str) else "",
        primary=primary,
        secondary=secondary,
        credits=credits,
        captured_at=str(timestamp) if isinstance(timestamp, str) else "",
    )


def _rate_limits_in(day_dirs: list[Path], resolved_root: Path) -> CodexRateLimits | None:
    """Newest rate-limit snapshot across ``day_dirs``, or None.

    Candidates are tried newest-mtime first and one that carries no rate-limited
    token_count event is skipped rather than ending the search.
    """
    for path in _rollout_files_by_mtime(day_dirs)[:_LATEST_CANDIDATE_FILES]:
        event = _last_token_count(path, resolved_root, require_rate_limits=True)
        if event is None:
            continue
        limits = _rate_limits_from(event)
        if limits is not None:
            return limits
    return None


def latest_rate_limits() -> CodexRateLimits | None:
    """Newest ChatGPT-subscription rate-limit snapshot in the rollout store.

    Blocking filesystem work — call it off the event loop. Returns None when
    there is no codex store, no rollout carrying a rate-limited token_count
    event within the searched bounds, or anything unreadable. Cached for
    ``RATE_LIMIT_CACHE_TTL_S`` per resolved store root, negative results
    included.
    """
    global _rl_cache, _rl_cache_root, _rl_cache_ts
    try:
        resolved_root = _resolved_root(codex_sessions_root())
    except Exception:
        # A best-effort side channel must never fail the request that reads it.
        logger.debug("codex_usage: cannot resolve the codex store", exc_info=True)
        return None
    key = str(resolved_root)
    now = time.monotonic()
    if _rl_cache_ts and _rl_cache_root == key and (now - _rl_cache_ts) < RATE_LIMIT_CACHE_TTL_S:
        return _rl_cache
    result: CodexRateLimits | None = None
    try:
        # Enumerating day directories is a listing of date components, cheap
        # enough to do once for both passes; only the per-pass slices are
        # stat'ed and read. The wider pass starts where the first one stopped,
        # so a total miss costs one walk of the store, not two.
        day_dirs = _day_dirs_newest_first(resolved_root, _WIDE_DAY_DIRS)
        result = _rate_limits_in(day_dirs[:_LATEST_DAY_DIRS], resolved_root)
        if result is None:
            result = _rate_limits_in(day_dirs[_LATEST_DAY_DIRS:], resolved_root)
    except Exception:
        logger.debug("codex_usage: rate-limit read failed", exc_info=True)
        result = None
    _rl_cache = result
    _rl_cache_root = key
    _rl_cache_ts = time.monotonic()
    return result


def session_token_usage(sid: str) -> CodexSessionTokens | None:
    """Cumulative token usage for the codex session ``sid``.

    ``sid`` is the ACP session id, which is also the rollout filename's uuid
    component. Blocking filesystem work — call it off the event loop. Returns
    None for an unknown/malformed id, a session with no completed turn, or any
    read error.

    A public seam with no consumer in the handler layer yet: the ``sessions``
    payload GET /api/usage/kiro returns carries no token fields on either
    backend, so there is nowhere to put a per-session total there. The two
    sound consumers are a per-DAY token sum for the Usage tab, summing the
    per-turn ``last_token_usage`` events rather than reading the cumulative
    total once, and the chat runner's live context payload, which wants this
    session's ``total_tokens`` against ``context_window``.
    """
    if not _SID_RE.match(sid or ""):
        return None
    try:
        resolved_root = _resolved_root(codex_sessions_root())
        pattern = f"*/*/*/{_ROLLOUT_PREFIX}*-{sid}{_ROLLOUT_SUFFIX}"
        matches: list[tuple[float, Path]] = []
        for path in resolved_root.glob(pattern):
            try:
                matches.append((path.stat().st_mtime, path))
            except OSError:
                continue
        for _, path in sorted(matches, key=lambda pair: pair[0], reverse=True):
            event = _last_token_count(path, resolved_root, require_rate_limits=False)
            if event is None:
                continue
            info = event["payload"].get("info")
            if not isinstance(info, dict):
                continue
            totals = info.get("total_token_usage")
            if not isinstance(totals, dict):
                continue
            return CodexSessionTokens(
                input_tokens=_as_int(totals.get("input_tokens")),
                cached_input_tokens=_as_int(totals.get("cached_input_tokens")),
                cache_write_input_tokens=_as_int(totals.get("cache_write_input_tokens")),
                output_tokens=_as_int(totals.get("output_tokens")),
                reasoning_output_tokens=_as_int(totals.get("reasoning_output_tokens")),
                total_tokens=_as_int(totals.get("total_tokens")),
                context_window=_as_int(info.get("model_context_window")),
            )
    except Exception:
        logger.debug("codex_usage: session token read failed for %s", sid, exc_info=True)
    return None
