"""ACP `clientCapabilities` advertisement.

Locks in what KiroCrew declares during the ACP `initialize` handshake, and that
BOTH transports declare it. Before this, the key was omitted entirely, so the
agent assumed the all-false default.
"""

from pathlib import Path

from kiro_crew.acp.types import (
    ACP_CLIENT_CAPABILITIES,
    ACP_CLIENT_CAPABILITIES_SPEC_ADAPTER,
)


def test_elicitation_is_declared() -> None:
    """kiro-cli gates `elicitation/create` on this capability being present."""
    assert ACP_CLIENT_CAPABILITIES["elicitation"] == {"form": {}, "url": {}}


def test_spec_adapter_does_not_declare_elicitation() -> None:
    """The spec-adapter set must NOT advertise elicitation.

    codex-acp gates MCP tool-call approvals on
    ``clientCapabilities.elicitation.form``: declared → every approval arrives
    as an ``elicitation/create`` request, which Kiro Crew rejects as unknown and
    codex-acp converts into ``action: "cancel"`` — silently cancelling every
    MCP tool call in the session. Absent → codex-acp falls back to
    ``session/request_permission``, which the normal approval pipeline serves.
    """
    assert "elicitation" not in ACP_CLIENT_CAPABILITIES_SPEC_ADAPTER


def test_spec_adapter_set_tracks_the_base_set_otherwise() -> None:
    """Apart from elicitation the two sets must not drift.

    The spec-adapter dict is derived from the base dict, so a capability added
    to the base automatically reaches spec adapters — this pins that contract.
    """
    expected = {k: v for k, v in ACP_CLIENT_CAPABILITIES.items() if k != "elicitation"}
    assert ACP_CLIENT_CAPABILITIES_SPEC_ADAPTER == expected


def test_fs_and_terminal_stay_false() -> None:
    """We serve no `fs/*` or `terminal/*` handler, so we must not advertise them.

    Advertising either would invite inbound requests that
    `_reject_unknown_server_request` turns into errors.
    """
    assert ACP_CLIENT_CAPABILITIES["fs"] == {
        "readTextFile": False,
        "writeTextFile": False,
    }
    assert ACP_CLIENT_CAPABILITIES["terminal"] is False


def test_both_acp_transports_send_capabilities() -> None:
    """Both transports must advertise, not just one.

    `AcpClient` and `AcpRuntime` build their `initialize` params independently,
    so a capability added to one silently stays dark on the other. Asserted on
    source because neither params dict is reachable without spawning a real
    agent subprocess.
    """
    for rel in ("src/kiro_crew/acp/client.py", "src/kiro_crew/acp/runtime.py"):
        src = Path(__file__).resolve().parents[1] / rel
        # encoding is explicit: read_text() defaults to the locale codec, which
        # is cp1252 on the Windows CI shards, and these files contain non-ASCII
        # (em dashes / arrows) in their comments.
        assert "ACP_CLIENT_CAPABILITIES" in src.read_text(encoding="utf-8"), rel
