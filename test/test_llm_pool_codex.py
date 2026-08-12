"""AcpWorker.start() must thread the configured ACP backend into AcpClient.

Without this, the knowledge-extraction pool always constructs a kiro-cli
``AcpClient`` (``acp_backend=""``), which raises/spawns nothing usable on a
codex-only host with no kiro-cli binary installed. See
docs/system-specs/modules/memory-skills-hooks.md for the knowledge subsystem
and docs/system-specs/modules/acp-client.md for ``configured_acp_backend``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kiro_crew.acp.types import ACP_BACKEND_CODEX
from kiro_crew.knowledge.llm_pool import AcpWorker


class TestAcpWorkerBackendSelection:
    @pytest.mark.asyncio
    async def test_start_passes_configured_codex_backend_to_client(self):
        """With agent.acp_backend="codex" on disk, the worker's AcpClient is
        constructed with acp_backend="codex" instead of defaulting to kiro-cli.
        """
        mock_client = AsyncMock()
        mock_client.is_ready = True
        with (
            patch(
                "kiro_crew.knowledge.llm_pool.configured_acp_backend",
                return_value=ACP_BACKEND_CODEX,
            ),
            patch("kiro_crew.knowledge.llm_pool.AcpClient", return_value=mock_client) as mk,
        ):
            worker = AcpWorker(sandbox_mode="off")
            await worker.start()
        assert mk.call_args.kwargs["acp_backend"] == ACP_BACKEND_CODEX

    @pytest.mark.asyncio
    async def test_start_defaults_backend_to_kiro(self):
        """With no configured backend (the default), AcpClient still gets an
        explicit acp_backend="" (kiro-cli), matching prior default behavior.
        """
        mock_client = AsyncMock()
        mock_client.is_ready = True
        with (
            patch("kiro_crew.knowledge.llm_pool.configured_acp_backend", return_value=""),
            patch("kiro_crew.knowledge.llm_pool.AcpClient", return_value=mock_client) as mk,
        ):
            worker = AcpWorker(sandbox_mode="off")
            await worker.start()
        assert mk.call_args.kwargs["acp_backend"] == ""

    @pytest.mark.asyncio
    async def test_start_resolves_backend_at_start_time_not_import_time(self):
        """A respawn (second start() call) re-reads the configured backend
        rather than caching the value seen on the worker's first start.
        """
        mock_client_1 = AsyncMock()
        mock_client_1.is_ready = True
        mock_client_2 = AsyncMock()
        mock_client_2.is_ready = True

        with (
            patch("kiro_crew.knowledge.llm_pool.configured_acp_backend", return_value=""),
            patch("kiro_crew.knowledge.llm_pool.AcpClient", return_value=mock_client_1) as mk,
        ):
            worker = AcpWorker(sandbox_mode="off")
            await worker.start()
        assert mk.call_args.kwargs["acp_backend"] == ""

        with (
            patch(
                "kiro_crew.knowledge.llm_pool.configured_acp_backend",
                return_value=ACP_BACKEND_CODEX,
            ),
            patch("kiro_crew.knowledge.llm_pool.AcpClient", return_value=mock_client_2) as mk,
        ):
            await worker.start()
        assert mk.call_args.kwargs["acp_backend"] == ACP_BACKEND_CODEX
