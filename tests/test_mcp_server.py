"""Tests for the MCP surface.

The MCP SDK is an optional extra. The tool *functions* are plain async
callables tested here without the SDK; building the actual server is gated and
degrades gracefully when `mcp` is not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ragfold import mcp_server
from ragfold.engines.bm25 import BM25Engine
from ragfold.engines.router import EngineRouter
from ragfold.engines.text_rag import TextRagEngine

CORPUS = [
    {"id": "a", "text": "Alpha invoice payment"},
    {"id": "b", "text": "Beta contract renewal"},
]


def _router() -> EngineRouter:
    return EngineRouter([BM25Engine(), TextRagEngine()])


def test_list_engines_tool_returns_rows():
    rows = mcp_server.list_engines(_router())

    assert [row["name"] for row in rows] == ["bm25", "text-rag"]


@pytest.mark.asyncio
async def test_retrieve_tool_returns_serializable_result():
    result = await mcp_server.retrieve(_router(), CORPUS, "invoice", engine="bm25", top_k=1)

    assert result["engine_name"] == "bm25"
    assert result["passages"][0]["document_id"] == "a"
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_hybrid_search_tool_fuses_engines():
    result = await mcp_server.hybrid_search(
        _router(), CORPUS, "invoice", engines=["bm25", "text-rag"], top_k=2
    )

    assert result["engine_name"] == "rrf(bm25,text-rag)"
    assert result["passages"][0]["document_id"] == "a"
    assert result["passages"][0]["metadata"]["fusion"]["consensus"] == 2


@pytest.mark.asyncio
async def test_hybrid_search_tool_accepts_weights():
    result = await mcp_server.hybrid_search(
        _router(),
        CORPUS,
        "invoice",
        engines=["bm25", "text-rag"],
        top_k=2,
        weights={"bm25": 2.0, "text-rag": 1.0},
    )

    assert result["metadata"]["fusion"]["weights"] == {"bm25": 2.0, "text-rag": 1.0}


def test_build_server_is_gated_when_mcp_missing():
    if importlib.util.find_spec("mcp") is not None:
        pytest.skip("mcp SDK is installed; gating path not exercised")

    with pytest.raises(Exception) as exc_info:  # noqa: PT011 - message is asserted below
        mcp_server.build_server()

    assert "ragfold[mcp]" in str(exc_info.value)


def test_mcp_extra_and_script_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "mcp = [" in pyproject
    assert "ragfold-mcp = " in pyproject
