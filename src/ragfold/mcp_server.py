"""Model Context Protocol (MCP) surface for ragfold.

ragfold ships the same capabilities through three surfaces: the Python API, the
`ragfold` CLI, and this MCP server for AI agents. The tool *functions* below are
plain async callables with no MCP dependency, so they are importable and
testable in the light CI path. The actual server (`build_server`) is gated
behind the optional `mcp` extra and imports the SDK lazily.

Run it with `ragfold-mcp` (after `pip install ragfold[mcp]`).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from typing import Any

from ragfold.engines import build_default_router
from ragfold.engines.router import EngineRouter

CorpusArg = Sequence[Mapping[str, Any] | str]
QueryArg = Sequence[Mapping[str, Any] | str]


def list_engines(router: EngineRouter) -> list[dict[str, Any]]:
    """List registered engines and their availability/capabilities."""

    return router.list_engines()


async def retrieve(
    router: EngineRouter,
    corpus: CorpusArg,
    query: str,
    *,
    engine: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve with a single engine (auto-selected when `engine` is None)."""

    result = await router.retrieve(corpus, query, top_k=top_k, engine_hint=engine)
    return result.to_dict()


async def hybrid_search(
    router: EngineRouter,
    corpus: CorpusArg,
    query: str,
    *,
    engines: list[str],
    top_k: int = 5,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run several engines and fuse them with Reciprocal Rank Fusion."""

    result = await router.retrieve_hybrid(
        corpus, query, engines=engines, top_k=top_k, k=k, weights=weights
    )
    return result.to_dict()


async def compare(
    router: EngineRouter,
    corpus: CorpusArg,
    queries: QueryArg,
    *,
    engines: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Compare engines across queries, returning per-engine result lists."""

    results = await router.compare(corpus, queries, engines=engines, top_k=top_k)
    return {name: [item.to_dict() for item in items] for name, items in results.items()}


def build_server(router: EngineRouter | None = None) -> Any:
    """Build a FastMCP server exposing ragfold's retrieval tools.

    Requires the optional `mcp` extra. Raises a clear ImportError otherwise.
    """

    if importlib.util.find_spec("mcp") is None:
        raise ImportError(
            "MCP support requires the optional dependency. Install it with "
            "`pip install ragfold[mcp]`."
        )

    # The server class was renamed FastMCP -> MCPServer in mcp 2.x; both expose
    # the same `.tool()` decorator and `.run()` used here. A genuine import
    # error past this point (API drift) surfaces as-is, not as "install the extra".
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2.0
    except ImportError:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x

    active_router = router or build_default_router()
    server = _Server("ragfold")

    @server.tool()
    def ragfold_list_engines() -> list[dict[str, Any]]:
        """List registered engines with availability and capabilities."""
        return list_engines(active_router)

    @server.tool()
    async def ragfold_retrieve(
        corpus: CorpusArg,
        query: str,
        engine: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve passages with a single (optionally named) engine."""
        return await retrieve(active_router, corpus, query, engine=engine, top_k=top_k)

    @server.tool()
    async def ragfold_hybrid_search(
        corpus: CorpusArg,
        query: str,
        engines: list[str],
        top_k: int = 5,
        k: int = 60,
        weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Fuse several engines for one query with Reciprocal Rank Fusion."""
        return await hybrid_search(
            active_router, corpus, query, engines=engines, top_k=top_k, k=k, weights=weights
        )

    @server.tool()
    async def ragfold_compare(
        corpus: CorpusArg,
        queries: QueryArg,
        engines: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Compare engines across queries, returning per-engine results."""
        return await compare(active_router, corpus, queries, engines=engines, top_k=top_k)

    return server


def main() -> None:
    """Entry point for the `ragfold-mcp` console script (stdio transport)."""

    build_server().run()


if __name__ == "__main__":
    main()
