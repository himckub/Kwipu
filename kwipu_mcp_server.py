"""
Kwipu MCP Server - Expose your knowledge graph to AI agents.

This server implements the Model Context Protocol (MCP) to allow
Claude Code, Cursor, Windsurf, VS Code Copilot, and other MCP-compatible
clients to query your Obsidian vault / knowledge base.

All queries are processed locally by Ollama, minimizing token usage
on the client side.

Usage:
    # As MCP server in Claude Desktop (claude_desktop_config.json):
    {
        "mcpServers": {
            "kwipu": {
                "command": "C:/path/to/python.exe",
                "args": ["C:/path/to/kwipu_mcp_server.py"]
            }
        }
    }
"""

import os
import sys

# Ensure UTF-8 on Windows
os.environ["PYTHONUTF8"] = "1"

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nest_asyncio
nest_asyncio.apply()

from mcp.server.fastmcp import FastMCP

# Set working directory to script location (resolves relative paths)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

from geode_graph import (
    WritHerGraphRAG,
    _init_llm,
)

# ==========================================
# MCP SERVER
# ==========================================
mcp = FastMCP("Kwipu")

# Redirect engine logs to stderr (stdout is reserved for MCP protocol)
import geode_graph

def _stderr_print(*args, **kwargs):
    kwargs["file"] = sys.stderr
    try:
        print(*args, **kwargs)
    except Exception:
        pass

geode_graph.safe_print = _stderr_print

# Lazy RAG instance
_rag_instance: WritHerGraphRAG | None = None


def _get_rag() -> WritHerGraphRAG:
    """Get or create the RAG engine (lazy, on first query)."""
    global _rag_instance
    if _rag_instance is None:
        _init_llm()
        _rag_instance = WritHerGraphRAG(fast_mode=True)
    return _rag_instance


# ==========================================
# TOOL
# ==========================================
@mcp.tool()
def query_graph(question: str) -> str:
    """Ask a question about your knowledge base. Searches across all notes
    using a knowledge graph with vector similarity, BM25, and temporal matching.
    Returns an answer with cited source files. Supports multiple languages.

    Args:
        question: Your question in natural language
    """
    rag = _get_rag()
    response = rag.ask(question)
    return str(response)


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    mcp.run()
