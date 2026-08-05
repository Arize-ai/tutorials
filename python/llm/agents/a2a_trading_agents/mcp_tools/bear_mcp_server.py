"""Bear Agent MCP Server - Risk-focused market analysis tools.

Run as a module from the project root so the `mcp_tools` package resolves:

    python -m mcp_tools.bear_mcp_server

The agents spawn this themselves over stdio; you only run it by hand to debug.
"""

from mcp_tools.bear_tools import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
