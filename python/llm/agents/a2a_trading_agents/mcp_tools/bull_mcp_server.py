"""Bull Agent MCP Server - Opportunity-focused market analysis tools.

Run as a module from the project root so the `mcp_tools` package resolves:

    python -m mcp_tools.bull_mcp_server

The agents spawn this themselves over stdio; you only run it by hand to debug.
"""

from mcp_tools.bull_tools import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
