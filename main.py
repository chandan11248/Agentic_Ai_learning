from fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Calculator")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()
