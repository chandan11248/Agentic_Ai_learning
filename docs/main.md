# main.py Documentation

## Overview
A minimal **MCP (Model Context Protocol) Server** built with `FastMCP`. Exposes simple calculator tools (`add`, `multiply`) that can be called by MCP-compatible clients.

---

## Core Idea
| Concept | Description |
|---------|-------------|
| **FastMCP** | High-level wrapper for creating MCP tool servers quickly. |
| **@mcp.tool()** | Decorator that registers a function as a remotely-callable tool. |
| **mcp.run()** | Starts the server, listening for tool invocations. |

---

## Syntax Breakdown

```python
from fastmcp import FastMCP

# 1. Instantiate server with a name
mcp = FastMCP("Calculator")

# 2. Register tools using decorator
@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

# 3. Run the server
if __name__ == "__main__":
    mcp.run()
```

---

## How to Run
```bash
python main.py
```
The server will start and wait for MCP client connections.

---

## Dependencies
- `fastmcp`
