# hitl_chatbot_demo.py Documentation

## Overview
A **Human-in-the-Loop (HITL) Chatbot** using **LangGraph** and **LangChain**. The bot can fetch stock prices and simulate stock purchases—but purchases require explicit human approval before execution.

---

## Core Idea
| Concept | Description |
|---------|-------------|
| **LangGraph StateGraph** | Defines a directed graph of nodes (LLM, tools) with conditional edges. |
| **interrupt()** | Pauses execution inside a tool, returning control to the caller for human input. |
| **Command(resume=...)** | Resumes paused execution with the human's decision. |
| **MemorySaver** | In-memory checkpointer that persists conversation state across turns. |
| **tools_condition** | Built-in router that checks if the LLM requested a tool call. |

---

## Architecture

```
START ──▶ chat_node ──▶ (tool call?) ──▶ tools ──▶ chat_node ──▶ ...
              │                            │
              └─── (no tool) ──▶ END       └─── interrupt() pauses here
```

---

## Key Syntax

### 1. Define a tool with HITL interrupt
```python
from langgraph.types import interrupt

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    
    if decision.lower() == "yes":
        return {"status": "success", ...}
    return {"status": "cancelled", ...}
```

### 2. Build the graph
```python
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=MemorySaver())
```

### 3. Handle interrupt in the loop
```python
result = chatbot.invoke(state, config={"configurable": {"thread_id": thread_id}})

interrupts = result.get("__interrupt__", [])
if interrupts:
    decision = input("Your decision: ")
    result = chatbot.invoke(Command(resume=decision), config=...)
```

---

## How to Run
```bash
# Ensure GROQ_API_KEY is set in .env
python hitl_chatbot_demo.py
```
Then chat; when you ask to buy stock, the bot will pause for your approval.

---

## Dependencies
- `langgraph`
- `langchain-core`
- `langchain-groq`
- `python-dotenv`
- `requests`
