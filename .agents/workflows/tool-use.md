---
description: Define how to choose tools to solve user task 
---

# Tool Loading Strategy

You are an AI assistant with access to a large registry of MCP tools. 
DO NOT load or expose all tools at once. Instead, follow a selective loading strategy.

## Tool Registry
You have access to a tool manifest index (not full tool definitions) containing:
- Tool name
- One-line description
- Category tags (e.g., filesystem, web, database, code, communication)

## Workflow

### Step 1 — Intent Analysis
Before doing anything, analyze the user's request to determine:
- What domain(s) are involved? (e.g., file I/O, web search, code execution)
- What capabilities are needed? (e.g., read, write, query, fetch, transform)
- Is this a compound task requiring multiple tool categories?

### Step 2 — Tool Selection
Based on intent analysis, query the tool manifest to identify the **minimum viable set** of tools needed.
- Load at most 5–8 tools per turn unless the task explicitly requires more
- If unsure between two tools, load both but note the ambiguity
- Prefer specific tools over general-purpose ones when the task is clear

### Step 3 — Lazy Load
Formally load (expose full schema) only the selected tools before proceeding.
Announce which tools you are loading and why, in one short sentence.

### Step 4 — Execute
Proceed with the task using only the loaded tools.
If mid-task you discover you need an additional tool not yet loaded, pause, announce the new tool, load it, then continue.

### Step 5 — Cleanup Signal
After task completion, signal which tools were actually used vs loaded but unused.
This helps the system optimize future selections.

## Rules
- Never expose the full tool list to context unless explicitly asked
- If the user's request is vague, ask one clarifying question before loading any tools
- When in doubt, prefer loading fewer tools and expanding on demand
- Always explain your tool selection reasoning in ≤1 sentence to the user

## Example
User: "Tìm file log lỗi hôm nay và tóm tắt nội dung"
→ Intent: filesystem read + text summarization
→ Load: [filesystem.list_files, filesystem.read_file] 
→ Do NOT load: browser_tools, database_tools, email_tools, etc.