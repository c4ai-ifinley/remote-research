# MCP Chatbot with Flight Checker

A chatbot system that connects to Model Context Protocol (MCP) servers with automated testing and human-guided optimization.

## Setup

### Prerequisites
```bash
pip install anthropic python-dotenv mcp arxiv colorama dspy-ai nest-asyncio
```

### Environment
Create `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
BASE_URL=https://your-endpoint.com  # optional
OPENAI_API_KEY=your_openai_key      # optional, for DSPy
```

### Run
```bash
python mcp_chatbot.py
```

## Files

**Core Files:**
- `mcp_chatbot.py` - Main chatbot application
- `flight_checker.py` - Automated testing system  
- `human_review_system.py` - Human-guided test fixing
- `dspy_optimizer.py` - AI prompt optimization
- `server_config.json` - MCP server configuration

**Supporting Files:**
- `color_utils.py` - Terminal colors
- `utils.py` - Utility functions
- `research_server.py` - Example MCP server
- `test_cases.json` - Generated test cases
- `failed_tests_review.json` - Failed tests for review

## MCP Protocol Features

**Tools** - Direct function calls:
```python
await session.call_tool("search_papers", {"topic": "AI"})
```

**Prompts** - Reusable templates:
```python
await session.get_prompt("generate_search_prompt", {"topic": "AI"})
```

**Resources** - Structured data access:
```python
await session.read_resource("papers://machine_learning")
```

**Sampling** - Human-in-the-loop decisions:
```python
# AI requests human guidance for complex decisions
result = await ctx.session.create_message(messages=[...])
```

## Usage

**Basic Commands:**
```
Query: List files in current directory
Query: @folders                    # List paper topics
Query: @machine_learning          # View ML papers  
Query: /flight-check              # Run diagnostics
```

**Human Review Workflow:**
1. Tests fail → `failed_tests_review.json` created
2. Edit file → Update prompts, change status to "fixed"
3. Run again → Fixes applied automatically

Example fix:
```json
{
  "current_prompt": "Test the read_file tool",
  "suggested_prompt": "Read files from current directory safely", 
  "status": "fixed"
}
```

## Configuration

**MCP Servers** (`server_config.json`):
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "research": {
      "command": "uv",
      "args": ["run", "research_server.py"]
    }
  }
}
```

**Flight Checker Modes:**
- `basic` - Core functionality testing only (recommended)
- `comprehensive` - Full testing including error handling

## Features

- Automated test generation from MCP schemas
- AI-powered prompt optimization with DSPy
- Human-guided test fixing via simple JSON editing
- Multi-server MCP support with automatic discovery
- Human oversight for autonomous AI decisions via MCP sampling