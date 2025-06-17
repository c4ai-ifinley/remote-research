MCP Research Chatbot
An intelligent chatbot with Model Context Protocol (MCP) integration, featuring automated tool testing and DSPy-powered prompt optimization.

🚀 Quick Setup
1. Clone and Install Dependencies
bash
git clone <your-repo>
cd <your-repo>
uv sync
2. Environment Configuration
bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or your preferred editor
Your .env file should look like:

bash
ANTHROPIC_API_KEY=sk-your-actual-anthropic-key
OPENAI_API_KEY=sk-your-actual-openai-key
BASE_URL=https://ai-incubator-api.pnnl.gov
3. Run the Chatbot
bash
uv run mcp_chatbot.py
🛠 Features
MCP Integration: Connect to multiple tools (filesystem, research, fetch)
Automated Flight Checks: Test all tools before operation
DSPy Optimization: AI-powered prompt improvement when tests fail
Self-Improving: System learns from failures and optimizes prompts
🔧 Manual Commands
/flight-check [verbosity] - Run diagnostics (quiet/minimal/normal/verbose/debug)
/prompts - List available prompts
@folders - See available research topics
@topic - Search papers in specific topic
📊 Flight Check System
The system automatically tests all tools on startup with configurable verbosity:

Quiet: Only final summary
Minimal: Tool names and results (default)
Normal: Include test descriptions
Verbose: Show response previews
Debug: Full diagnostic output
Failed tests trigger automatic DSPy optimization using the o3-mini model for intelligent prompt improvement.

