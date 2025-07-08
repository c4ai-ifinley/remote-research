from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import json
import asyncio
import nest_asyncio
import os
import traceback

# Import the enhanced flight check system
from enhanced_flight_check import EnhancedFlightChecker, VerbosityLevel

# Import color utilities
from color_utils import (
    system_print,
    error_print,
    success_print,
    warning_print,
    chat_input_print,
    colored_print,
    Colors,
    header_print,
)

nest_asyncio.apply()

load_dotenv()


class MCP_ChatBot:
    def __init__(self):
        self.exit_stack = AsyncExitStack()

        # Get configuration from environment variables
        base_url = os.getenv("BASE_URL", "https://ai-incubator-api.pnnl.gov")
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        # Initialize Anthropic client with custom base URL and API key
        self.anthropic = Anthropic(base_url=base_url, api_key=api_key)

        system_print(f"Initialized Anthropic client with base URL: {base_url}")

        # Tools list required for Anthropic API
        self.available_tools = []
        # Prompts list for quick display
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}

        # Initialize flight checker
        self.flight_checker = None  # Will be initialized after connections

    async def connect_to_server(self, server_name, server_config):
        try:
            print(f"Attempting to connect to {server_name}...")
            print(f"Server config: {server_config}")

            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            print(f"Initializing session for {server_name}...")
            await session.initialize()
            print(f"Session initialized for {server_name}")

            try:
                # List available tools
                print(f"Listing tools for {server_name}...")
                response = await session.list_tools()
                if response and response.tools:
                    print(f"Found {len(response.tools)} tools for {server_name}")
                    for tool in response.tools:
                        self.sessions[tool.name] = session
                        self.available_tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                                "annotations": tool.annotations,
                            }
                        )
                else:
                    print(f"No tools found for {server_name}")

                # List available prompts
                print(f"Listing prompts for {server_name}...")
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    print(
                        f"Found {len(prompts_response.prompts)} prompts for {server_name}"
                    )
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append(
                            {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": prompt.arguments,
                            }
                        )
                else:
                    print(f"No prompts found for {server_name}")

                # List available resources
                print(f"Listing resources for {server_name}...")
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    print(
                        f"Found {len(resources_response.resources)} resources for {server_name}"
                    )
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
                else:
                    print(f"No resources found for {server_name}")

                print(f"Successfully connected to {server_name}")

            except Exception as e:
                print(f"Error listing capabilities for {server_name}: {e}")
                print(f"Error type: {type(e).__name__}")

        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            print(f"Full traceback: {traceback.format_exc()}")

    async def connect_to_servers(self):
        try:
            # Check if server_config.json exists
            config_file = (
                "server_config.json"  # Change to "test_config.json" for testing
            )
            if not os.path.exists(config_file):
                print(f"No {config_file} found. Running without MCP servers.")
                return

            with open(config_file, "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})

            if not servers:
                print("No servers configured. Running without MCP servers.")
                return

            for server_name, server_config in servers.items():
                try:
                    await self.connect_to_server(server_name, server_config)
                except Exception as e:
                    print(f"Failed to connect to {server_name}, skipping: {e}")
                    continue

            # Initialize enhanced flight checker after all connections are established
            self.flight_checker = EnhancedFlightChecker(self)

        except Exception as e:
            print(f"Error loading server config: {e}")
            print("Continuing without MCP servers...")
            # Don't raise the exception, continue without serversload(file)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server config: {e}")
            raise

    async def process_query(self, query):
        messages = [{"role": "user", "content": query}]

        while True:
            response = self.anthropic.messages.create(
                max_tokens=2024,
                model="claude-sonnet-4-20250514-v1-birthright",
                tools=self.available_tools,
                messages=messages,
            )

            assistant_content = []
            has_tool_use = False

            for content in response.content:
                if content.type == "text":
                    print(content.text)
                    assistant_content.append(content)
                elif content.type == "tool_use":
                    has_tool_use = True
                    assistant_content.append(content)

                    # Get session and call tool
                    session = self.sessions.get(content.name)
                    if not session:
                        print(f"Tool '{content.name}' not found in sessions.")
                        print(f"Available sessions: {list(self.sessions.keys())}")
                        # Add a tool result indicating the tool wasn't found
                        messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content.id,
                                        "content": f"Error: Tool '{content.name}' not available",
                                        "is_error": True,
                                    }
                                ],
                            }
                        )
                        continue

                    try:
                        print(
                            f"Calling tool: {content.name} with args: {content.input}"
                        )
                        result = await session.call_tool(
                            content.name, arguments=content.input
                        )
                        messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content.id,
                                        "content": result.content,
                                    }
                                ],
                            }
                        )
                    except Exception as e:
                        print(f"Error calling tool {content.name}: {e}")
                        messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content.id,
                                        "content": f"Error: {str(e)}",
                                        "is_error": True,
                                    }
                                ],
                            }
                        )

            # Exit loop if no tool was used
            if not has_tool_use:
                break

    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)

        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break

        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return

        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")

    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return

        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt["arguments"]:
                print(f"  Arguments:")
                for arg in prompt["arguments"]:
                    arg_name = arg.name if hasattr(arg, "name") else arg.get("name", "")
                    print(f"    - {arg_name}")

    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return

        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content

                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, "text"):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(
                        item.text if hasattr(item, "text") else str(item)
                        for item in prompt_content
                    )

                print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            print(f"Error: {e}")

    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")
        print(
            f"Available prompts: {[prompt['name'] for prompt in self.available_prompts]}"
        )

        # Run flight check if we have tools and flight checker
        if self.flight_checker and self.available_tools:
            system_print("Running system flight check...")

            # Use verbose verbosity to see optimization in action
            flight_report = await self.flight_checker.run_flight_check(
                parallel=False, verbosity=VerbosityLevel.VERBOSE
            )

            if not flight_report.system_ready:
                warning_print("System not ready for full operation!")
                user_input = input("Continue anyway? (y/N): ").strip().lower()
                if user_input != "y":
                    error_print("Aborting startup due to critical failures.")
                    return

            # Export report for debugging
            self.flight_checker.export_report(flight_report)

        print("\nType your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")
        print("Use /flight-check to run diagnostics again")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue

                if query.lower() == "quit":
                    break

                # Handle flight check command
                if query.lower().startswith("/flight-check"):
                    if self.flight_checker:
                        parts = query.split()
                        verbosity = VerbosityLevel.NORMAL  # Default for manual runs

                        if len(parts) > 1:
                            verbosity_map = {
                                "quiet": VerbosityLevel.QUIET,
                                "minimal": VerbosityLevel.MINIMAL,
                                "normal": VerbosityLevel.NORMAL,
                                "verbose": VerbosityLevel.VERBOSE,
                                "debug": VerbosityLevel.DEBUG,
                            }
                            verbosity_str = parts[1].lower()
                            verbosity = verbosity_map.get(
                                verbosity_str, VerbosityLevel.NORMAL
                            )

                        await self.flight_checker.run_flight_check(
                            parallel=False, verbosity=verbosity
                        )
                    else:
                        error_print("Flight checker not available (no tools connected)")
                    continue

                # Check for @resource syntax first
                if query.startswith("@"):
                    # Remove @ sign
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue

                # Check for /command syntax
                if query.startswith("/"):
                    parts = query.split()
                    command = parts[0].lower()

                    if command == "/prompts":
                        await self.list_prompts()
                    elif command == "/prompt":
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue

                        prompt_name = parts[1]
                        args = {}

                        # Parse arguments
                        for arg in parts[2:]:
                            if "=" in arg:
                                key, value = arg.split("=", 1)
                                args[key] = value

                        await self.execute_prompt(prompt_name, args)
                    else:
                        error_print(f"Unknown command: {command}")
                    continue

                await self.process_query(query)

            except Exception as e:
                error_print(f"Error: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
