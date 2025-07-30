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

# Import the flight check system
from flight_checker import FlightChecker, VerbosityLevel

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
                        # Clean tool schema for Anthropic API - remove annotations and other unsupported fields
                        clean_tool = {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                        }
                        # Remove any annotations or other fields that might cause issues
                        self.available_tools.append(clean_tool)
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
            config_file = "server_config.json"
            if not os.path.exists(config_file):
                print(f"No {config_file} found. Running without MCP servers.")
                return

            with open(config_file, "r") as file:
                content = file.read().strip()
                if not content:
                    print(f"{config_file} is empty. Running without MCP servers.")
                    return

                data = json.loads(content)

            servers = data.get("mcpServers", {})

            if not servers:
                print("No servers configured. Running without MCP servers.")
                return

            connected_count = 0
            for server_name, server_config in servers.items():
                try:
                    await self.connect_to_server(server_name, server_config)
                    connected_count += 1
                except Exception as e:
                    print(f"Failed to connect to {server_name}, skipping: {e}")
                    continue

            if connected_count > 0:
                print(f"Successfully connected to {connected_count} server(s)")
                # Initialize flight checker after successful connections
                # Use basic mode to avoid duplicates and focus on core functionality
                self.flight_checker = FlightChecker(
                    self,
                    test_mode="basic",  # Only generate basic functionality tests
                    load_learned_tests=False,  # Don't load learned tests to avoid duplicates
                )
            else:
                print("No servers connected successfully")

        except json.JSONDecodeError as e:
            print(f"Error parsing server config JSON: {e}")
            print("Running without MCP servers...")
        except Exception as e:
            print(f"Error loading server config: {e}")
            print("Running without MCP servers...")

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

                    # Use the new method that handles sampling oversight
                    try:
                        result = await self.call_tool_with_sampling_oversight(content)
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

    async def call_tool_with_sampling_oversight(self, content):
        """Call tool but intercept any sampling requests it makes"""
        session = self.sessions.get(content.name)
        if not session:
            raise Exception(f"Tool '{content.name}' not found in sessions.")

        # Store original create_message method if it exists
        original_create_message = None
        if hasattr(session, "create_message"):
            original_create_message = session.create_message
            # Replace with our oversight version
            session.create_message = self.handle_sampling_with_oversight

        try:
            result = await session.call_tool(content.name, arguments=content.input)
            return result
        finally:
            # Restore original create_message method
            if original_create_message:
                session.create_message = original_create_message

    async def handle_sampling_with_oversight(self, **sampling_params):
        """Handle MCP sampling requests with human oversight"""

        print("\n" + "=" * 60)
        print("AUTONOMOUS SYSTEM REQUESTING HUMAN GUIDANCE")
        print("=" * 60)

        # Extract the question the autonomous system is asking
        messages = sampling_params.get("messages", [])
        if not messages:
            print("No messages in sampling request")
            return self._create_rejection_response()

        user_message = (
            messages[0].content.text
            if hasattr(messages[0].content, "text")
            else str(messages[0].content)
        )
        system_prompt = sampling_params.get("systemPrompt", "")

        if system_prompt:
            print(
                f"System context: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}"
            )

        print("\nAutonomous system is asking:")
        print("-" * 40)
        print(user_message)
        print("-" * 40)

        # Human oversight decision
        while True:
            print("\nWhat should I do?")
            choice = (
                input(
                    "[approve] Send to LLM / [reject] Deny request / [modify] Change question: "
                )
                .lower()
                .strip()
            )

            if choice == "approve":
                print("Sending to LLM for guidance...")
                break
            elif choice == "reject":
                print("Request denied by human supervisor")
                return self._create_rejection_response()
            elif choice == "modify":
                new_question = input("Enter your modified question: ")
                # Update the message content
                if hasattr(messages[0].content, "text"):
                    messages[0].content.text = new_question
                print("Modified request approved")
                break
            else:
                print("Please enter 'approve', 'reject', or 'modify'")

        # Send approved request to LLM
        anthropic_messages = []
        for msg in messages:
            content_text = (
                msg.content.text if hasattr(msg.content, "text") else str(msg.content)
            )
            anthropic_messages.append({"role": msg.role, "content": content_text})

        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514-v1-birthright",
            messages=anthropic_messages,
            system=sampling_params.get("systemPrompt"),
            max_tokens=sampling_params.get("max_tokens", 200),
            temperature=sampling_params.get("temperature", 0.7),
        )

        # Show human the LLM's response
        llm_response = response.content[0].text
        print(f"\nLLM Response:")
        print("-" * 40)
        print(llm_response)
        print("-" * 40)

        # Human can approve/modify the response
        while True:
            choice = (
                input(
                    "Send this response to autonomous system? [approve/modify/reject]: "
                )
                .lower()
                .strip()
            )

            if choice == "approve":
                print("Response approved and sent to autonomous system")
                return self._create_mcp_style_response(llm_response)
            elif choice == "modify":
                modified_response = input("Enter your modified response: ")
                print("Modified response sent to autonomous system")
                return self._create_mcp_style_response(modified_response)
            elif choice == "reject":
                print("Response rejected - sending default response")
                return self._create_rejection_response()
            else:
                print("Please enter 'approve', 'modify', or 'reject'")

    def _create_mcp_style_response(self, content):
        """Create a response in the format expected by MCP sampling"""
        from mcp.types import TextContent

        return type(
            "MockResponse", (), {"content": TextContent(type="text", text=content)}
        )()

    def _create_rejection_response(self):
        """Create a rejection response"""
        return self._create_mcp_style_response("skip")

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
                    elif command == "/tools":
                        print("\nAvailable tools:")
                        for tool in self.available_tools:
                            print(f"- {tool['name']}: {tool['description']}")
                    elif command == "/test-cases":
                        if self.flight_checker:
                            print("\nGenerated test cases:")
                            for (
                                tool_name,
                                test_list,
                            ) in self.flight_checker.test_cases.items():
                                print(f"\n{tool_name}:")
                                for test_case in test_list:
                                    print(
                                        f"  - {test_case.test_name}: {test_case.description}"
                                    )
                                    if test_case.context_requirements:
                                        print(
                                            f"    Context needed: {', '.join(test_case.context_requirements)}"
                                        )
                        else:
                            error_print("Flight checker not available")
                    else:
                        error_print(f"Unknown command: {command}")
                        print(
                            "Available commands: /prompts, /prompt, /tools, /test-cases, /flight-check"
                        )
                    continue

                await self.process_query(query)

            except Exception as e:
                error_print(f"Error: {str(e)}")

    async def cleanup(self):
        # Clean up flight checker test files if they exist
        if hasattr(self, "flight_checker") and self.flight_checker:
            self.flight_checker.cleanup_test_files()

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
