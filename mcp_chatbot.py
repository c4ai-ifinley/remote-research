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
from enhanced_flight_check import (
    EnhancedFlightChecker,
    FlightCheckIntegration,
    VerbosityLevel,
)

# Import the new prompt-based flight checker
from prompt_based_flight_checker import PromptBasedFlightChecker, TestResult

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

        # Enhanced flight checker integration
        self.flight_checker = None
        self.flight_check_enabled = (
            os.getenv("FLIGHT_CHECK_ENABLED", "true").lower() == "true"
        )
        self.flight_check_on_startup = (
            os.getenv("FLIGHT_CHECK_ON_STARTUP", "true").lower() == "true"
        )

        # Prompt-based flight checker integration
        self.prompt_checker = None
        self.prompt_check_enabled = (
            os.getenv("PROMPT_CHECK_ENABLED", "true").lower() == "true"
        )

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
            config_file = "server_config.json"
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

            # Initialize enhanced flight checker after all connections
            if self.flight_check_enabled and self.available_tools:
                await self._initialize_enhanced_flight_checker()

            # Initialize prompt-based flight checker
            if self.prompt_check_enabled and self.available_tools:
                await self._initialize_prompt_checker()

        except Exception as e:
            print(f"Error loading server config: {e}")
            print("Continuing without MCP servers...")

    async def _initialize_enhanced_flight_checker(self):
        """Initialize the enhanced flight checker after MCP connections"""
        try:
            system_print("Initializing Enhanced Generic Flight Checker...")
            self.flight_checker = EnhancedFlightChecker(self)
            success_print("Enhanced flight checker initialized successfully!")

            if self.flight_check_on_startup:
                await self._run_startup_flight_check()

        except Exception as e:
            warning_print(f"Enhanced flight checker initialization failed: {e}")
            warning_print("Continuing without flight checker...")

    async def _initialize_prompt_checker(self):
        """Initialize the prompt-based flight checker"""
        try:
            system_print("Initializing Prompt-Based Flight Checker...")
            self.prompt_checker = PromptBasedFlightChecker(self)
            success_print("Prompt-based flight checker initialized successfully!")

            # Optionally run a quick prompt check on startup
            startup_prompt_check = (
                os.getenv("PROMPT_CHECK_ON_STARTUP", "false").lower() == "true"
            )
            if startup_prompt_check:
                await self._run_startup_prompt_check()

        except Exception as e:
            warning_print(f"Prompt-based flight checker initialization failed: {e}")
            warning_print("Continuing without prompt checker...")

    async def _run_startup_prompt_check(self):
        """Run quick prompt check during startup"""
        system_print("Running startup prompt verification...")

        try:
            results = await self.prompt_checker.run_prompt_flight_check(verbose=False)

            passed = sum(1 for r in results if r.result == TestResult.PASS)
            total = len(results)
            no_tool_calls = sum(
                1 for r in results if r.result == TestResult.NO_TOOL_CALL
            )

            if passed == total:
                success_print(f"All {total} prompts correctly trigger tool calls!")
            elif no_tool_calls > 0:
                warning_print(
                    f"{no_tool_calls}/{total} prompts failed to trigger tool calls"
                )
                warning_print("   This may indicate prompt-to-tool connectivity issues")

                failed_tools = [
                    r.test.tool_name
                    for r in results
                    if r.result == TestResult.NO_TOOL_CALL
                ]
                system_print(f"   Affected tools: {', '.join(failed_tools)}")

                if not self._should_continue_with_prompt_issues():
                    raise Exception(
                        "User chose to abort due to prompt connectivity issues"
                    )
            else:
                warning_print(f"{total - passed}/{total} prompts have issues")

        except Exception as e:
            error_print(f"Startup prompt check failed: {e}")
            if "User chose to abort" in str(e):
                raise
            warning_print("Continuing despite prompt check failure...")

    def _should_continue_with_prompt_issues(self) -> bool:
        """Ask user whether to continue with prompt connectivity issues"""
        try:
            print("\nPrompt connectivity issues may mean your Planner LLM")
            print("won't be able to reliably call tools.")
            response = input("Continue anyway? (y/N): ").strip().lower()
            return response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
            return False

    async def _run_startup_flight_check(self):
        """Run enhanced flight check during startup"""
        system_print("Running enhanced startup flight check...")

        try:
            # Use minimal verbosity for startup
            report = await self.flight_checker.run_flight_check(
                verbosity=VerbosityLevel.MINIMAL
            )

            if report.system_ready:
                success_print(
                    "All systems operational with dynamic dependencies - ready for takeoff!"
                )

                # Show dependency insights briefly
                insights = self.flight_checker.get_dependency_insights()
                if insights["total_dependencies"] > 0:
                    system_print(
                        f"Discovered {insights['total_dependencies']} tool dependencies"
                    )
                    system_print(
                        f"Execution order: {' → '.join(insights['execution_chains'][:3])}..."
                    )
            else:
                warning_print(
                    f"System partially ready - {report.critical_failures} critical issues detected"
                )

                # Show critical issues
                for test_report in report.test_reports:
                    if (
                        test_report.result.value in ["FAIL", "TIMEOUT"]
                        and test_report.test_case.critical
                    ):
                        error_print(
                            f"   Critical: {test_report.test_case.tool_name} - {test_report.error_message}"
                        )

                # Ask user if they want to continue
                if not self._should_continue_with_issues():
                    raise Exception("User chose to abort due to critical failures")

        except Exception as e:
            error_print(f"Startup flight check failed: {e}")
            if "User chose to abort" in str(e):
                raise
            warning_print("Continuing despite flight check failure...")

    def _should_continue_with_issues(self) -> bool:
        """Ask user whether to continue with critical issues"""
        try:
            response = (
                input("\nContinue despite critical issues? (y/N): ").strip().lower()
            )
            return response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
            return False

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

    async def _handle_flight_command(self, query: str):
        """Handle enhanced flight check commands"""
        if not self.flight_checker:
            error_print("Enhanced flight checker not available")
            return

        parts = query.split()
        command = parts[0].lower()

        if command == "/flight-check":
            # Parse verbosity level
            verbosity = VerbosityLevel.NORMAL
            if len(parts) > 1:
                verbosity_map = {
                    "quiet": VerbosityLevel.QUIET,
                    "minimal": VerbosityLevel.MINIMAL,
                    "normal": VerbosityLevel.NORMAL,
                    "verbose": VerbosityLevel.VERBOSE,
                    "debug": VerbosityLevel.DEBUG,
                }
                verbosity_str = parts[1].lower()
                verbosity = verbosity_map.get(verbosity_str, VerbosityLevel.NORMAL)

            # Run enhanced flight check
            await self.flight_checker.run_flight_check(verbosity=verbosity)

        elif command == "/flight-status":
            # Quick status check
            ready = await FlightCheckIntegration.run_quick_check(self)
            if ready:
                success_print("System ready for operations!")
            else:
                warning_print("System has issues - run /flight-check for details")

        elif command == "/flight-deps":
            # Show dependency insights
            insights = self.flight_checker.get_dependency_insights()
            print(f"\nDependency Analysis:")
            print(f"   Total dependencies: {insights['total_dependencies']}")
            print(f"   Dependency types: {dict(insights['dependency_types'])}")
            print(f"   Most connected: {insights['most_connected_tools'][:3]}")
            print(f"   Isolated tools: {insights['isolated_tools']}")
            print(
                f"   Optimized execution order: {' → '.join(insights['execution_chains'])}"
            )

        elif command == "/flight-tools":
            # Show discovered tools
            print(f"\nDiscovered Tools:")
            for tool in self.flight_checker.discovered_tools:
                print(f"   - {tool['name']}: {tool['description'][:50]}...")

        elif command == "/flight-export":
            # Export current report
            if hasattr(self.flight_checker, "_last_report"):
                filename = self.flight_checker.export_report(
                    self.flight_checker._last_report
                )
                success_print(f"Report exported to: {filename}")
            else:
                warning_print(
                    "No recent flight check report to export. Run /flight-check first."
                )

    async def _handle_prompt_command(self, query: str):
        """Handle prompt-based flight check commands"""
        if not self.prompt_checker:
            error_print("Prompt-based flight checker not available")
            return

        parts = query.split()
        command = parts[0].lower()

        if command == "/prompt-check":
            # Run prompt-based flight check
            system_print("Running prompt-based flight check...")
            results = await self.prompt_checker.run_prompt_flight_check(verbose=True)

            # Show summary
            passed = sum(1 for r in results if r.result == TestResult.PASS)
            no_tool_calls = sum(
                1 for r in results if r.result == TestResult.NO_TOOL_CALL
            )

            if passed == len(results):
                success_print("All prompts correctly trigger their intended tools!")
            elif no_tool_calls > 0:
                warning_print(f"{no_tool_calls} prompts failed to trigger tool calls")
                print("Consider running /prompt-config to adjust prompts")

        elif command == "/prompt-config":
            # Show configuration file location and basic info
            print(f"\nPrompt test configuration: {self.prompt_checker.config_path}")
            print(f"Edit this file to customize your prompt tests.")
            print(f"Current tests configured: {len(self.prompt_checker.test_config)}")

            for tool_name, test in self.prompt_checker.test_config.items():
                status = "enabled" if test.enabled else "disabled"
                print(f"  - {tool_name}: {test.description} ({status})")

            print(f"\nAfter editing, restart the chatbot to reload configuration.")

        elif command == "/prompt-add":
            # Interactive prompt test addition
            await self._add_custom_prompt_test()

        elif command == "/prompt-results":
            # Show latest results
            if os.path.exists(self.prompt_checker.results_path):
                with open(self.prompt_checker.results_path, "r") as f:
                    data = json.load(f)

                summary = data.get("summary", {})
                print(f"\nLatest prompt test results:")
                print(f"  Total tests: {summary.get('total', 0)}")
                print(f"  Passed: {summary.get('passed', 0)}")
                print(f"  Failed: {summary.get('failed', 0)}")
                print(f"  No tool call: {summary.get('no_tool_call', 0)}")
                print(f"  Errors: {summary.get('errors', 0)}")
                print(f"  Timestamp: {data.get('timestamp', 'Unknown')}")
            else:
                warning_print("No prompt test results found. Run /prompt-check first.")

        else:
            error_print(f"Unknown prompt command: {command}")
            print("Available prompt commands:")
            print("  /prompt-check - Run prompt-based flight check")
            print("  /prompt-config - Show and edit test configuration")
            print("  /prompt-add - Add a custom prompt test")
            print("  /prompt-results - Show latest test results")

    async def _add_custom_prompt_test(self):
        """Interactive addition of custom prompt test"""
        try:
            print("\nAdding custom prompt test...")
            
            available_tools = [tool["name"] for tool in self.available_tools]
            print(f"Available tools: {', '.join(available_tools)}")
            
            tool_name = input("Tool name: ").strip()
            if tool_name not in available_tools:
                error_print(f"Tool '{tool_name}' not found")
                return
            
            prompt = input("Prompt to test: ").strip()
            if not prompt:
                error_print("Prompt cannot be empty")
                return
            
            print("Expected arguments (JSON format):")
            print("Example: {\"path\": \"server_config.json\"}")
            args_input = input("Expected args: ").strip()
            
            try:
                expected_args = json.loads(args_input) if args_input else {}
            except json.JSONDecodeError:
                error_print("Invalid JSON format for arguments")
                return
            
            description = input("Description (optional): ").strip()
            if not description:
                description = f"Custom test for {tool_name}"
            
            self.prompt_checker.add_custom_test(tool_name, prompt, expected_args, description)
            success_print(f"Added custom test for {tool_name}")
            
            run_now = input("Run this test now? (y/N): ").strip().lower()
            if run_now == 'y':
                custom_tests = [name for name in self.prompt_checker.test_config.keys() if name.startswith("custom_")]
                if custom_tests:
                    latest_test = self.prompt_checker.test_config[custom_tests[-1]]
                    print(f"\nTesting: {latest_test.prompt}")
                    
                    result = await self.prompt_checker._run_prompt_test(latest_test, verbose=True)
                    self.prompt_checker._print_test_result(result)
        
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled")
        except Exception as e:
            error_print(f"Failed to add custom test: {e}")

    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")
        print(
            f"Available prompts: {[prompt['name'] for prompt in self.available_prompts]}"
        )

        # Show enhanced flight check status
        if self.flight_checker:
            success_print(
                "Enhanced Flight Checker: Available with Dynamic Dependency Analysis"
            )
            insights = self.flight_checker.get_dependency_insights()
            if insights["total_dependencies"] > 0:
                print(f"  Dependencies: {insights['total_dependencies']} discovered")
        else:
            warning_print("Enhanced Flight Checker: Disabled")

        # Show prompt-based flight check status
        if self.prompt_checker:
            success_print("Prompt-Based Flight Checker: Available")
            print(
                f"  Tests: {len(self.prompt_checker.test_config)} prompt-to-tool tests configured"
            )
        else:
            warning_print("Prompt-Based Flight Checker: Disabled")

        print("\nCommands:")
        print("Type your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")

        if self.flight_checker:
            print("\nEnhanced Flight Check Commands:")
            print(
                "Use /flight-check [verbosity] to run diagnostics with dynamic dependency analysis"
            )
            print("Use /flight-status to see system health")
            print("Use /flight-deps to see dependency graph and execution order")
            print("Use /flight-tools to see all discovered tools")
            print("Use /flight-export to export latest report")

        if self.prompt_checker:
            print("\nPrompt-Based Flight Check Commands:")
            print("Use /prompt-check to verify prompts trigger correct tool calls")
            print("Use /prompt-config to edit prompt test configuration")
            print("Use /prompt-add to add a custom prompt test")
            print("Use /prompt-results to see latest prompt test results")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue

                if query.lower() == "quit":
                    break

                # Handle enhanced flight check commands
                if query.lower().startswith("/flight-"):
                    await self._handle_flight_command(query)
                    continue

                # Handle prompt-based flight check commands
                if query.lower().startswith("/prompt-"):
                    await self._handle_prompt_command(query)
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


# Enhanced standalone scripts for your existing system
async def standalone_enhanced_flight_check():
    """Run standalone enhanced flight check"""
    system_print("Running Standalone Enhanced Flight Check...")

    chatbot = MCP_ChatBot()
    await chatbot.connect_to_servers()

    # Run comprehensive enhanced flight check
    report = await FlightCheckIntegration.run_comprehensive_check(
        chatbot, export_report=True
    )

    # Print summary
    if report.system_ready:
        success_print(
            "🚀 System ready for operations with dynamic dependency analysis!"
        )
    else:
        error_print(f"❌ {report.critical_failures} critical issues found")

        # Show details
        for test_report in report.test_reports:
            if (
                test_report.result.value in ["FAIL", "TIMEOUT"]
                and test_report.test_case.critical
            ):
                print(
                    f"   - {test_report.test_case.tool_name}: {test_report.error_message}"
                )

    await chatbot.cleanup()
    return report.system_ready


async def enhanced_health_monitor():
    """Continuous enhanced health monitoring"""
    from datetime import datetime

    system_print("Starting Enhanced Health Monitor with Dynamic Dependency Analysis...")

    while True:
        try:
            chatbot = MCP_ChatBot()
            await chatbot.connect_to_servers()

            # Quick enhanced health check
            ready = await FlightCheckIntegration.run_quick_check(chatbot)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if ready:
                success_print(f"[{timestamp}] Enhanced health check passed")
            else:
                warning_print(f"[{timestamp}] Enhanced health check failed")

                # Run detailed check for failures
                checker = EnhancedFlightChecker(chatbot)
                report = await checker.run_flight_check(
                    verbosity=VerbosityLevel.MINIMAL
                )

                error_print(f"Critical failures: {report.critical_failures}")
                error_print(f"Dependency failures: {report.dependency_failures}")
                error_print(f"Total failures: {report.failed}")

                # Show dependency insights on failure
                insights = checker.get_dependency_insights()
                system_print(f"Dependencies analyzed: {insights['total_dependencies']}")

            await chatbot.cleanup()

            # Wait before next check (5 minutes)
            await asyncio.sleep(300)

        except KeyboardInterrupt:
            system_print("Enhanced health monitor stopped")
            break
        except Exception as e:
            error_print(f"Enhanced health monitor error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry


def create_enhanced_config():
    """Create enhanced flight check configuration"""
    import json

    enhanced_config = {
        "flight_checker": {
            "enabled": True,
            "run_on_startup": True,
            "default_verbosity": "normal",
            "timeout_multiplier": 1.0,
            "max_optimization_attempts": 3,
            "auto_export_reports": True,
        },
        "dspy_config": {
            "optimization_enabled": True,
            "model": "openai/o3-mini-birthright",
            "max_tokens": 20000,
            "temperature": 0.7,
            "fallback_to_rules": True,
        },
        "dependency_analysis": {
            "enabled": True,
            "confidence_threshold": 0.5,
            "max_dependency_depth": 3,
            "auto_discovery": True,
        },
        "test_generation": {
            "auto_generate": True,
            "use_dspy": True,
            "pattern_matching_fallback": True,
            "min_test_coverage": 0.8,
        },
        "learning": {
            "save_successful_tests": True,
            "learned_tests_file": "learned_flight_tests.json",
            "max_learned_tests_per_tool": 10,
            "adaptive_criteria": True,
        },
    }

    config_file = "enhanced_flight_config.json"
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump(enhanced_config, f, indent=2)
        success_print(f"Created enhanced flight check configuration: {config_file}")

    return config_file


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "flight-check":
            # Run standalone enhanced flight check
            asyncio.run(standalone_enhanced_flight_check())
        elif command == "health-monitor":
            # Run enhanced health monitor
            asyncio.run(enhanced_health_monitor())
        elif command == "create-config":
            # Create enhanced configuration file
            create_enhanced_config()
        elif command == "test-dspy":
            # Test DSPy connection
            from dspy_optimizer import DSPyFlightOptimizer

            optimizer = DSPyFlightOptimizer()
            if optimizer.test_dspy_connection():
                success_print("DSPy connection test passed!")
            else:
                error_print("DSPy connection test failed!")
        else:
            print("Enhanced MCP Chatbot Usage:")
            print("  python mcp_chatbot.py                    # Run enhanced chatbot")
            print(
                "  python mcp_chatbot.py flight-check       # Run standalone enhanced flight check"
            )
            print(
                "  python mcp_chatbot.py health-monitor     # Run continuous enhanced health monitor"
            )
            print(
                "  python mcp_chatbot.py create-config      # Create enhanced configuration"
            )
            print("  python mcp_chatbot.py test-dspy          # Test DSPy connection")
    else:
        # Run enhanced chatbot
        asyncio.run(main())
