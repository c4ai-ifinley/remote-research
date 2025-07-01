"""
Generic MCP Flight Checker Framework
Zero-configuration testing for any MCP server with auto-discovery and DSPy optimization
"""

import os
import json
import asyncio
import time
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

# Import dependencies with fallbacks
try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

try:
    import colorama
    from colorama import Fore, Back, Style

    colorama.init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


# =============================================================================
# Core Types and Enums
# =============================================================================


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    TIMEOUT = "TIMEOUT"


class VerbosityLevel(Enum):
    QUIET = 0
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3
    DEBUG = 4


@dataclass
class TestCase:
    """Generic test case for any MCP tool"""

    tool_name: str
    test_name: str
    description: str
    prompt: str
    expected_indicators: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    critical: bool = True
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)


@dataclass
class TestReport:
    """Results from running a single test"""

    test_case: TestCase
    result: TestResult
    execution_time: float
    response: Optional[str] = None
    error_message: Optional[str] = None
    validation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlightReport:
    """Complete flight check results"""

    total_tests: int
    passed: int
    failed: int
    skipped: int
    timeout: int
    critical_failures: int
    execution_time: float
    test_reports: List[TestReport] = field(default_factory=list)
    system_ready: bool = False


# =============================================================================
# Cross-Platform Color Output
# =============================================================================


class Colors:
    """Cross-platform color support"""

    def __init__(self):
        if COLORS_AVAILABLE:
            self.DEBUG = Fore.GREEN + Style.BRIGHT
            self.INFO = Fore.CYAN + Style.BRIGHT
            self.SUCCESS = Fore.GREEN + Style.NORMAL
            self.WARNING = Fore.YELLOW + Style.NORMAL
            self.ERROR = Fore.RED + Style.BRIGHT
            self.RESET = Style.RESET_ALL
        else:
            # Fallback to no colors
            self.DEBUG = self.INFO = self.SUCCESS = self.WARNING = self.ERROR = (
                self.RESET
            ) = ""

    def print(self, text: str, color: str = "", end: str = "\n"):
        """Print with color support"""
        print(f"{color}{text}{self.RESET}", end=end)


colors = Colors()


def debug_print(text: str):
    colors.print(f"[DEBUG] {text}", colors.DEBUG)


def info_print(text: str):
    colors.print(f"[INFO] {text}", colors.INFO)


def success_print(text: str):
    colors.print(f"[SUCCESS] {text}", colors.SUCCESS)


def warning_print(text: str):
    colors.print(f"[WARNING] {text}", colors.WARNING)


def error_print(text: str):
    colors.print(f"[ERROR] {text}", colors.ERROR)


# =============================================================================
# DSPy Optimization Engine
# =============================================================================


class OptimizationContext:
    """Context for prompt optimization"""

    def __init__(
        self,
        tool_name: str,
        original_prompt: str,
        failure_reason: str,
        tool_schema: Dict,
        previous_attempts: List[str] = None,
    ):
        self.tool_name = tool_name
        self.original_prompt = original_prompt
        self.failure_reason = failure_reason
        self.tool_schema = tool_schema
        self.previous_attempts = previous_attempts or []


class PromptOptimizer:
    """DSPy-powered prompt optimization"""

    def __init__(self):
        self.dspy_available = DSPY_AVAILABLE
        self.optimizer = None
        self._setup_dspy()

    def _setup_dspy(self):
        """Initialize DSPy if available"""
        if not self.dspy_available:
            warning_print("DSPy not available - limited optimization capabilities")
            return

        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("BASE_URL")

            if not openai_key:
                warning_print("OPENAI_API_KEY not found - DSPy disabled")
                self.dspy_available = False
                return

            config = {
                "model": "openai/o3-mini-birthright",
                "api_key": openai_key,
                "max_tokens": 20000,
                "temperature": 1.0,
            }

            if base_url:
                config["base_url"] = base_url

            lm = dspy.LM(**config)
            dspy.configure(lm=lm)

            # Define optimization signature
            class OptimizePromptSignature(dspy.Signature):
                tool_name = dspy.InputField(desc="Name of the MCP tool")
                tool_description = dspy.InputField(desc="What the tool does")
                original_prompt = dspy.InputField(desc="The failing prompt")
                failure_info = dspy.InputField(desc="Why it failed and requirements")
                optimized_prompt = dspy.OutputField(
                    desc="Natural, human-like prompt that would work"
                )

            self.optimizer = dspy.ChainOfThought(OptimizePromptSignature)
            success_print("DSPy optimizer initialized")

        except Exception as e:
            error_print(f"DSPy setup failed: {e}")
            self.dspy_available = False

    def optimize_prompt(self, context: OptimizationContext) -> str:
        """Optimize a failing prompt using DSPy only"""
        if not self.dspy_available or not self.optimizer:
            warning_print("DSPy not available - cannot optimize prompts")
            warning_print(
                "Install dspy-ai and set OPENAI_API_KEY for prompt optimization"
            )
            return context.original_prompt

        return self._dspy_optimize(context)

    def _dspy_optimize(self, context: OptimizationContext) -> str:
        """Use DSPy for optimization with enhanced context"""
        try:
            tool_desc = self._generate_tool_description(
                context.tool_name, context.tool_schema
            )

            # Create comprehensive failure context for DSPy
            failure_info = f"""
FAILED PROMPT: "{context.original_prompt}"
FAILURE REASON: {context.failure_reason}

TOOL CONTEXT:
{tool_desc}

REQUIREMENTS FOR NEW PROMPT:
- Must sound like something a human would naturally ask
- Should be specific about what information is wanted
- Must work with the tool's actual parameters
- Avoid generic phrases like "use the tool" or "with appropriate parameters"

EXAMPLES OF GOOD NATURAL PROMPTS:
- "Find papers about machine learning"
- "What's in the config.json file?"
- "Show me the files in this directory"
- "Tell me about paper 2108.07258v1"
- "Get the content from https://example.com"

{self._format_previous_attempts(context.previous_attempts)}
"""

            result = self.optimizer(
                tool_name=context.tool_name,
                tool_description=tool_desc,
                original_prompt=context.original_prompt,
                failure_info=failure_info,
            )

            optimized = result.optimized_prompt.strip()

            # Validate that the optimization is actually better
            if (
                self._is_prompt_natural(optimized)
                and optimized != context.original_prompt
            ):
                return optimized
            else:
                warning_print(f"DSPy optimization didn't improve prompt quality")
                return context.original_prompt

        except Exception as e:
            error_print(f"DSPy optimization failed: {e}")
            return context.original_prompt

    def _format_previous_attempts(self, previous_attempts: List[str]) -> str:
        """Format previous attempts for DSPy context"""
        if not previous_attempts:
            return ""

        attempts_text = "PREVIOUS FAILED ATTEMPTS (must be different):\n"
        for i, attempt in enumerate(previous_attempts[-3:], 1):
            attempts_text += f"{i}. '{attempt}'\n"
        attempts_text += "\nThe new prompt must be significantly different from all previous attempts."
        return attempts_text

    def _is_prompt_natural(self, prompt: str) -> bool:
        """Validate that a prompt sounds natural and human-like"""
        # Check for bad patterns that indicate generic/artificial prompts
        bad_patterns = [
            "use the",
            "with appropriate parameters",
            "execute the function",
            "return results",
            "tool with",
            "appropriate parameters",
        ]

        prompt_lower = prompt.lower()

        # Reject if it contains bad patterns
        if any(pattern in prompt_lower for pattern in bad_patterns):
            return False

        # Check for good natural patterns
        good_patterns = [
            "find",
            "search",
            "show me",
            "what's",
            "tell me",
            "get",
            "can you",
            "i want",
            "i need",
            "look up",
        ]

        # Should have at least one natural pattern
        if not any(pattern in prompt_lower for pattern in good_patterns):
            return False

        # Should be reasonably short (humans don't write super long prompts)
        if len(prompt.split()) > 15:
            return False

        return True

    def _generate_tool_description(self, tool_name: str, tool_schema: Dict) -> str:
        """Generate description from tool schema"""
        desc = tool_schema.get("description", f"Tool: {tool_name}")

        if "inputSchema" in tool_schema:
            props = tool_schema["inputSchema"].get("properties", {})
            if props:
                desc += f"\nParameters: {list(props.keys())}"

        return desc


# =============================================================================
# Auto-Discovery Engine
# =============================================================================


class ToolDiscovery:
    """Automatically discovers MCP tools and generates test cases"""

    def __init__(self, config_file: str = "mcp_flight_config.json"):
        self.config_file = config_file
        self.test_templates = self._load_natural_templates()
        self.learned_tests = self._load_learned_tests()

        # DEBUG: Make sure we're not accidentally using the wrong file
        if self.config_file == "server_config.json":
            error_print(
                "CRITICAL ERROR: Flight checker trying to use server_config.json!"
            )
            error_print("This would corrupt your server configuration!")
            self.config_file = "mcp_flight_config.json"
            warning_print(f"Changed config file to {self.config_file}")

        debug_print(f"ToolDiscovery initialized with config file: {self.config_file}")

    def _load_natural_templates(self) -> Dict[str, Dict]:
        """Load natural language test templates that humans would actually use"""
        return {
            "search": {
                "prompts": [
                    "Find papers about {topic}",
                    "Search for research on {topic}",
                    "Look up academic papers related to {topic}",
                    "I need papers about {topic}",
                    "Can you find studies on {topic}?",
                ],
                "indicators": ["found", "papers", "results", "search"],
                "criteria": {"min_response_length": 10},
            },
            "read": {
                "prompts": [
                    "What's in the {filename} file?",
                    "Show me the contents of {filename}",
                    "Can you read {filename}?",
                    "Open {filename} and tell me what's inside",
                    "I want to see what's in {filename}",
                ],
                "indicators": ["content", "file", "contains"],
                "criteria": {"min_response_length": 5},
            },
            "fetch": {
                "prompts": [
                    "Get the content from {url}",
                    "What's at {url}?",
                    "Fetch {url} for me",
                    "Can you load {url}?",
                    "I want to see what's on {url}",
                ],
                "indicators": ["content", "data", "response"],
                "criteria": {"min_response_length": 10},
            },
            "list": {
                "prompts": [
                    "What files are in {path}?",
                    "Show me what's in the {path} directory",
                    "List the contents of {path}",
                    "What's in this folder?",
                    "Can you show me the files here?",
                ],
                "indicators": ["files", "directory", "contents"],
                "criteria": {"min_response_length": 5},
            },
            "extract": {
                "prompts": [
                    "Tell me about paper {item_id}",
                    "What do you know about {item_id}?",
                    "Get me information on {item_id}",
                    "Show me details for {item_id}",
                    "I want to know more about {item_id}",
                ],
                "indicators": ["information", "details", "about"],
                "criteria": {"acceptable_not_found": True},
            },
        }

    def _load_learned_tests(self) -> Dict[str, List[Dict]]:
        """Load previously successful test cases"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    return config.get("learned_tests", {})
        except Exception as e:
            debug_print(f"Could not load learned tests: {e}")
        return {}

    def save_learned_tests(self):
        """Save successful test cases for future use - simple version"""
        try:
            config = {
                "learned_tests": self.learned_tests,
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)

            debug_print(f"Successfully saved learned tests to {self.config_file}")

        except Exception as e:

            error_print(f"Could not save learned tests: {e}")

    def add_successful_test(self, test_case: TestCase):
        """Add a successful test case to learned tests"""
        tool_name = test_case.tool_name
        if tool_name not in self.learned_tests:
            self.learned_tests[tool_name] = []

        test_data = {
            "test_name": test_case.test_name,
            "prompt": test_case.prompt,
            "expected_indicators": test_case.expected_indicators,
            "success_criteria": test_case.success_criteria,
            "last_success": datetime.now().isoformat(),
            "success_count": 1,
        }

        # Check if similar test already exists
        for existing_test in self.learned_tests[tool_name]:
            if existing_test["prompt"] == test_case.prompt:
                existing_test["success_count"] += 1
                existing_test["last_success"] = datetime.now().isoformat()
                return

        # Add new successful test
        self.learned_tests[tool_name].append(test_data)
        debug_print(f"Learned successful test for {tool_name}: {test_case.prompt}")
        self.save_learned_tests()

    def discover_and_generate_tests(
        self, available_tools: List[Dict]
    ) -> Dict[str, List[TestCase]]:
        """Auto-discover tools and generate appropriate test cases"""
        test_cases = {}

        for tool in available_tools:
            tool_name = tool["name"]
            tool_schema = tool

            info_print(f"Discovering tests for tool: {tool_name}")

            # First, try to use learned successful tests
            learned_cases = self._get_learned_test_cases(tool_name, tool_schema)

            # If we have learned tests, use them; otherwise generate new ones
            if learned_cases:
                test_cases[tool_name] = learned_cases
                success_print(
                    f"Using {len(learned_cases)} learned test cases for {tool_name}"
                )
            else:
                # Generate natural language test cases
                cases = self._generate_natural_test_cases(tool_name, tool_schema)
                test_cases[tool_name] = cases
                if cases:
                    debug_print(
                        f"Generated {len(cases)} natural test cases for {tool_name}"
                    )
                else:
                    debug_print(
                        f"No natural test cases generated for {tool_name} - will create discovery test"
                    )

        return test_cases

    def _get_learned_test_cases(
        self, tool_name: str, tool_schema: Dict
    ) -> List[TestCase]:
        """Convert learned test data back to TestCase objects"""
        if tool_name not in self.learned_tests:
            return []

        cases = []
        for learned_test in self.learned_tests[tool_name]:
            case = TestCase(
                tool_name=tool_name,
                test_name=learned_test["test_name"],
                description=f"Learned test case (success count: {learned_test['success_count']})",
                prompt=learned_test["prompt"],
                expected_indicators=learned_test.get("expected_indicators", []),
                timeout_seconds=30.0,
                critical=True,
                success_criteria=learned_test.get("success_criteria", {}),
            )
            cases.append(case)

        return cases

    def _classify_tool(self, tool_name: str, tool_schema: Dict) -> str:
        """Classify tool type based on name and schema"""
        name_lower = tool_name.lower()
        description = tool_schema.get("description", "").lower()

        # Classification patterns
        patterns = {
            "search": ["search", "find", "query", "lookup"],
            "read": ["read", "open", "get", "load"],
            "fetch": ["fetch", "download", "retrieve", "request"],
            "list": ["list", "ls", "dir", "directory", "contents"],
            "extract": ["extract", "parse", "analyze", "info"],
        }

        for tool_type, keywords in patterns.items():
            if any(
                keyword in name_lower or keyword in description for keyword in keywords
            ):
                return tool_type

        # If we can't classify it, return None - we'll skip generating generic prompts
        return None

    def _generate_natural_test_cases(
        self, tool_name: str, tool_schema: Dict
    ) -> List[TestCase]:
        """Generate natural language test cases for a specific tool"""
        tool_type = self._classify_tool(tool_name, tool_schema)

        # If we can't classify the tool, don't generate generic test cases
        # Let DSPy handle it when it learns from usage
        if tool_type is None:
            warning_print(
                f"Could not classify tool {tool_name} - will learn from usage"
            )
            return []

        template = self.test_templates.get(tool_type)
        if not template:
            return []

        cases = []

        # Get realistic parameters for this tool
        params = self._extract_realistic_parameters(tool_schema)

        # Generate 1-2 natural test cases using different prompt styles
        for i, prompt_template in enumerate(template["prompts"][:2]):
            try:
                prompt = prompt_template.format(**params)

                cases.append(
                    TestCase(
                        tool_name=tool_name,
                        test_name=f"{tool_name}_natural_test_{i+1}",
                        description=f"Natural language test for {tool_name}",
                        prompt=prompt,
                        expected_indicators=template["indicators"],
                        timeout_seconds=30.0,
                        critical=(
                            i == 0
                        ),  # First test is critical, others are nice-to-have
                        success_criteria=template["criteria"],
                    )
                )
            except KeyError as e:
                # If template formatting fails, skip this template
                debug_print(f"Skipping template for {tool_name}: missing parameter {e}")
                continue

        return cases

    def _extract_realistic_parameters(self, tool_schema: Dict) -> Dict[str, Any]:
        """Extract realistic parameter values that humans would actually use"""
        params = {}

        input_schema = tool_schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})

        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "").lower()

            # Generate realistic values based on parameter name and description
            if "topic" in param_name.lower() or "query" in param_name.lower():
                # Use realistic research topics that humans search for
                topics = [
                    "machine learning",
                    "climate change",
                    "quantum computing",
                    "artificial intelligence",
                ]
                params[param_name] = topics[0]  # Default to first one

            elif "url" in param_name.lower():
                # Use a real, reliable test URL
                params[param_name] = "https://httpbin.org/json"

            elif "path" in param_name.lower() or "file" in param_name.lower():
                if "dir" in param_name.lower() or "directory" in param_name.lower():
                    params[param_name] = "."  # Current directory
                else:
                    # CRITICAL: Use a safe test file, NOT server_config.json
                    # Create a safe test file to avoid corrupting important configs
                    safe_test_files = ["README.md", "package.json", "pyproject.toml"]
                    # First try to find an existing safe file
                    for test_file in safe_test_files:
                        if os.path.exists(test_file):
                            params[param_name] = test_file
                            break
                    else:
                        # If no safe files exist, create a temporary test file
                        test_file = "flight_check_test.txt"
                        if not os.path.exists(test_file):
                            try:
                                with open(test_file, "w") as f:
                                    f.write(
                                        "This is a test file for MCP flight checker.\nSafe to read and delete.\n"
                                    )
                                debug_print(f"Created safe test file: {test_file}")
                            except Exception as e:
                                debug_print(f"Could not create test file: {e}")
                                test_file = "."  # Fallback to directory
                        params[param_name] = test_file

            elif "id" in param_name.lower() and "paper" in param_desc:
                # Use realistic arXiv paper IDs
                params[param_name] = "2108.07258v1"  # A real paper ID

            elif "id" in param_name.lower():
                params[param_name] = "example_123"

            elif param_type == "integer" and (
                "max" in param_name.lower() or "limit" in param_name.lower()
            ):
                params[param_name] = 3  # Reasonable small number for testing

            elif param_type == "integer":
                params[param_name] = 1

            elif param_type == "boolean":
                params[param_name] = True

            else:
                # For unknown parameters, use a descriptive placeholder
                params[param_name] = f"test_{param_name}"

        # Add some common template variables that might be used
        # CRITICAL: Use safe files only
        params.update(
            {
                "filename": "flight_check_test.txt",  # Safe test file
                "item_id": "2108.07258v1",
                "url": "https://httpbin.org/json",
            }
        )

        return params


# =============================================================================
# Main Flight Checker
# =============================================================================


class MCPFlightChecker:
    """Generic MCP flight checker with auto-discovery and optimization"""

    def __init__(
        self,
        chatbot_instance,
        verbosity: VerbosityLevel = VerbosityLevel.MINIMAL,
        config_file: str = "mcp_flight_config.json",
    ):
        self.chatbot = chatbot_instance
        self.verbosity = verbosity
        self.discovery = ToolDiscovery(config_file)
        self.optimizer = PromptOptimizer()
        self.test_cases: Dict[str, List[TestCase]] = {}

        # Auto-discover tools on initialization
        self._auto_discover()

    def _auto_discover(self):
        """Automatically discover available tools and generate test cases"""
        if not self.chatbot.available_tools:
            warning_print("No tools available for testing")
            return

        info_print(f"Auto-discovering {len(self.chatbot.available_tools)} tools...")

        self.test_cases = self.discovery.discover_and_generate_tests(
            self.chatbot.available_tools
        )

        # Count total tests, including learned ones
        total_tests = sum(len(cases) for cases in self.test_cases.values())
        tools_with_tests = len(
            [tool for tool, cases in self.test_cases.items() if cases]
        )

        if total_tests > 0:
            success_print(
                f"Ready with {total_tests} test cases for {tools_with_tests} tools"
            )
        else:
            warning_print("No test cases generated - will learn from usage patterns")

    async def run_flight_check(self, optimize_failures: bool = True) -> FlightReport:
        """Execute comprehensive flight check"""
        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            colors.print("=" * 60, colors.INFO)
            colors.print("MCP FLIGHT CHECK INITIATED", colors.INFO)
            colors.print("=" * 60, colors.INFO)

        start_time = time.time()
        all_reports = []
        failed_tests = []

        # If no test cases exist, create minimal discovery tests using DSPy
        if not any(self.test_cases.values()):
            warning_print(
                "No existing test cases - creating minimal tests for DSPy learning"
            )
            await self._create_discovery_tests()

        # Run all test cases
        for tool_name, test_cases in self.test_cases.items():
            if not test_cases:
                continue

            if self.verbosity.value >= VerbosityLevel.MINIMAL.value:
                info_print(f"Testing {tool_name}...")

            for test_case in test_cases:
                if self.verbosity.value >= VerbosityLevel.VERBOSE.value:
                    debug_print(f"  Running: {test_case.test_name}")

                report = await self._run_single_test(test_case)
                all_reports.append(report)

                if report.result == TestResult.FAIL and test_case.critical:
                    failed_tests.append((test_case, report))

                self._print_test_result(report)

        # Optimize failed tests if enabled
        if optimize_failures and failed_tests:
            await self._optimize_failed_tests(failed_tests)

        # Save successful tests for future use
        self._save_successful_tests(all_reports)

        # Generate final report
        total_time = time.time() - start_time
        report = self._generate_report(all_reports, total_time)

        self._print_summary(report)
        return report

    async def _create_discovery_tests(self):
        """Create minimal test cases for tools that we couldn't classify"""
        for tool in self.chatbot.available_tools:
            tool_name = tool["name"]

            if tool_name in self.test_cases and self.test_cases[tool_name]:
                continue  # Already has test cases

            # Create a very simple discovery test that DSPy can optimize
            discovery_test = TestCase(
                tool_name=tool_name,
                test_name=f"{tool_name}_discovery",
                description=f"Discovery test for {tool_name} - will be optimized by DSPy",
                prompt=f"Help me use {tool_name}",  # Very simple, will trigger optimization
                expected_indicators=["result", "output", "response", "content", "data"],
                timeout_seconds=30.0,
                critical=True,
                success_criteria={"min_response_length": 3},
            )

            self.test_cases[tool_name] = [discovery_test]
            debug_print(f"Created discovery test for unclassified tool: {tool_name}")

    async def _run_single_test(self, test_case: TestCase) -> TestReport:
        """Execute a single test case"""
        start_time = time.time()

        # Check if tool exists
        if test_case.tool_name not in [t["name"] for t in self.chatbot.available_tools]:
            return TestReport(
                test_case=test_case,
                result=TestResult.SKIP,
                execution_time=0,
                error_message=f"Tool '{test_case.tool_name}' not available",
            )

        try:
            # Execute the prompt through the chatbot
            response = await self._execute_prompt(test_case)

            # Validate response
            validation = self._validate_response(test_case, response)

            return TestReport(
                test_case=test_case,
                result=TestResult.PASS if validation["valid"] else TestResult.FAIL,
                execution_time=time.time() - start_time,
                response=response[:200] + "..." if len(response) > 200 else response,
                error_message=None if validation["valid"] else validation["reason"],
                validation_details=validation,
            )

        except asyncio.TimeoutError:
            return TestReport(
                test_case=test_case,
                result=TestResult.TIMEOUT,
                execution_time=time.time() - start_time,
                error_message=f"Timeout after {test_case.timeout_seconds}s",
            )
        except Exception as e:
            return TestReport(
                test_case=test_case,
                result=TestResult.FAIL,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def _execute_prompt(self, test_case: TestCase) -> str:
        """Execute a test prompt through the chatbot system with function call verification"""
        # Track whether the actual tool was called
        tool_called = False
        tool_response = None

        # Get the session for direct tool calling
        session = self.chatbot.sessions.get(test_case.tool_name)
        if not session:
            raise Exception("Tool session not found")

        try:
            # Extract tool arguments from prompt
            tool_args = self._extract_args_from_prompt(test_case)

            # Call the tool directly to verify it actually works
            result = await asyncio.wait_for(
                session.call_tool(test_case.tool_name, arguments=tool_args),
                timeout=test_case.timeout_seconds,
            )

            tool_called = True
            tool_response = result.content[0].text if result.content else str(result)

            # Store the actual function call result for validation
            test_case._actual_tool_response = tool_response
            test_case._tool_called_successfully = True

            return tool_response

        except Exception as e:
            test_case._tool_called_successfully = False
            test_case._tool_error = str(e)
            raise e

    def _extract_args_from_prompt(self, test_case: TestCase) -> Dict[str, Any]:
        """Extract tool arguments from prompt (basic implementation)"""
        # This would be enhanced with LLM-based parameter extraction
        tool_schema = next(
            (
                t
                for t in self.chatbot.available_tools
                if t["name"] == test_case.tool_name
            ),
            {},
        )

        return self.discovery._extract_realistic_parameters(tool_schema)

    def _validate_response(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Validate test response against success criteria with function call verification"""

    def _validate_response(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Validate test response against success criteria with function call verification"""
        validation = {
            "valid": False,
            "reason": "",
            "checks_performed": [],
            "function_called": False,
            "response_valid": False,
        }

        # CRITICAL: First check if the actual function was called successfully
        if not getattr(test_case, "_tool_called_successfully", False):
            validation["reason"] = (
                f"Tool function was not called successfully: {getattr(test_case, '_tool_error', 'Unknown error')}"
            )
            return validation

        validation["function_called"] = True
        validation["checks_performed"].append("Function call verified")

        # Now validate the actual response from the function call
        criteria = test_case.success_criteria
        response_lower = response.lower()

        # Check minimum length
        if "min_response_length" in criteria:
            min_len = criteria["min_response_length"]
            if len(response) >= min_len:
                validation["checks_performed"].append(
                    f"Length OK ({len(response)} >= {min_len})"
                )
            else:
                validation["reason"] = (
                    f"Response too short ({len(response)} < {min_len})"
                )
                return validation

        # Check for expected indicators
        found_indicators = [
            ind
            for ind in test_case.expected_indicators
            if ind.lower() in response_lower
        ]

        # Tool-specific validation
        tool_validation = self._validate_tool_specific_response(test_case, response)
        if not tool_validation["valid"]:
            validation["reason"] = tool_validation["reason"]
            return validation

        validation["checks_performed"].extend(tool_validation["checks_performed"])

        if found_indicators:
            validation["checks_performed"].append(
                f"Found indicators: {found_indicators}"
            )
            validation["response_valid"] = True
        elif criteria.get("acceptable_not_found", False):
            not_found_phrases = [
                "no information",
                "not found",
                "no results",
                "no saved information",
            ]
            if any(phrase in response_lower for phrase in not_found_phrases):
                validation["response_valid"] = True
                validation["checks_performed"].append("Acceptable 'not found' response")
        else:
            validation["reason"] = (
                f"Expected indicators not found: {test_case.expected_indicators}"
            )
            return validation

        # If we get here, both function call and response validation passed
        validation["valid"] = True
        return validation

    def _validate_tool_specific_response(
        self, test_case: TestCase, response: str
    ) -> Dict[str, Any]:
        """Tool-specific response validation"""
        validation = {"valid": True, "reason": "", "checks_performed": []}

        tool_name = test_case.tool_name
        response_lower = response.lower()

        if "search" in tool_name.lower():
            # For search tools, expect some kind of results
            if len(response.strip()) < 5:
                validation["valid"] = False
                validation["reason"] = "Search returned empty or very short response"
            elif "error" in response_lower or "failed" in response_lower:
                validation["valid"] = False
                validation["reason"] = "Search returned error message"
            else:
                validation["checks_performed"].append("Search response format valid")

        elif "read" in tool_name.lower() or "file" in tool_name.lower():
            # For file reading, expect content or clear error message
            if len(response.strip()) == 0:
                validation["valid"] = False
                validation["reason"] = "File read returned empty response"
            else:
                validation["checks_performed"].append("File read response has content")

        elif "list" in tool_name.lower() or "directory" in tool_name.lower():
            # For directory listing, expect some structure
            if not any(
                indicator in response_lower
                for indicator in ["file", "directory", "folder", ".", "/"]
            ):
                validation["valid"] = False
                validation["reason"] = (
                    "Directory listing doesn't contain expected file indicators"
                )
            else:
                validation["checks_performed"].append("Directory listing format valid")

        elif "fetch" in tool_name.lower():
            # For fetch operations, expect content or clear error
            if len(response.strip()) < 10:
                validation["valid"] = False
                validation["reason"] = "Fetch returned very short response"
            else:
                validation["checks_performed"].append(
                    "Fetch response has adequate content"
                )

        elif "extract" in tool_name.lower():
            # For extraction, accept "not found" as valid
            if (
                "no saved information" in response_lower
                or "not found" in response_lower
            ):
                validation["checks_performed"].append(
                    "Extraction returned valid 'not found' response"
                )
            elif len(response.strip()) < 5:
                validation["valid"] = False
                validation["reason"] = "Extraction returned empty response"
            else:
                validation["checks_performed"].append("Extraction returned content")

        return validation

    async def _optimize_failed_tests(self, failed_tests: List[tuple]):
        """Optimize prompts for failed tests and save successful ones"""
        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            warning_print(f"Optimizing {len(failed_tests)} failed tests...")

        for test_case, report in failed_tests:
            tool_schema = next(
                (
                    t
                    for t in self.chatbot.available_tools
                    if t["name"] == test_case.tool_name
                ),
                {},
            )

            context = OptimizationContext(
                tool_name=test_case.tool_name,
                original_prompt=test_case.prompt,
                failure_reason=report.error_message or "Validation failed",
                tool_schema=tool_schema,
                previous_attempts=[
                    h.get("optimized_prompt", "")
                    for h in test_case.optimization_history
                ],
            )

            optimized_prompt = self.optimizer.optimize_prompt(context)

            if optimized_prompt != test_case.prompt:
                if self.verbosity.value >= VerbosityLevel.VERBOSE.value:
                    debug_print(f"  Original: {test_case.prompt}")
                    success_print(f"  Optimized: {optimized_prompt}")

                # Update test case
                test_case.prompt = optimized_prompt
                test_case.optimization_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "original_prompt": context.original_prompt,
                        "optimized_prompt": optimized_prompt,
                        "failure_reason": context.failure_reason,
                    }
                )

    def _save_successful_tests(self, test_reports: List[TestReport]):
        """Save tests that passed for future use"""
        for report in test_reports:
            if report.result == TestResult.PASS and getattr(
                report.test_case, "_tool_called_successfully", False
            ):
                self.discovery.add_successful_test(report.test_case)

    def _print_test_result(self, report: TestReport):
        """Print test result based on verbosity"""
        if self.verbosity == VerbosityLevel.QUIET:
            return

        status_colors = {
            TestResult.PASS: colors.SUCCESS,
            TestResult.FAIL: colors.ERROR,
            TestResult.SKIP: colors.WARNING,
            TestResult.TIMEOUT: colors.WARNING,
        }

        status = report.result.value
        color = status_colors[report.result]

        if self.verbosity == VerbosityLevel.MINIMAL:
            colors.print(f"  {status} ({report.execution_time:.2f}s)", color)
        else:
            colors.print(
                f"    {report.test_case.test_name}: {status} ({report.execution_time:.2f}s)",
                color,
            )

            if (
                report.error_message
                and self.verbosity.value >= VerbosityLevel.VERBOSE.value
            ):
                error_print(f"      {report.error_message}")

    def _generate_report(
        self, test_reports: List[TestReport], total_time: float
    ) -> FlightReport:
        """Generate comprehensive flight report"""
        passed = sum(1 for r in test_reports if r.result == TestResult.PASS)
        failed = sum(1 for r in test_reports if r.result == TestResult.FAIL)
        skipped = sum(1 for r in test_reports if r.result == TestResult.SKIP)
        timeout = sum(1 for r in test_reports if r.result == TestResult.TIMEOUT)

        critical_failures = sum(
            1
            for r in test_reports
            if r.result in [TestResult.FAIL, TestResult.TIMEOUT]
            and r.test_case.critical
        )

        return FlightReport(
            total_tests=len(test_reports),
            passed=passed,
            failed=failed,
            skipped=skipped,
            timeout=timeout,
            critical_failures=critical_failures,
            execution_time=total_time,
            test_reports=test_reports,
            system_ready=critical_failures == 0,
        )

    def _print_summary(self, report: FlightReport):
        """Print flight check summary"""
        colors.print("\n" + "=" * 60, colors.INFO)
        colors.print("FLIGHT CHECK SUMMARY", colors.INFO)
        colors.print("=" * 60, colors.INFO)

        colors.print(f"Total Tests: {report.total_tests}", colors.INFO)
        colors.print(f"Passed: {report.passed}", colors.SUCCESS)
        colors.print(f"Failed: {report.failed}", colors.ERROR)
        colors.print(f"Skipped: {report.skipped}", colors.WARNING)
        colors.print(f"Timeout: {report.timeout}", colors.WARNING)
        colors.print(f"Critical Failures: {report.critical_failures}", colors.ERROR)
        colors.print(f"Execution Time: {report.execution_time:.2f}s", colors.INFO)

        if report.system_ready:
            success_print("\nSYSTEM READY FOR TAKEOFF!")
            success_print("All critical systems operational.")
        else:
            error_print(f"\nSYSTEM NOT READY")
            error_print(f"{report.critical_failures} critical failure(s) detected!")

        colors.print("=" * 60, colors.INFO)


# =============================================================================
# Easy Integration Functions
# =============================================================================


async def quick_flight_check(
    chatbot_instance,
    verbosity: VerbosityLevel = VerbosityLevel.MINIMAL,
    config_file: str = "mcp_flight_config.json",
) -> FlightReport:
    """One-liner flight check for any MCP chatbot"""
    checker = MCPFlightChecker(chatbot_instance, verbosity, config_file)
    return await checker.run_flight_check()


def setup_environment():
    """Setup environment for optimal flight checking"""
    # Check for required environment variables
    required_vars = ["ANTHROPIC_API_KEY"]
    optional_vars = ["OPENAI_API_KEY", "BASE_URL"]

    missing_required = [var for var in required_vars if not os.getenv(var)]
    missing_optional = [var for var in optional_vars if not os.getenv(var)]

    if missing_required:
        error_print(f"Missing required environment variables: {missing_required}")
        return False

    if missing_optional:
        warning_print(
            f"Missing optional variables (limited functionality): {missing_optional}"
        )

    return True


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    # Example usage - drop this into any MCP project

    async def example_usage():
        # Your existing chatbot instance
        # chatbot = MCP_ChatBot()
        # await chatbot.connect_to_servers()

        # Zero-configuration flight check
        # report = await quick_flight_check(chatbot, VerbosityLevel.VERBOSE)

        # Or for more control:
        # checker = MCPFlightChecker(chatbot, VerbosityLevel.NORMAL)
        # report = await checker.run_flight_check(optimize_failures=True)

        print("Flight checker ready to use!")

    asyncio.run(example_usage())
