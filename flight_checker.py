"""
MCP Tool Flight Checker with Auto Test Generation and DSPy Optimization
Works with any MCP tools using only schema information.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union
from enum import Enum
import asyncio
import json
import time
import re
import os
from datetime import datetime
from pathlib import Path
from utils import atomic_write_json, ensure_test_cases_config

# Import the DSPy optimizer
from dspy_optimizer import DSPyOptimizer, OptimizationContext

# Import color utilities
from color_utils import (
    flight_check_print,
    debug_print,
    system_print,
    error_print,
    success_print,
    warning_print,
    dspy_print,
    optimization_print,
    header_print,
    separator_print,
    test_result_print,
    Colors,
    colored_print,
)


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    TIMEOUT = "TIMEOUT"
    CONTEXT_MISSING = "CONTEXT_MISSING"


class VerbosityLevel(Enum):
    QUIET = 0
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3
    DEBUG = 4


@dataclass
class TestCase:
    """Test case that works with any MCP tool"""

    tool_name: str
    test_name: str
    description: str
    prompt: str
    tool_schema: Dict[str, Any]
    generated_arguments: Dict[str, Any] = field(default_factory=dict)
    expected_indicators: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    critical: bool = True
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)
    auto_generated: bool = True
    context_requirements: List[str] = field(default_factory=list)
    test_type: str = "basic"  # 'basic', 'parameter', 'error', 'learned'


@dataclass
class TestReport:
    """Results from running a single test"""

    test_case: TestCase
    result: TestResult
    execution_time: float
    response: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    context_issues: List[str] = field(default_factory=list)


@dataclass
class FlightCheckReport:
    """Complete flight check results"""

    total_tests: int
    passed: int
    failed: int
    skipped: int
    timeout: int
    context_missing: int
    critical_failures: int
    execution_time: float
    test_reports: List[TestReport] = field(default_factory=list)
    system_ready: bool = False
    optimization_suggestions: List[Dict] = field(default_factory=list)


class TestGenerator:
    """Automatically generates test cases from MCP tool schemas"""

    def __init__(self, test_mode="comprehensive"):
        """
        Initialize test generator

        Args:
            test_mode: 'basic' (only basic functionality), 'comprehensive' (all test types)
        """
        self.test_mode = test_mode
        self.generation_strategies = {
            "basic_functionality": self._generate_basic_test,
            "parameter_validation": self._generate_parameter_test,
            "error_handling": self._generate_error_test,
        }

    def generate_test_cases(self, tool_schema: Dict[str, Any]) -> List[TestCase]:
        """Generate test cases for a tool based on its MCP schema"""
        tool_name = tool_schema.get("name", "unknown_tool")
        description = tool_schema.get("description", "No description available")
        input_schema = tool_schema.get("input_schema", {})

        test_cases = []

        if self.test_mode == "basic":
            # Only generate basic functionality test
            basic_test = self._generate_basic_test(tool_name, description, input_schema)
            if basic_test:
                test_cases.append(basic_test)
        else:
            # Generate all test types (comprehensive mode)
            # Generate basic functionality test
            basic_test = self._generate_basic_test(tool_name, description, input_schema)
            if basic_test:
                test_cases.append(basic_test)

            # Generate parameter-specific tests if schema has parameters
            if input_schema.get("properties"):
                param_test = self._generate_parameter_test(
                    tool_name, description, input_schema
                )
                if param_test:
                    test_cases.append(param_test)

            # Generate error handling test
            error_test = self._generate_error_test(tool_name, description, input_schema)
            if error_test:
                test_cases.append(error_test)

        return test_cases

    def _generate_basic_test(
        self, tool_name: str, description: str, input_schema: Dict
    ) -> Optional[TestCase]:
        """Generate a basic functionality test"""

        # Generate arguments from schema
        generated_args, context_reqs = self._generate_arguments_from_schema(
            input_schema
        )

        # Create prompt
        prompt = f"Test the {tool_name} tool with basic functionality"

        # Determine success criteria based on schema
        success_criteria = self._determine_success_criteria(input_schema, description)

        return TestCase(
            tool_name=tool_name,
            test_name="basic_functionality",
            description=f"Basic functionality test for {tool_name}",
            prompt=prompt,
            tool_schema=input_schema,
            generated_arguments=generated_args,
            success_criteria=success_criteria,
            context_requirements=context_reqs,
            auto_generated=True,
            test_type="basic",
        )

    def _generate_parameter_test(
        self, tool_name: str, description: str, input_schema: Dict
    ) -> Optional[TestCase]:
        """Generate a parameter validation test"""

        properties = input_schema.get("properties", {})
        if not properties:
            return None

        # Generate arguments with parameter focus
        generated_args, context_reqs = self._generate_arguments_from_schema(
            input_schema, focus_on_params=True
        )

        param_names = list(properties.keys())
        prompt = f"Test the {tool_name} tool with parameters: {', '.join(param_names)}"

        success_criteria = {
            "expects_parameters": True,
            "parameter_count": len(param_names),
            "no_parameter_errors": True,
        }

        return TestCase(
            tool_name=tool_name,
            test_name="parameter_validation",
            description=f"Parameter validation test for {tool_name}",
            prompt=prompt,
            tool_schema=input_schema,
            generated_arguments=generated_args,
            success_criteria=success_criteria,
            context_requirements=context_reqs,
            auto_generated=True,
            test_type="parameter",
        )

    def _generate_error_test(
        self, tool_name: str, description: str, input_schema: Dict
    ) -> Optional[TestCase]:
        """Generate an error handling test"""

        # Generate minimal or invalid arguments
        generated_args = {}

        prompt = f"Test error handling for {tool_name} tool"

        success_criteria = {
            "handles_errors_gracefully": True,
            "provides_error_message": True,
        }

        return TestCase(
            tool_name=tool_name,
            test_name="error_handling",
            description=f"Error handling test for {tool_name}",
            prompt=prompt,
            tool_schema=input_schema,
            generated_arguments=generated_args,
            success_criteria=success_criteria,
            context_requirements=[],
            auto_generated=True,
            critical=False,  # Error tests are less critical
            test_type="error",
        )

    def _generate_arguments_from_schema(
        self, input_schema: Dict, focus_on_params: bool = False
    ) -> tuple[Dict[str, Any], List[str]]:
        """Generate arguments from MCP schema and identify context requirements"""

        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        generated_args = {}
        context_requirements = []

        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            prop_description = prop_schema.get("description", "")

            # Try to generate a reasonable value
            value, needs_context = self._generate_value_for_property(
                prop_name, prop_type, prop_description, prop_schema
            )

            if needs_context:
                context_requirements.append(f"{prop_name}: {prop_description}")

            if value is not None:
                generated_args[prop_name] = value
            elif prop_name in required:
                # For required parameters we can't generate, mark as context requirement
                context_requirements.append(
                    f"REQUIRED: {prop_name}: {prop_description}"
                )

        return generated_args, context_requirements

    def _generate_value_for_property(
        self, name: str, prop_type: str, description: str, schema: Dict
    ) -> tuple[Any, bool]:
        """Generate a value for a property and indicate if context is needed"""

        name_lower = name.lower()
        desc_lower = description.lower()

        # Check if this requires external context
        context_indicators = [
            "file",
            "path",
            "url",
            "id",
            "key",
            "token",
            "specific",
            "existing",
            "valid",
        ]

        needs_context = any(
            indicator in name_lower or indicator in desc_lower
            for indicator in context_indicators
        )

        if prop_type == "string":
            # Try to generate reasonable string values
            if "file" in name_lower or "path" in name_lower:
                # For file paths, use generic test values
                return "test_file.txt", True
            elif "url" in name_lower:
                return "https://example.com/test", True
            elif "id" in name_lower:
                return "test_id_123", True
            elif "topic" in name_lower or "query" in name_lower:
                return "test query", False
            elif "name" in name_lower:
                return "test_name", False
            else:
                return "test_value", needs_context

        elif prop_type == "integer" or prop_type == "number":
            # Check for common numeric parameters
            if "max" in name_lower or "limit" in name_lower:
                return 5, False
            elif "count" in name_lower:
                return 3, False
            else:
                return 1, needs_context

        elif prop_type == "boolean":
            return True, False

        elif prop_type == "array":
            items_schema = schema.get("items", {})
            if items_schema.get("type") == "string":
                return ["test_item"], needs_context
            else:
                return [], needs_context

        elif prop_type == "object":
            return {}, needs_context

        else:
            return None, True

    def _determine_success_criteria(
        self, input_schema: Dict, description: str
    ) -> Dict[str, Any]:
        """Determine success criteria based on schema and description"""

        criteria = {
            "min_response_length": 10,  # Basic response length check
            "no_error_keywords": ["error", "failed", "exception", "invalid"],
        }

        # Add criteria based on description keywords
        desc_lower = description.lower()

        if "json" in desc_lower:
            criteria["expects_json"] = True

        if "list" in desc_lower or "array" in desc_lower:
            criteria["expects_list"] = True

        if "search" in desc_lower or "find" in desc_lower:
            criteria["expects_results"] = True
            criteria["acceptable_not_found"] = (
                True  # Search can legitimately return no results
            )

        if "read" in desc_lower or "get" in desc_lower:
            criteria["expects_content"] = True
            criteria["acceptable_not_found"] = True  # Files might not exist

        return criteria


class FlightChecker:
    """Flight checker that works with any MCP tools"""

    def __init__(
        self,
        chatbot_instance,
        config_path: str = "test_cases.json",
        test_mode: str = "basic",  # 'basic' or 'comprehensive'
        load_learned_tests: bool = False,
    ):
        self.chatbot = chatbot_instance
        self.config_path = config_path
        self.test_mode = test_mode
        self.load_learned_tests = load_learned_tests
        self.verbosity = VerbosityLevel.MINIMAL
        self.test_generator = TestGenerator(test_mode=test_mode)
        self.optimizer = DSPyOptimizer(config_path)
        self.test_cases: Dict[str, List[TestCase]] = {}
        self.learned_tests_path = "learned_tests.json"
        self.created_test_files: List[str] = []  # Track files we create for cleanup

        # Import and initialize human review system
        from human_review_system import HumanReviewSystem

        self.human_review = HumanReviewSystem()

        system_print("Initializing Flight Checker...")
        debug_print(f"Config path: {self.config_path}")
        debug_print(f"Test mode: {self.test_mode}")
        debug_print(f"Load learned tests: {self.load_learned_tests}")
        debug_print(f"Available tools: {len(self.chatbot.available_tools)}")

        # Test DSPy connection
        dspy_print("Testing DSPy connection...")
        if self.optimizer.test_dspy_connection():
            success_print("DSPy optimizer is ready!")
        else:
            warning_print("DSPy optimizer failed - will use rule-based fallback")

        # Load configuration and generate tests
        try:
            self.load_existing_test_cases()
            self.auto_generate_test_cases()

            if self.load_learned_tests:
                self.load_learned_tests_file()

            total_tests = sum(len(tests) for tests in self.test_cases.values())
            success_print(
                f"Flight Checker initialized with {total_tests} test cases for {len(self.test_cases)} tools"
            )

        except Exception as e:
            error_print(f"Error during flight checker initialization: {e}")
            debug_print(f"Will continue with empty test cases")
            import traceback

            debug_print(f"Full traceback:\n{traceback.format_exc()}")

    def cleanup_test_files(self):
        """Clean up any test files that were created during testing"""
        for test_file in self.created_test_files:
            try:
                if os.path.exists(test_file):
                    os.remove(test_file)
                    debug_print(f"Cleaned up test file: {test_file}")
            except Exception as e:
                debug_print(f"Could not clean up test file {test_file}: {e}")

        self.created_test_files.clear()

    def load_existing_test_cases(self):
        """Load any existing test cases from configuration"""
        # Ensure config file exists and is valid
        ensure_test_cases_config(self.config_path)

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            # Load existing test cases if they exist
            test_cases_config = config.get("test_cases", {})
            for tool_name, tool_tests in test_cases_config.items():
                self.test_cases[tool_name] = []
                for test_config in tool_tests:
                    try:
                        test_case = TestCase(
                            tool_name=tool_name,
                            test_name=test_config.get("test_name", "unknown"),
                            description=test_config.get(
                                "description", "No description"
                            ),
                            prompt=test_config.get("prompt", f"Test {tool_name}"),
                            tool_schema=test_config.get("tool_schema", {}),
                            generated_arguments=test_config.get(
                                "generated_arguments", {}
                            ),
                            expected_indicators=test_config.get(
                                "expected_indicators", []
                            ),
                            timeout_seconds=test_config.get("timeout_seconds", 30.0),
                            critical=test_config.get("critical", True),
                            success_criteria=test_config.get("success_criteria", {}),
                            optimization_history=test_config.get(
                                "optimization_history", []
                            ),
                            auto_generated=test_config.get("auto_generated", False),
                            context_requirements=test_config.get(
                                "context_requirements", []
                            ),
                            test_type=test_config.get("test_type", "basic"),
                        )
                        self.test_cases[tool_name].append(test_case)
                    except Exception as e:
                        debug_print(f"Error loading test case for {tool_name}: {e}")

        except Exception as e:
            debug_print(f"Error loading test cases: {e}")

    def _create_empty_config(self):
        """Create an empty but valid configuration structure"""
        try:
            ensure_test_cases_config(self.config_path)
            debug_print(f"Ensured valid config at {self.config_path}")
        except Exception as e:
            error_print(f"Failed to create config file: {e}")

    def auto_generate_test_cases(self):
        """Automatically generate test cases for all available tools"""
        system_print("Auto-generating test cases from MCP tool schemas...")

        if not self.chatbot.available_tools:
            warning_print("No tools available for test generation")
            return

        generated_count = 0
        for tool in self.chatbot.available_tools:
            tool_name = tool.get("name")
            if not tool_name:
                debug_print("Tool missing name, skipping")
                continue

            # Check if we already have test cases for this tool
            existing_tests = self.test_cases.get(tool_name, [])

            # Skip if we already have auto-generated tests for this tool
            if any(test.auto_generated for test in existing_tests):
                debug_print(
                    f"Auto-generated test cases already exist for {tool_name}, skipping"
                )
                continue

            debug_print(f"Generating test cases for {tool_name}...")

            # Generate test cases from schema
            try:
                generated_tests = self.test_generator.generate_test_cases(tool)

                if generated_tests:
                    # Add to existing tests or create new list
                    if tool_name not in self.test_cases:
                        self.test_cases[tool_name] = []

                    self.test_cases[tool_name].extend(generated_tests)
                    generated_count += len(generated_tests)
                    debug_print(
                        f"Generated {len(generated_tests)} test cases for {tool_name}"
                    )
                else:
                    warning_print(f"No test cases generated for {tool_name}")
            except Exception as e:
                error_print(f"Error generating test cases for {tool_name}: {e}")
                debug_print(f"Tool schema: {tool}")

        if generated_count > 0:
            success_print(f"Auto-generated {generated_count} test cases")
            # Save generated test cases immediately
            self._save_test_cases()
        else:
            warning_print("No test cases were generated")

    def _save_test_cases(self):
        """Save current test cases to configuration file"""
        try:
            # Load existing config or create minimal structure
            config = {"test_cases": {}}

            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            existing_config = json.loads(content)
                            # Preserve existing dspy_config if it exists
                            if "dspy_config" in existing_config:
                                config["dspy_config"] = existing_config["dspy_config"]
                except (json.JSONDecodeError, Exception):
                    pass  # Use minimal config if existing file is corrupted

            # Always ensure dspy_config exists
            if "dspy_config" not in config:
                config["dspy_config"] = {"optimization_enabled": True}

            # Update test cases
            for tool_name, test_list in self.test_cases.items():
                config["test_cases"][tool_name] = []
                for test_case in test_list:
                    test_config = {
                        "test_name": test_case.test_name,
                        "description": test_case.description,
                        "prompt": test_case.prompt,
                        "tool_schema": test_case.tool_schema,
                        "generated_arguments": test_case.generated_arguments,
                        "expected_indicators": test_case.expected_indicators,
                        "timeout_seconds": test_case.timeout_seconds,
                        "critical": test_case.critical,
                        "success_criteria": test_case.success_criteria,
                        "optimization_history": test_case.optimization_history,
                        "auto_generated": test_case.auto_generated,
                        "context_requirements": test_case.context_requirements,
                        "test_type": test_case.test_type,
                    }
                    config["test_cases"][tool_name].append(test_config)

            atomic_write_json(config, self.config_path)
            debug_print(
                f"Saved {sum(len(tests) for tests in self.test_cases.values())} test cases to {self.config_path}"
            )

        except Exception as e:
            error_print(f"Error saving test cases: {e}")

    def load_learned_tests_file(self):
        """Load additional tests that succeeded previously"""
        if not self.load_learned_tests:
            return

        path = Path(self.learned_tests_path)
        if not path.exists():
            debug_print(f"No learned tests file found at {self.learned_tests_path}")
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            warning_print(f"Error loading learned tests: {e}")
            return

        loaded_count = 0
        for tool_name, tests in data.items():
            for t in tests:
                # Check if we already have this test
                existing_tests = self.test_cases.get(tool_name, [])
                test_name = t.get("test_name", "learned")

                # Skip if we already have a test with the same name
                if any(test.test_name == test_name for test in existing_tests):
                    continue

                case = TestCase(
                    tool_name=tool_name,
                    test_name=test_name,
                    description=t.get("description", "learned test"),
                    prompt=t.get("prompt", ""),
                    tool_schema=t.get("tool_schema", {}),
                    generated_arguments=t.get("generated_arguments", {}),
                    optimization_history=t.get("optimization_history", []),
                    auto_generated=False,
                    test_type="learned",
                )
                self.test_cases.setdefault(tool_name, []).append(case)
                loaded_count += 1

        if loaded_count > 0:
            success_print(f"Loaded {loaded_count} learned test cases")

    async def run_single_test(self, test_case: TestCase) -> TestReport:
        """Execute a single test case"""
        start_time = time.time()

        # Check if tool exists
        if test_case.tool_name not in [
            tool["name"] for tool in self.chatbot.available_tools
        ]:
            return TestReport(
                test_case=test_case,
                result=TestResult.SKIP,
                execution_time=0,
                error_message=f"Tool '{test_case.tool_name}' not available",
            )

        # Check for context requirements
        if test_case.context_requirements:
            context_issues = self._check_context_availability(test_case)
            if context_issues:
                return TestReport(
                    test_case=test_case,
                    result=TestResult.CONTEXT_MISSING,
                    execution_time=time.time() - start_time,
                    error_message="Required context not available",
                    context_issues=context_issues,
                )

        try:
            # Execute the test
            response = await self._execute_test(test_case)

            # Validate the response
            validation_result = self._validate_response(test_case, response)

            return TestReport(
                test_case=test_case,
                result=(
                    TestResult.PASS if validation_result["valid"] else TestResult.FAIL
                ),
                execution_time=time.time() - start_time,
                response=response[:500] + "..." if len(response) > 500 else response,
                error_message=(
                    None if validation_result["valid"] else validation_result["reason"]
                ),
                validation_details=validation_result,
            )

        except asyncio.TimeoutError:
            return TestReport(
                test_case=test_case,
                result=TestResult.TIMEOUT,
                execution_time=time.time() - start_time,
                error_message=f"Test timed out after {test_case.timeout_seconds} seconds",
            )
        except Exception as e:
            return TestReport(
                test_case=test_case,
                result=TestResult.FAIL,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def _check_context_availability(self, test_case: TestCase) -> List[str]:
        """Check if required context is available for the test"""
        issues = []

        for requirement in test_case.context_requirements:
            if "REQUIRED:" in requirement:
                # This is a required parameter we couldn't generate
                issues.append(f"Missing required parameter: {requirement}")
            elif "file" in requirement.lower() or "path" in requirement.lower():
                # Check if we can find safe test files or create them
                safe_files = self._find_available_files()
                if not safe_files:
                    # We can create safe test files, so this isn't really a blocker
                    debug_print(
                        f"No existing safe test files, but can create them for: {requirement}"
                    )

        return issues

    def _find_available_files(self) -> List[str]:
        """Find safe test files that could be used for testing"""
        safe_test_files = []
        try:
            # Check current directory for safe test files only
            current_dir = Path(".")
            for file_path in current_dir.iterdir():
                if file_path.is_file() and self._is_safe_test_file(file_path.name):
                    safe_test_files.append(str(file_path))
        except Exception:
            pass

        return safe_test_files

    def _is_safe_test_file(self, filename: str) -> bool:
        """Check if a file is safe to use for testing (won't break the system)"""
        filename_lower = filename.lower()

        # Never use critical system files
        dangerous_files = [
            # Python source files
            ".py",
            # Configuration files
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".conf",
            # Environment files
            ".env",
            # Documentation
            ".md",
            ".rst",
            ".txt",
            # Database files
            ".db",
            ".sqlite",
            ".sqlite3",
            # Log files
            ".log",
            # Package files
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            # Hidden files
            ".",
        ]

        # Check if filename contains dangerous extensions or patterns
        for dangerous in dangerous_files:
            if dangerous in filename_lower:
                return False

        # Only allow explicitly safe test files
        safe_patterns = ["test_file", "sample", "dummy", "example", "temp"]

        return any(pattern in filename_lower for pattern in safe_patterns)

    def _create_safe_test_file(self) -> str:
        """Create a safe temporary test file for testing"""
        import time

        # Create a temporary file with safe content
        timestamp = int(time.time())
        test_filename = f"test_file_{timestamp}.txt"

        try:
            with open(test_filename, "w") as f:
                f.write("This is a safe test file created for MCP tool testing.\n")
                f.write(f"Created at: {time.ctime()}\n")
                f.write("This file can be safely deleted.\n")

            # Track the file for cleanup
            self.created_test_files.append(test_filename)
            debug_print(f"Created safe test file: {test_filename}")
            return test_filename
        except Exception as e:
            error_print(f"Failed to create safe test file: {e}")
            return "safe_test_file.txt"  # Fallback to safe name

    async def _execute_test(self, test_case: TestCase) -> str:
        """Execute a test case using the generated arguments"""
        session = self.chatbot.sessions.get(test_case.tool_name)
        if not session:
            raise Exception("Tool session not found")

        # Use generated arguments or try to improve them
        test_args = test_case.generated_arguments.copy()

        # Apply context-aware argument improvements
        test_args = self._improve_arguments_with_context(test_case, test_args)

        result = await asyncio.wait_for(
            session.call_tool(test_case.tool_name, arguments=test_args),
            timeout=test_case.timeout_seconds,
        )

        return result.content[0].text if result.content else str(result)

    def _improve_arguments_with_context(
        self, test_case: TestCase, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Improve arguments using available context - SAFELY"""
        improved_args = args.copy()

        # For file-related parameters, use safe test files only
        for arg_name, arg_value in args.items():
            if isinstance(arg_value, str) and (
                "file" in arg_name.lower() or "path" in arg_name.lower()
            ):
                # First try to find existing safe test files
                safe_files = self._find_available_files()
                if safe_files:
                    # Use the first safe test file
                    improved_args[arg_name] = safe_files[0]
                    debug_print(f"Using safe test file for {arg_name}: {safe_files[0]}")
                else:
                    # Create a new safe test file
                    safe_file = self._create_safe_test_file()
                    improved_args[arg_name] = safe_file
                    debug_print(f"Created safe test file for {arg_name}: {safe_file}")

        return improved_args

    def _validate_response(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Response validation based on success criteria"""
        validation_details = {
            "valid": False,
            "reason": "",
            "checks_performed": [],
        }

        criteria = test_case.success_criteria

        # Check minimum response length
        min_length = criteria.get("min_response_length", 10)
        if len(response) >= min_length:
            validation_details["checks_performed"].append(
                f"Length check passed ({len(response)} >= {min_length})"
            )
        else:
            validation_details["reason"] = (
                f"Response too short ({len(response)} < {min_length})"
            )
            return validation_details

        # Check for error keywords
        error_keywords = criteria.get("no_error_keywords", [])
        response_lower = response.lower()
        found_errors = [kw for kw in error_keywords if kw.lower() in response_lower]
        if found_errors:
            # Check if "not found" responses are acceptable
            if criteria.get("acceptable_not_found", False):
                not_found_phrases = [
                    "no saved information",
                    "not found",
                    "no information",
                    "does not exist",
                ]
                if any(phrase in response_lower for phrase in not_found_phrases):
                    validation_details["valid"] = True
                    validation_details["checks_performed"].append(
                        "Acceptable 'not found' response"
                    )
                    return validation_details

            validation_details["reason"] = f"Error keywords found: {found_errors}"
            return validation_details

        # Check JSON expectation
        if criteria.get("expects_json", False):
            try:
                json.loads(response)
                validation_details["checks_performed"].append("Valid JSON detected")
            except:
                if "{" in response and "}" in response:
                    validation_details["checks_performed"].append(
                        "JSON-like structure detected"
                    )
                else:
                    validation_details["reason"] = "Expected JSON content not found"
                    return validation_details

        # If we reach here, validation passed
        validation_details["valid"] = True
        return validation_details

    async def run_flight_check(
        self, parallel: bool = False, verbosity: VerbosityLevel = None
    ) -> FlightCheckReport:
        """Execute all flight checks with optimization capabilities"""
        if verbosity is not None:
            self.verbosity = verbosity

        # Apply any pending human fixes first
        fixes_applied = self.human_review.apply_human_fixes(self)
        if fixes_applied > 0:
            success_print(
                f"Applied {fixes_applied} human fixes - consider re-running tests"
            )

        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            header_print("Starting Tool Flight Check")
            separator_print()

        start_time = time.time()
        all_reports = []
        failed_tests = []

        # Run tests for all tools
        for tool_name, test_list in self.test_cases.items():
            if self.verbosity.value >= VerbosityLevel.MINIMAL.value:
                flight_check_print(f"\nTesting {tool_name}...")

            # Group tests by type for better organization
            test_groups = {}
            for test_case in test_list:
                test_type = getattr(test_case, "test_type", "basic")
                if test_type not in test_groups:
                    test_groups[test_type] = []
                test_groups[test_type].append(test_case)

            # Run tests in a logical order
            for test_type in ["basic", "parameter", "error", "learned"]:
                if test_type not in test_groups:
                    continue

                for test_case in test_groups[test_type]:
                    if self.verbosity.value >= VerbosityLevel.NORMAL.value:
                        colored_print(
                            f"  Running {test_case.test_name} ({test_case.test_type}): {test_case.description}",
                            Colors.FLIGHT_CHECK,
                        )
                    elif self.verbosity.value >= VerbosityLevel.MINIMAL.value:
                        colored_print(
                            f"  {test_case.test_name}...", Colors.FLIGHT_CHECK, end=" "
                        )

                    report = await self.run_single_test(test_case)
                    all_reports.append(report)

                    if report.result == TestResult.FAIL and test_case.critical:
                        failed_tests.append((test_case, report))
                    elif report.result == TestResult.PASS:
                        self._record_success(test_case)

                    self._print_test_result(report)

        # Attempt optimization for failed critical tests
        if failed_tests and self.optimizer.optimization_enabled:
            await self._optimize_failed_tests(failed_tests)

        # Generate summary report
        total_time = time.time() - start_time
        report = self._generate_flight_report(all_reports, total_time)

        # Print summary
        self._print_flight_summary(report)

        # Save failed tests for human review if there are critical failures
        if report.critical_failures > 0:
            self.human_review.save_failed_tests_for_review(all_reports)

        # Clean up any test files we created
        self.cleanup_test_files()

        return report

    async def _optimize_failed_tests(self, failed_tests: List[tuple]):
        """Optimize prompts for failed tests using DSPy"""
        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            optimization_print(
                f"Attempting optimization for {len(failed_tests)} failed tests..."
            )

        for test_case, report in failed_tests:
            if self.verbosity.value >= VerbosityLevel.VERBOSE.value:
                optimization_print(
                    f"  Optimizing {test_case.tool_name}.{test_case.test_name}..."
                )

            # Create optimization context
            context = OptimizationContext(
                tool_name=test_case.tool_name,
                tool_schema=test_case.tool_schema,
                original_prompt=test_case.prompt,
                failure_reason=f"{report.error_message} | Response: {report.response[:100] if report.response else 'None'}",
                generated_arguments=test_case.generated_arguments,
                success_criteria=test_case.success_criteria,
                previous_attempts=[
                    entry.get("optimized_prompt", "")
                    for entry in test_case.optimization_history
                ],
            )

            optimized_prompt = self.optimizer.optimize_prompt(context)

            if optimized_prompt != test_case.prompt:
                if self.verbosity.value >= VerbosityLevel.VERBOSE.value:
                    colored_print(f"    Original: '{test_case.prompt}'", Colors.WARNING)
                    colored_print(
                        f"    Optimized: '{optimized_prompt}'", Colors.SUCCESS
                    )

                # Update the test case with optimized prompt
                test_case.prompt = optimized_prompt

                # Add optimization record
                optimization_record = {
                    "timestamp": datetime.now().isoformat(),
                    "original_prompt": context.original_prompt,
                    "optimized_prompt": optimized_prompt,
                    "failure_context": context.failure_reason,
                    "strategy": "dspy_optimization",
                }
                test_case.optimization_history.append(optimization_record)

                # Save updated configuration
                self._save_test_cases()
            else:
                warning_print(
                    f"    No optimization applied for {test_case.tool_name}.{test_case.test_name}"
                )

    def _record_success(self, test_case: TestCase):
        """Record a successful test case"""
        path = Path(self.learned_tests_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        else:
            data = {}

        tests = data.setdefault(test_case.tool_name, [])
        for t in tests:
            if t.get("test_name") == test_case.test_name:
                t["success_count"] = t.get("success_count", 0) + 1
                t["last_success"] = datetime.now().isoformat()
                break
        else:
            tests.append(
                {
                    "test_name": test_case.test_name,
                    "description": test_case.description,
                    "prompt": test_case.prompt,
                    "tool_schema": test_case.tool_schema,
                    "generated_arguments": test_case.generated_arguments,
                    "success_count": 1,
                    "last_success": datetime.now().isoformat(),
                    "optimization_history": test_case.optimization_history,
                }
            )

        atomic_write_json(data, path)

    def _print_test_result(self, report: TestReport):
        """Print test result based on current verbosity level"""
        if self.verbosity == VerbosityLevel.QUIET:
            return

        status_text = {
            TestResult.PASS: "PASS",
            TestResult.FAIL: "FAIL",
            TestResult.SKIP: "SKIP",
            TestResult.TIMEOUT: "TIMEOUT",
            TestResult.CONTEXT_MISSING: "CONTEXT_MISSING",
        }

        color_map = {
            TestResult.PASS: Colors.TEST_PASS,
            TestResult.FAIL: Colors.TEST_FAIL,
            TestResult.SKIP: Colors.TEST_SKIP,
            TestResult.TIMEOUT: Colors.TEST_TIMEOUT,
            TestResult.CONTEXT_MISSING: Colors.WARNING,
        }

        if self.verbosity == VerbosityLevel.MINIMAL:
            colored_print(
                f"{status_text[report.result]} ({report.execution_time:.2f}s)",
                color_map[report.result],
            )
            if report.result in [TestResult.FAIL, TestResult.CONTEXT_MISSING]:
                error_print(f"    {report.error_message}")
                if report.context_issues:
                    for issue in report.context_issues:
                        warning_print(f"    Context: {issue}")

        elif self.verbosity.value >= VerbosityLevel.NORMAL.value:
            colored_print(
                f"    {status_text[report.result]} ({report.execution_time:.2f}s)",
                color_map[report.result],
            )
            if report.error_message:
                error_print(f"    {report.error_message}")

            if report.context_issues:
                for issue in report.context_issues:
                    warning_print(f"    Context Issue: {issue}")

            if self.verbosity.value >= VerbosityLevel.VERBOSE.value:
                if report.result == TestResult.FAIL and report.response:
                    colored_print(
                        f"    Response preview: {report.response[:200]}...",
                        Colors.TOOL_RESPONSE,
                    )
                elif report.result == TestResult.PASS and report.response:
                    colored_print(
                        f"    Response looks good: {report.response[:100]}...",
                        Colors.TOOL_RESPONSE,
                    )

                if self.verbosity == VerbosityLevel.DEBUG and report.validation_details:
                    debug_print(f"    Validation details: {report.validation_details}")

    def _generate_flight_report(
        self, test_reports: List[TestReport], total_time: float
    ) -> FlightCheckReport:
        """Generate comprehensive flight check report"""
        passed = sum(1 for r in test_reports if r.result == TestResult.PASS)
        failed = sum(1 for r in test_reports if r.result == TestResult.FAIL)
        skipped = sum(1 for r in test_reports if r.result == TestResult.SKIP)
        timeout = sum(1 for r in test_reports if r.result == TestResult.TIMEOUT)
        context_missing = sum(
            1 for r in test_reports if r.result == TestResult.CONTEXT_MISSING
        )

        critical_failures = sum(
            1
            for r in test_reports
            if r.result in [TestResult.FAIL, TestResult.TIMEOUT]
            and r.test_case.critical
        )

        system_ready = critical_failures == 0

        return FlightCheckReport(
            total_tests=len(test_reports),
            passed=passed,
            failed=failed,
            skipped=skipped,
            timeout=timeout,
            context_missing=context_missing,
            critical_failures=critical_failures,
            execution_time=total_time,
            test_reports=test_reports,
            system_ready=system_ready,
        )

    def _print_flight_summary(self, report: FlightCheckReport):
        """Print formatted flight check summary"""
        separator_print()
        header_print("FLIGHT CHECK SUMMARY")
        separator_print()

        flight_check_print(f"Total Tests: {report.total_tests}")
        success_print(f"Passed: {report.passed}")
        error_print(f"Failed: {report.failed}")
        warning_print(f"Skipped: {report.skipped}")
        colored_print(f"Timeout: {report.timeout}", Colors.TEST_TIMEOUT)
        colored_print(f"Context Missing: {report.context_missing}", Colors.WARNING)
        error_print(f"Critical Failures: {report.critical_failures}")
        system_print(f"Total Time: {report.execution_time:.2f}s")

        if report.system_ready:
            success_print(
                "\nSYSTEM READY FOR TAKEOFF! All critical systems operational."
            )
        else:
            error_print(
                f"\nSYSTEM NOT READY - {report.critical_failures} critical failure(s) detected!"
            )
            error_print("Critical failures:")
            for test_report in report.test_reports:
                if (
                    test_report.result in [TestResult.FAIL, TestResult.TIMEOUT]
                    and test_report.test_case.critical
                ):
                    colored_print(
                        f"   - {test_report.test_case.tool_name}.{test_report.test_case.test_name}: {test_report.error_message}",
                        Colors.ERROR,
                    )

        # Show context issues summary
        context_issues = []
        for test_report in report.test_reports:
            if test_report.result == TestResult.CONTEXT_MISSING:
                context_issues.extend(test_report.context_issues)

        if context_issues:
            warning_print(f"\nContext Issues Detected ({len(context_issues)} total):")
            unique_issues = list(set(context_issues))
            for issue in unique_issues[:5]:  # Show first 5 unique issues
                warning_print(f"   - {issue}")
            if len(unique_issues) > 5:
                warning_print(f"   ... and {len(unique_issues) - 5} more")

        separator_print()

    def export_report(self, report: FlightCheckReport, filename: str = None):
        """Export flight check report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flight_check_report_{timestamp}.json"

        # Convert to serializable format
        report_dict = {
            "summary": {
                "total_tests": report.total_tests,
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
                "timeout": report.timeout,
                "context_missing": report.context_missing,
                "critical_failures": report.critical_failures,
                "execution_time": report.execution_time,
                "system_ready": report.system_ready,
                "timestamp": datetime.now().isoformat(),
            },
            "test_details": [
                {
                    "tool_name": tr.test_case.tool_name,
                    "test_name": tr.test_case.test_name,
                    "test_type": getattr(tr.test_case, "test_type", "basic"),
                    "description": tr.test_case.description,
                    "prompt_used": tr.test_case.prompt,
                    "tool_schema": tr.test_case.tool_schema,
                    "generated_arguments": tr.test_case.generated_arguments,
                    "result": tr.result.value,
                    "execution_time": tr.execution_time,
                    "critical": tr.test_case.critical,
                    "auto_generated": tr.test_case.auto_generated,
                    "error_message": tr.error_message,
                    "context_issues": tr.context_issues,
                    "response_preview": (
                        tr.response[:200] + "..."
                        if tr.response and len(tr.response) > 200
                        else tr.response
                    ),
                    "validation_details": tr.validation_details,
                    "optimization_history": tr.test_case.optimization_history,
                    "context_requirements": tr.test_case.context_requirements,
                }
                for tr in report.test_reports
            ],
        }

        atomic_write_json(report_dict, filename)
        print(f"Flight check report exported to: {filename}")
        return filename
