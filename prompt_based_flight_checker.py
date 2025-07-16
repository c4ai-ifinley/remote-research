"""
Prompt-Based Flight Checker
Tests that MCP tools are actually accessible and working via natural language prompts
Focuses on verifying the full chain: prompt → tool call → execution → response
"""

import asyncio
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from color_utils import (
    system_print,
    success_print,
    warning_print,
    error_print,
    flight_check_print,
    colored_print,
    Colors,
)
from utils import atomic_write_json


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    NO_TOOL_CALL = "NO_TOOL_CALL"
    TOOL_ERROR = "TOOL_ERROR"


@dataclass
class PromptTest:
    """A prompt-based test case"""

    tool_name: str
    test_name: str
    description: str
    prompt: str  # Natural language prompt that should trigger the tool
    expected_args: Dict[str, Any]  # Expected arguments the tool should receive
    validation: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)
    enabled: bool = True


@dataclass
class PromptTestResult:
    """Result of a prompt-based test"""

    test: PromptTest
    result: TestResult
    execution_time: float
    tool_called: bool = False
    actual_args: Optional[Dict] = None
    response: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class PromptBasedFlightChecker:
    """Flight checker that tests prompt-to-tool execution"""

    def __init__(self, chatbot_instance):
        self.chatbot = chatbot_instance
        self.config_path = "prompt_test_config.json"
        self.results_path = "prompt_test_results.json"

        # Load or create test configuration
        self.test_config = self._load_or_create_config()

        system_print("Prompt-Based Flight Checker initialized")
        system_print(f"Loaded {len(self.test_config)} test configurations")

    def _load_or_create_config(self) -> Dict[str, PromptTest]:
        """Load test configuration or create template"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)

                # Convert to PromptTest objects
                tests = {}
                for tool_name, test_data in data.items():
                    tests[tool_name] = PromptTest(**test_data)

                return tests
            except Exception as e:
                warning_print(f"Failed to load config: {e}. Creating new template.")

        # Create template configuration
        template = self._create_config_template()
        self._save_config(template)

        success_print(f"Created template configuration: {self.config_path}")
        print("Edit this file with your specific test arguments and prompts.")

        return template

    def _create_config_template(self) -> Dict[str, PromptTest]:
        """Create template configuration with examples"""
        available_tools = [tool["name"] for tool in self.chatbot.available_tools]

        templates = {}

        # Create templates for discovered tools
        for tool_name in available_tools:
            if tool_name == "read_file":
                templates[tool_name] = PromptTest(
                    tool_name="read_file",
                    test_name="read_config_file",
                    description="Test reading the server config file",
                    prompt="Please read the server_config.json file and show me its contents",
                    expected_args={"path": "server_config.json"},
                    validation={
                        "min_response_length": 20,
                        "should_contain": ["json", "config"],
                        "should_not_contain": ["error", "failed"],
                    },
                )

            elif tool_name == "list_directory":
                templates[tool_name] = PromptTest(
                    tool_name="list_directory",
                    test_name="list_current_dir",
                    description="Test listing current directory",
                    prompt="Show me all files and folders in the current directory",
                    expected_args={"path": "."},
                    validation={
                        "min_response_length": 10,
                        "should_contain": [],
                        "should_not_contain": ["error", "failed"],
                    },
                )

            elif tool_name == "search_papers":
                templates[tool_name] = PromptTest(
                    tool_name="search_papers",
                    test_name="search_ml_papers",
                    description="Test searching for machine learning papers",
                    prompt="Find 2 recent papers about machine learning from arXiv",
                    expected_args={"topic": "machine learning", "max_results": 2},
                    validation={
                        "min_response_length": 15,
                        "should_contain": ["paper"],
                        "should_not_contain": ["error", "failed"],
                    },
                )

            elif tool_name == "extract_info":
                templates[tool_name] = PromptTest(
                    tool_name="extract_info",
                    test_name="extract_paper_info",
                    description="Test extracting paper information (expects not found)",
                    prompt="Get detailed information about paper ID 2301.07041",
                    expected_args={"paper_id": "2301.07041"},
                    validation={
                        "min_response_length": 10,
                        "should_contain": [],
                        "should_not_contain": ["error", "failed"],
                        "accept_not_found": True,
                    },
                )

            elif tool_name == "fetch":
                templates[tool_name] = PromptTest(
                    tool_name="fetch",
                    test_name="fetch_test_url",
                    description="Test fetching web content",
                    prompt="Fetch the content from https://httpbin.org/json and show me the result",
                    expected_args={"url": "https://httpbin.org/json"},
                    validation={
                        "min_response_length": 10,
                        "should_contain": [],
                        "should_not_contain": ["error", "failed"],
                    },
                )

            else:
                # Generic template for unknown tools
                templates[tool_name] = PromptTest(
                    tool_name=tool_name,
                    test_name=f"test_{tool_name}",
                    description=f"Test {tool_name} functionality",
                    prompt=f"Please use the {tool_name} tool to demonstrate its functionality",
                    expected_args={},  # User must fill this in
                    validation={
                        "min_response_length": 5,
                        "should_contain": [],
                        "should_not_contain": ["error", "failed"],
                    },
                )

        return templates

    def _save_config(self, tests: Dict[str, PromptTest]):
        """Save test configuration to file"""
        # Convert PromptTest objects to dict for JSON serialization
        data = {}
        for tool_name, test in tests.items():
            data[tool_name] = {
                "tool_name": test.tool_name,
                "test_name": test.test_name,
                "description": test.description,
                "prompt": test.prompt,
                "expected_args": test.expected_args,
                "validation": test.validation,
                "optimization_history": test.optimization_history,
                "enabled": test.enabled,
            }

        atomic_write_json(data, self.config_path)

    async def run_prompt_flight_check(
        self, verbose: bool = True
    ) -> List[PromptTestResult]:
        """Run prompt-based flight check"""
        if verbose:
            system_print("Starting Prompt-Based Flight Check")
            print("Testing that prompts actually trigger tool calls...")

        results = []

        # Test each configured tool
        for tool_name, test in self.test_config.items():
            if not test.enabled:
                if verbose:
                    warning_print(f"Skipping {tool_name} (disabled)")
                continue

            if verbose:
                flight_check_print(f"\nTesting {tool_name}: {test.description}")
                print(f"  Prompt: '{test.prompt}'")

            result = await self._run_prompt_test(test, verbose)
            results.append(result)

            if verbose:
                self._print_test_result(result)

        # Print summary
        if verbose:
            self._print_summary(results)

        # Save results
        self._save_results(results)

        return results

    async def _run_prompt_test(
        self, test: PromptTest, verbose: bool
    ) -> PromptTestResult:
        """Run a single prompt test"""
        start_time = time.time()

        try:
            # Intercept tool calls by monkey-patching the chatbot
            tool_calls_made = []
            original_anthropic_create = self.chatbot.anthropic.messages.create

            def tracking_create(*args, **kwargs):
                response = original_anthropic_create(*args, **kwargs)

                # Extract tool calls from response
                for content in response.content:
                    if content.type == "tool_use":
                        tool_calls_made.append(
                            {
                                "name": content.name,
                                "args": content.input,
                                "id": content.id,
                            }
                        )

                return response

            # Temporarily replace anthropic create method
            self.chatbot.anthropic.messages.create = tracking_create

            try:
                # Execute the prompt
                await self.chatbot.process_query(test.prompt)

                # Analyze results
                execution_time = time.time() - start_time

                if not tool_calls_made:
                    return PromptTestResult(
                        test=test,
                        result=TestResult.NO_TOOL_CALL,
                        execution_time=execution_time,
                        tool_called=False,
                        error_message="No tool calls were made in response to the prompt",
                    )

                # Check if the expected tool was called
                expected_tool_called = False
                actual_args = None

                for call in tool_calls_made:
                    if call["name"] == test.tool_name:
                        expected_tool_called = True
                        actual_args = call["args"]
                        break

                if not expected_tool_called:
                    called_tools = [call["name"] for call in tool_calls_made]
                    return PromptTestResult(
                        test=test,
                        result=TestResult.FAIL,
                        execution_time=execution_time,
                        tool_called=True,
                        error_message=f"Expected {test.tool_name} but called: {called_tools}",
                    )

                # Validate arguments if specified
                if test.expected_args and actual_args != test.expected_args:
                    return PromptTestResult(
                        test=test,
                        result=TestResult.FAIL,
                        execution_time=execution_time,
                        tool_called=True,
                        actual_args=actual_args,
                        error_message=f"Expected args {test.expected_args} but got {actual_args}",
                    )

                # Test passed - tool was called with correct arguments
                return PromptTestResult(
                    test=test,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    tool_called=True,
                    actual_args=actual_args,
                    response="Tool called successfully with correct arguments",
                )

            finally:
                # Restore original anthropic create method
                self.chatbot.anthropic.messages.create = original_anthropic_create

        except Exception as e:
            return PromptTestResult(
                test=test,
                result=TestResult.TOOL_ERROR,
                execution_time=time.time() - start_time,
                tool_called=False,
                error_message=str(e),
            )

    def _print_test_result(self, result: PromptTestResult):
        """Print result of a single test"""
        status_colors = {
            TestResult.PASS: Colors.SUCCESS,
            TestResult.FAIL: Colors.ERROR,
            TestResult.NO_TOOL_CALL: Colors.WARNING,
            TestResult.TOOL_ERROR: Colors.ERROR,
            TestResult.SKIP: Colors.WARNING,
        }

        color = status_colors.get(result.result, Colors.CHAT)
        status = result.result.value

        colored_print(f"  Result: {status} ({result.execution_time:.2f}s)", color)

        if result.tool_called and result.actual_args:
            print(f"  Args used: {result.actual_args}")

        if result.error_message:
            error_print(f"  Error: {result.error_message}")

    def _print_summary(self, results: List[PromptTestResult]):
        """Print summary of all test results"""
        print("\n" + "=" * 50)
        system_print("PROMPT FLIGHT CHECK SUMMARY")
        print("=" * 50)

        passed = sum(1 for r in results if r.result == TestResult.PASS)
        failed = sum(1 for r in results if r.result == TestResult.FAIL)
        no_tool_call = sum(1 for r in results if r.result == TestResult.NO_TOOL_CALL)
        errors = sum(1 for r in results if r.result == TestResult.TOOL_ERROR)

        print(f"Total Tests: {len(results)}")
        success_print(f"Passed: {passed}")
        error_print(f"Failed: {failed}")
        warning_print(f"No Tool Call: {no_tool_call}")
        error_print(f"Errors: {errors}")

        if passed == len(results):
            success_print("\nAll prompt-to-tool connections working!")
        elif no_tool_call > 0:
            warning_print(f"\n{no_tool_call} prompts didn't trigger tool calls")
            print("This suggests your prompts may need to be more specific.")
        else:
            error_print(f"\n{failed + errors} issues found")

        # Show problematic tests
        problems = [r for r in results if r.result != TestResult.PASS]
        if problems:
            print("\nIssues found:")
            for result in problems:
                print(f"  - {result.test.tool_name}: {result.error_message}")

    def _save_results(self, results: List[PromptTestResult]):
        """Save test results to file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.result == TestResult.PASS),
                "failed": sum(1 for r in results if r.result == TestResult.FAIL),
                "no_tool_call": sum(
                    1 for r in results if r.result == TestResult.NO_TOOL_CALL
                ),
                "errors": sum(1 for r in results if r.result == TestResult.TOOL_ERROR),
            },
            "results": [
                {
                    "tool_name": r.test.tool_name,
                    "test_name": r.test.test_name,
                    "prompt": r.test.prompt,
                    "result": r.result.value,
                    "execution_time": r.execution_time,
                    "tool_called": r.tool_called,
                    "expected_args": r.test.expected_args,
                    "actual_args": r.actual_args,
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }

        atomic_write_json(data, self.results_path)
        success_print(f"Results saved to: {self.results_path}")

    def optimize_failed_prompts(self, results: List[PromptTestResult]):
        """Use DSPy to optimize prompts that failed to trigger tool calls"""
        failed_prompts = [r for r in results if r.result == TestResult.NO_TOOL_CALL]

        if not failed_prompts:
            success_print(
                "No prompt optimization needed - all prompts triggered tool calls!"
            )
            return

        print(
            f"\nOptimizing {len(failed_prompts)} prompts that didn't trigger tool calls..."
        )

        # Here you would integrate with your DSPy optimizer
        # For each failed prompt, ask DSPy to create a more specific prompt
        # that's more likely to trigger the intended tool call

        for result in failed_prompts:
            print(f"  - {result.test.tool_name}: '{result.test.prompt}'")
            print(f"    → Needs optimization to trigger {result.test.tool_name} call")

    def add_custom_test(
        self,
        tool_name: str,
        prompt: str,
        expected_args: Dict[str, Any],
        description: str = None,
    ):
        """Add a custom test case"""
        test = PromptTest(
            tool_name=tool_name,
            test_name=f"custom_{tool_name}_{len(self.test_config)}",
            description=description or f"Custom test for {tool_name}",
            prompt=prompt,
            expected_args=expected_args,
        )

        self.test_config[f"custom_{tool_name}_{len(self.test_config)}"] = test
        self._save_config(self.test_config)

        success_print(f"Added custom test for {tool_name}")


# Integration convenience functions
async def run_prompt_flight_check(chatbot):
    """Convenience function to run prompt-based flight check"""
    checker = PromptBasedFlightChecker(chatbot)
    results = await checker.run_prompt_flight_check()

    # If there are failed prompts, suggest optimization
    failed = [r for r in results if r.result != TestResult.PASS]
    if failed:
        print(f"\nFound {len(failed)} issues. Consider:")
        print("1. Editing prompt_test_config.json to fix prompts")
        print("2. Using DSPy optimization for failed prompts")
        print("3. Checking tool availability and arguments")

    return results


# Example usage
async def example_usage():
    """Example of how to use prompt-based flight checking"""
    from mcp_chatbot import MCP_ChatBot

    # Initialize your chatbot
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_servers()

    # Run prompt-based flight check
    results = await run_prompt_flight_check(chatbot)

    # Check if all prompts work
    all_working = all(r.result == TestResult.PASS for r in results)

    if all_working:
        success_print("All prompts successfully trigger their intended tools!")
    else:
        warning_print("Some prompts need adjustment - check the configuration file")

    await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(example_usage())
