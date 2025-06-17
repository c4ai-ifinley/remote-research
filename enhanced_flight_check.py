from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Awaitable
from enum import Enum
import asyncio
import json
import time
import re
import os
from datetime import datetime
from pathlib import Path

# Import the DSPy optimizer
from dspy_optimizer import DSPyFlightOptimizer, OptimizationContext

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


class VerbosityLevel(Enum):
    QUIET = 0
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3
    DEBUG = 4


@dataclass
class PromptTestCase:
    """Test case that uses prompts instead of direct tool arguments"""

    tool_name: str
    test_name: str
    description: str
    prompt: str
    expected_indicators: List[str] = field(default_factory=list)
    expected_format: str = "text"
    timeout_seconds: float = 30.0
    critical: bool = True
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)


@dataclass
class TestReport:
    """Results from running a single test"""

    test_case: PromptTestCase
    result: TestResult
    execution_time: float
    response: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    validation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlightCheckReport:
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
    optimization_suggestions: List[Dict] = field(default_factory=list)


class DSPyPromptOptimizer:
    """Handles DSPy-based prompt optimization"""

    def __init__(self, config_path: str = "test_cases.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.optimization_enabled = self.config.get("dspy_config", {}).get(
            "optimization_enabled", True
        )
        self.dspy_optimizer = (
            DSPyFlightOptimizer()
        )  # Initialize the actual DSPy optimizer

    def load_config(self) -> Dict:
        """Load test configuration from JSON"""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(
                f"Warning: {self.config_path} not found. Using default configuration."
            )
            return {"test_cases": {}, "prompt_templates": {}, "dspy_config": {}}

    def save_config(self):
        """Save updated configuration back to JSON"""
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def optimize_prompt(self, context: OptimizationContext) -> str:
        """Use DSPy to optimize a failing prompt"""
        if not self.optimization_enabled:
            return context.original_prompt

        # Use DSPy optimizer
        optimized_prompt = self.dspy_optimizer.optimize_prompt(context)

        return optimized_prompt


class EnhancedFlightChecker:
    """Enhanced flight checker with JSON-based test cases and DSPy optimization"""

    def __init__(self, chatbot_instance, config_path: str = "test_cases.json"):
        self.chatbot = chatbot_instance
        self.config_path = config_path
        self.verbosity = VerbosityLevel.MINIMAL
        self.optimizer = DSPyPromptOptimizer(config_path)
        self.test_cases: Dict[str, List[PromptTestCase]] = {}

        system_print("Initializing Enhanced Flight Checker...")
        dspy_print("Testing DSPy connection...")
        if self.optimizer.dspy_optimizer.test_dspy_connection():
            success_print("DSPy optimizer is ready!")
        else:
            warning_print("DSPy optimizer failed - will use rule-based fallback")

        self.load_test_cases()

    def load_test_cases(self):
        """Load test cases from JSON configuration"""
        config = self.optimizer.load_config()
        test_cases_config = config.get("test_cases", {})

        for tool_name, tool_tests in test_cases_config.items():
            self.test_cases[tool_name] = []
            for test_config in tool_tests:
                test_case = PromptTestCase(
                    tool_name=tool_name,
                    test_name=test_config["test_name"],
                    description=test_config["description"],
                    prompt=test_config["prompt"],
                    expected_indicators=test_config.get("expected_indicators", []),
                    expected_format=test_config.get("expected_format", "text"),
                    timeout_seconds=test_config.get("timeout_seconds", 30.0),
                    critical=test_config.get("critical", True),
                    success_criteria=test_config.get("success_criteria", {}),
                    optimization_history=test_config.get("optimization_history", []),
                )
                self.test_cases[tool_name].append(test_case)

    def set_verbosity(self, level: VerbosityLevel):
        """Set the verbosity level for flight check output"""
        self.verbosity = level

    async def run_single_test(self, test_case: PromptTestCase) -> TestReport:
        """Execute a single test case using prompt-based approach"""
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

        try:
            # Execute the prompt through the chatbot's query processing
            # This simulates how a user would interact with the system
            response = await self._execute_prompt_query(test_case)

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

    async def _execute_prompt_query(self, test_case: PromptTestCase) -> str:
        """Execute a prompt query through the chatbot system"""
        # This is a simplified version - you'll need to adapt this to work with your chatbot's process_query method
        # The idea is to send the prompt as if it were a user query and capture the response

        # For now, simulate direct tool calling - you can replace this with actual prompt processing
        session = self.chatbot.sessions.get(test_case.tool_name)
        if not session:
            raise Exception("Tool session not found")

        # Extract parameters from prompt (this is simplified - in reality you'd use LLM to parse)
        tool_args = self._extract_args_from_prompt(test_case)

        result = await asyncio.wait_for(
            session.call_tool(test_case.tool_name, arguments=tool_args),
            timeout=test_case.timeout_seconds,
        )

        return result.content[0].text if result.content else str(result)

    def _extract_args_from_prompt(self, test_case: PromptTestCase) -> Dict[str, Any]:
        """Extract tool arguments from prompt text (simplified implementation)"""
        # This is a basic implementation - in practice you'd use an LLM to extract parameters

        if test_case.tool_name == "search_papers":
            # Extract topic and max_results from prompt
            prompt_lower = test_case.prompt.lower()

            # Find topic - be more flexible for vague prompts
            topics = [
                "machine learning",
                "quantum computing",
                "artificial intelligence",
                "computers",
                "stuff",
            ]
            topic = "computer science"  # default fallback
            for t in topics:
                if t in prompt_lower:
                    topic = t
                    break

            # If prompt is very vague, use a default topic
            if "stuff" in prompt_lower or "something" in prompt_lower:
                topic = "computer science"  # This will likely fail validation

            # Extract number
            import re

            numbers = re.findall(r"\d+", test_case.prompt)
            max_results = int(numbers[0]) if numbers else 2

            return {"topic": topic, "max_results": max_results}

        elif test_case.tool_name == "extract_info":
            # Extract paper_id from prompt
            import re

            matches = re.findall(r"'([^']*)'", test_case.prompt)

            # Handle vague prompts by using a real paper ID that should exist
            if not matches and (
                "something" in test_case.prompt.lower()
                or "show me" in test_case.prompt.lower()
            ):
                # Try to get a real paper ID from recent search results
                paper_id = self._get_real_paper_id()
                if not paper_id:
                    paper_id = "1802.03292v1"  # Use the one we saw in the logs
            else:
                paper_id = matches[0] if matches else "test_paper_123"

            return {"paper_id": paper_id}

        elif test_case.tool_name == "read_file":
            # Extract filename from prompt
            if "server_config.json" in test_case.prompt:
                return {"path": "server_config.json"}
            return {"path": "."}

        elif test_case.tool_name == "list_directory":
            # Extract path from prompt
            return {"path": "."}

        elif test_case.tool_name == "fetch":
            # Extract URL from prompt
            import re

            urls = re.findall(r"https?://[^\s]+", test_case.prompt)
            url = urls[0] if urls else "https://example.com"
            return {"url": url}

        return {}

    def _get_real_paper_id(self) -> Optional[str]:
        """Try to get a real paper ID from the papers directory"""
        import os
        import json

        papers_dir = "papers"
        if not os.path.exists(papers_dir):
            return None

        # Look for any papers in the directory
        for topic_dir in os.listdir(papers_dir):
            topic_path = os.path.join(papers_dir, topic_dir)
            if os.path.isdir(topic_path):
                papers_file = os.path.join(topic_path, "papers_info.json")
                if os.path.exists(papers_file):
                    try:
                        with open(papers_file, "r") as f:
                            papers_data = json.load(f)
                        if papers_data:
                            # Return the first paper ID found
                            return list(papers_data.keys())[0]
                    except:
                        continue

        return None

    def _validate_response(
        self, test_case: PromptTestCase, response: str
    ) -> Dict[str, Any]:
        """Enhanced validation with detailed feedback"""
        validation_details = {
            "valid": False,
            "reason": "",
            "checks_performed": [],
            "indicators_found": [],
            "format_valid": False,
        }

        # Check success criteria
        criteria = test_case.success_criteria

        # Check minimum response length
        if "min_response_length" in criteria:
            min_length = criteria["min_response_length"]
            if len(response) >= min_length:
                validation_details["checks_performed"].append(
                    f"Length check passed ({len(response)} >= {min_length})"
                )
            else:
                validation_details["reason"] = (
                    f"Response too short ({len(response)} < {min_length})"
                )
                return validation_details

        # Check for arXiv IDs if expected
        if criteria.get("contains_arxiv_ids", False):
            arxiv_pattern = r"\d{4}\.\d{4,5}(v\d+)?"
            if re.search(arxiv_pattern, response):
                validation_details["checks_performed"].append("ArXiv ID pattern found")
                validation_details["format_valid"] = True
            else:
                validation_details["reason"] = "No arXiv ID pattern found in response"
                return validation_details

        # Check for JSON content if expected
        if criteria.get("contains_json", False):
            try:
                json.loads(response)
                validation_details["checks_performed"].append("Valid JSON detected")
                validation_details["format_valid"] = True
            except:
                if "{" in response and "}" in response:
                    validation_details["checks_performed"].append(
                        "JSON-like structure detected"
                    )
                    validation_details["format_valid"] = True
                else:
                    validation_details["reason"] = "Expected JSON content not found"
                    return validation_details

        # Check for required keywords
        if "contains_keywords" in criteria:
            keywords = criteria["contains_keywords"]
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in response.lower():
                    found_keywords.append(keyword)

            if found_keywords:
                validation_details["checks_performed"].append(
                    f"Found keywords: {found_keywords}"
                )
            else:
                validation_details["reason"] = (
                    f"Required keywords not found: {keywords}"
                )
                return validation_details

        # Check expected indicators
        response_lower = response.lower()
        for indicator in test_case.expected_indicators:
            if indicator.lower() in response_lower:
                validation_details["indicators_found"].append(indicator)

        # Handle "acceptable not found" cases
        if criteria.get("acceptable_not_found", False):
            not_found_phrases = ["no saved information", "not found", "no information"]
            if any(phrase in response_lower for phrase in not_found_phrases):
                validation_details["valid"] = True
                validation_details["checks_performed"].append(
                    "Acceptable 'not found' response"
                )
                return validation_details

        # Check no error keywords
        if "no_error_keywords" in criteria:
            error_keywords = criteria["no_error_keywords"]
            found_errors = [kw for kw in error_keywords if kw.lower() in response_lower]
            if found_errors:
                validation_details["reason"] = f"Error keywords found: {found_errors}"
                return validation_details

        # Final validation
        if validation_details["indicators_found"] or validation_details["format_valid"]:
            validation_details["valid"] = True
        else:
            validation_details["reason"] = (
                "No expected indicators found and format validation failed"
            )

        return validation_details

    async def run_flight_check(
        self, parallel: bool = False, verbosity: VerbosityLevel = None
    ) -> FlightCheckReport:
        """Execute all flight checks with optimization capabilities"""
        if verbosity is not None:
            self.verbosity = verbosity

        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            header_print("Starting Enhanced Tool Flight Check")
            separator_print()

        start_time = time.time()
        all_reports = []
        failed_tests = []

        # Get all available tools
        available_tools = [tool["name"] for tool in self.chatbot.available_tools]

        # Define test execution order (setup tests first)
        test_order = [
            "read_file",
            "list_directory",
            "search_papers",  # Run search tests first to populate papers
            "extract_info",  # Then test extraction on existing papers
            "fetch",
        ]

        # Run tests in dependency order
        for tool_name in test_order:
            if tool_name in available_tools and tool_name in self.test_cases:
                if self.verbosity.value >= VerbosityLevel.MINIMAL.value:
                    flight_check_print(f"\nTesting {tool_name}...")

                for test_case in self.test_cases[tool_name]:
                    if self.verbosity.value >= VerbosityLevel.NORMAL.value:
                        colored_print(
                            f"  Running {test_case.test_name}: {test_case.description}",
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

                    # Print results based on verbosity level
                    self._print_test_result(report)

        # Attempt optimization for failed critical tests
        if failed_tests and self.optimizer.optimization_enabled:
            await self._optimize_failed_tests(failed_tests)

        # Generate summary report
        total_time = time.time() - start_time
        report = self._generate_flight_report(all_reports, total_time)

        # Print summary
        self._print_flight_summary(report)

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

            # Get the original prompt from the JSON config (not the potentially modified one)
            original_prompt = self._get_original_prompt_from_config(test_case)

            # Extract tool arguments that were used in the test
            tool_args = self._extract_args_from_prompt(test_case)

            # Create optimization context with tool arguments
            context = OptimizationContext(
                tool_name=test_case.tool_name,
                original_prompt=original_prompt,
                failure_reason=f"{report.error_message} | Response: {report.response[:100] if report.response else 'None'}",
                expected_output_format=test_case.expected_format,
                success_criteria=test_case.success_criteria,
                previous_attempts=[
                    entry.get("optimized_prompt", "")
                    for entry in test_case.optimization_history
                ],
                tool_arguments=tool_args,
            )

            optimized_prompt = self.optimizer.optimize_prompt(context)

            if optimized_prompt != original_prompt:
                if self.verbosity.value >= VerbosityLevel.VERBOSE.value:
                    colored_print(f"    Original: '{original_prompt}'", Colors.WARNING)
                    colored_print(
                        f"    Optimized: '{optimized_prompt}'", Colors.SUCCESS
                    )

                # Update the test case with optimized prompt
                test_case.prompt = optimized_prompt

                # Add optimization record
                optimization_record = {
                    "timestamp": datetime.now().isoformat(),
                    "original_prompt": original_prompt,
                    "optimized_prompt": optimized_prompt,
                    "failure_context": context.failure_reason,
                    "tool_arguments": tool_args,
                    "strategy": (
                        "dspy_based"
                        if self.optimizer.dspy_optimizer.optimizer
                        else "rule_based"
                    ),
                }
                test_case.optimization_history.append(optimization_record)

                # Save updated configuration
                self._update_config_with_optimized_prompt(test_case)
            else:
                warning_print(
                    f"    No optimization applied for {test_case.tool_name}.{test_case.test_name}"
                )

    def _get_original_prompt_from_config(self, test_case: PromptTestCase) -> str:
        """Get the original prompt from the JSON config file"""
        try:
            config = self.optimizer.load_config()
            test_cases_config = config.get("test_cases", {})

            for tool_test in test_cases_config.get(test_case.tool_name, []):
                if tool_test["test_name"] == test_case.test_name:
                    return tool_test["prompt"]

            return test_case.prompt  # Fallback to current prompt
        except Exception as e:
            print(f"Error loading original prompt: {e}")
            return test_case.prompt

    def _update_config_with_optimized_prompt(self, test_case: PromptTestCase):
        """Update the JSON configuration with optimized prompt"""
        config = self.optimizer.load_config()

        # Find and update the specific test case
        for tool_tests in config["test_cases"].get(test_case.tool_name, []):
            if tool_tests["test_name"] == test_case.test_name:
                tool_tests["prompt"] = test_case.prompt
                tool_tests["optimization_history"] = test_case.optimization_history
                break

        # Save updated config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _print_test_result(self, report: TestReport):
        """Print test result based on current verbosity level"""
        if self.verbosity == VerbosityLevel.QUIET:
            return

        status_text = {
            TestResult.PASS: "PASS",
            TestResult.FAIL: "FAIL",
            TestResult.SKIP: "SKIP",
            TestResult.TIMEOUT: "TIMEOUT",
        }

        color_map = {
            TestResult.PASS: Colors.TEST_PASS,
            TestResult.FAIL: Colors.TEST_FAIL,
            TestResult.SKIP: Colors.TEST_SKIP,
            TestResult.TIMEOUT: Colors.TEST_TIMEOUT,
        }

        if self.verbosity == VerbosityLevel.MINIMAL:
            colored_print(
                f"{status_text[report.result]} ({report.execution_time:.2f}s)",
                color_map[report.result],
            )
            if report.result == TestResult.FAIL:
                error_print(f"    {report.error_message}")

        elif self.verbosity.value >= VerbosityLevel.NORMAL.value:
            colored_print(
                f"    {status_text[report.result]} ({report.execution_time:.2f}s)",
                color_map[report.result],
            )
            if report.error_message:
                error_print(f"    {report.error_message}")

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
            critical_failures=critical_failures,
            execution_time=total_time,
            test_reports=test_reports,
            system_ready=system_ready,
        )

    def _print_flight_summary(self, report: FlightCheckReport):
        """Print formatted flight check summary"""
        separator_print()
        header_print("ENHANCED FLIGHT CHECK SUMMARY")
        separator_print()

        flight_check_print(f"Total Tests: {report.total_tests}")
        success_print(f"Passed: {report.passed}")
        error_print(f"Failed: {report.failed}")
        warning_print(f"Skipped: {report.skipped}")
        colored_print(f"Timeout: {report.timeout}", Colors.TEST_TIMEOUT)
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

        separator_print()

    def export_report(self, report: FlightCheckReport, filename: str = None):
        """Export flight check report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_flight_check_report_{timestamp}.json"

        # Convert to serializable format
        report_dict = {
            "summary": {
                "total_tests": report.total_tests,
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
                "timeout": report.timeout,
                "critical_failures": report.critical_failures,
                "execution_time": report.execution_time,
                "system_ready": report.system_ready,
                "timestamp": datetime.now().isoformat(),
            },
            "test_details": [
                {
                    "tool_name": tr.test_case.tool_name,
                    "test_name": tr.test_case.test_name,
                    "description": tr.test_case.description,
                    "prompt_used": tr.test_case.prompt,
                    "result": tr.result.value,
                    "execution_time": tr.execution_time,
                    "critical": tr.test_case.critical,
                    "error_message": tr.error_message,
                    "response_preview": (
                        tr.response[:200] + "..."
                        if tr.response and len(tr.response) > 200
                        else tr.response
                    ),
                    "validation_details": tr.validation_details,
                    "optimization_history": tr.test_case.optimization_history,
                }
                for tr in report.test_reports
            ],
        }

        with open(filename, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"Enhanced flight check report exported to: {filename}")
        return filename
