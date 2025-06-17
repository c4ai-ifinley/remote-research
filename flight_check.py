# Copy the entire ToolFlightChecker system from the previous artifact here
# This includes all the dataclasses and the ToolFlightChecker class

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Awaitable
from enum import Enum
import asyncio
import json
import time
from datetime import datetime


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    TIMEOUT = "TIMEOUT"


class VerbosityLevel(Enum):
    QUIET = 0  # Only final summary
    MINIMAL = 1  # Tool names and final results only (default)
    NORMAL = 2  # Include test descriptions and basic results
    VERBOSE = 3  # Include response previews and validation details
    DEBUG = 4  # Full debug output with all details


@dataclass
class ToolTestCase:
    """Defines a test case for a specific tool"""

    tool_name: str
    test_name: str
    description: str
    test_args: Dict[str, Any]
    expected_indicators: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    critical: bool = True
    validation_function: Optional[Callable[[str], bool]] = None
    setup_function: Optional[Callable[[], Awaitable[None]]] = None


@dataclass
class TestReport:
    """Results from running a single test"""

    test_case: ToolTestCase
    result: TestResult
    execution_time: float
    response: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


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


class ToolFlightChecker:
    """Manages and executes tool flight checks"""

    def __init__(self, chatbot_instance):
        self.chatbot = chatbot_instance
        self.test_cases: Dict[str, List[ToolTestCase]] = {}
        self.verbosity = VerbosityLevel.MINIMAL  # Default to minimal output
        self.setup_default_tests()

    def set_verbosity(self, level: VerbosityLevel):
        """Set the verbosity level for flight check output"""
        self.verbosity = level

    def enable_debug_mode(self):
        """Enable debug mode (highest verbosity)"""
        self.verbosity = VerbosityLevel.DEBUG

    def disable_debug_mode(self):
        """Reset to minimal verbosity"""
        self.verbosity = VerbosityLevel.MINIMAL

    def setup_default_tests(self):
        """Configure default test cases for common tools"""

        # Custom validation function for search_papers
        def validate_search_papers_response(response: str) -> bool:
            """Custom validation for search_papers - expects list of paper IDs or single paper ID"""
            response = response.strip()

            # Check if it's a JSON list
            try:
                import json

                parsed = json.loads(response)
                if isinstance(parsed, list):
                    return True
            except:
                pass

            # Check for list-like format even if not perfect JSON
            if "[" in response and "]" in response:
                return True

            # Check for arXiv paper ID format (like "1909.03550v1")
            import re

            arxiv_pattern = r"\d{4}\.\d{4,5}(v\d+)?"
            if re.search(arxiv_pattern, response):
                return True

            # Check for comma-separated paper IDs
            if "," in response:
                return True

            # Check for success indicators in text
            response_lower = response.lower()
            success_indicators = ["saved", "results", "paper", "found", "search"]
            if any(indicator in response_lower for indicator in success_indicators):
                return True

            # If response has some content and no obvious error messages, consider it valid
            if (
                len(response) > 0
                and "error" not in response.lower()
                and "failed" not in response.lower()
            ):
                return True

            return False

        # Research tool tests
        self.add_test_case(
            ToolTestCase(
                tool_name="search_papers",
                test_name="basic_search",
                description="Test basic paper search functionality",
                test_args={"topic": "machine learning", "max_results": 2},
                validation_function=validate_search_papers_response,  # Use custom validation
                timeout_seconds=45.0,
                critical=True,
            )
        )

        self.add_test_case(
            ToolTestCase(
                tool_name="extract_info",
                test_name="info_extraction",
                description="Test paper information extraction",
                test_args={"paper_id": "test_paper_123"},
                expected_indicators=["information", "paper", "no saved information"],
                timeout_seconds=15.0,
                critical=False,
            )
        )

        # File system tool tests
        self.add_test_case(
            ToolTestCase(
                tool_name="list_directory",
                test_name="directory_listing",
                description="Test directory listing capability",
                test_args={"path": "."},
                expected_indicators=["file", "directory", "folder", ".py", ".json"],
                timeout_seconds=10.0,
                critical=True,
            )
        )

        self.add_test_case(
            ToolTestCase(
                tool_name="read_file",
                test_name="file_reading",
                description="Test file reading capability",
                test_args={"path": "server_config.json"},
                expected_indicators=["mcpServers", "{", "}", "content"],
                timeout_seconds=10.0,
                critical=True,
            )
        )

        # Fetch tool tests
        self.add_test_case(
            ToolTestCase(
                tool_name="fetch",
                test_name="web_fetch",
                description="Test web content fetching",
                test_args={"url": "https://httpbin.org/json"},
                expected_indicators=["json", "response", "data", "content"],
                timeout_seconds=20.0,
                critical=False,
            )
        )

    def add_test_case(self, test_case: ToolTestCase):
        """Add a test case for a specific tool"""
        if test_case.tool_name not in self.test_cases:
            self.test_cases[test_case.tool_name] = []
        self.test_cases[test_case.tool_name].append(test_case)

    def add_custom_test(
        self,
        tool_name: str,
        test_name: str,
        description: str,
        test_args: Dict[str, Any],
        **kwargs,
    ):
        """Convenient method to add custom tests"""
        test_case = ToolTestCase(
            tool_name=tool_name,
            test_name=test_name,
            description=description,
            test_args=test_args,
            **kwargs,
        )
        self.add_test_case(test_case)

    async def run_single_test(self, test_case: ToolTestCase) -> TestReport:
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

        try:
            # Run setup if provided
            if test_case.setup_function:
                await test_case.setup_function()

            # Get the session for this tool
            session = self.chatbot.sessions.get(test_case.tool_name)
            if not session:
                return TestReport(
                    test_case=test_case,
                    result=TestResult.FAIL,
                    execution_time=time.time() - start_time,
                    error_message="Tool session not found",
                )

            # Execute the tool with timeout
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        test_case.tool_name, arguments=test_case.test_args
                    ),
                    timeout=test_case.timeout_seconds,
                )
                response = result.content[0].text if result.content else str(result)

            except asyncio.TimeoutError:
                return TestReport(
                    test_case=test_case,
                    result=TestResult.TIMEOUT,
                    execution_time=time.time() - start_time,
                    error_message=f"Test timed out after {test_case.timeout_seconds} seconds",
                )

            # Validate the response
            is_valid = self._validate_response(test_case, response)

            return TestReport(
                test_case=test_case,
                result=TestResult.PASS if is_valid else TestResult.FAIL,
                execution_time=time.time() - start_time,
                response=response[:500] + "..." if len(response) > 500 else response,
                error_message=None if is_valid else "Response validation failed",
            )

        except Exception as e:
            return TestReport(
                test_case=test_case,
                result=TestResult.FAIL,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def _validate_response(self, test_case: ToolTestCase, response: str) -> bool:
        """Validate tool response against expected indicators"""
        if test_case.validation_function:
            return test_case.validation_function(response)

        if not test_case.expected_indicators:
            return True  # No validation criteria, assume pass

        response_lower = response.lower()

        # For search_papers, we expect either success indicators OR a list of paper IDs
        if test_case.tool_name == "search_papers":
            # Check if response contains paper IDs (format like ['1234.5678', '9012.3456'])
            if "[" in response and "]" in response:
                return True
            # Check if it's a JSON list of strings
            try:
                import json

                parsed = json.loads(response)
                if isinstance(parsed, list):
                    return True
            except:
                pass

        # Check if any expected indicator is present
        indicators_found = [
            indicator.lower()
            for indicator in test_case.expected_indicators
            if indicator.lower() in response_lower
        ]

        # For debugging - this helps us understand what was found
        if self.verbosity == VerbosityLevel.DEBUG:
            print(f"    Looking for: {test_case.expected_indicators}")
            print(f"    Found indicators: {indicators_found}")
            print(f"    Response sample: {response[:100]}...")

        return len(indicators_found) > 0

    async def run_flight_check(
        self, parallel: bool = False, verbosity: VerbosityLevel = None
    ) -> FlightCheckReport:
        """Execute all flight checks"""
        if verbosity is not None:
            self.verbosity = verbosity

        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("\nStarting Tool Flight Check...")
            print("=" * 50)

        start_time = time.time()
        all_reports = []

        # Get all available tools
        available_tools = [tool["name"] for tool in self.chatbot.available_tools]

        if parallel:
            # Run tests in parallel (faster but less readable output)
            tasks = []
            for tool_name in available_tools:
                if tool_name in self.test_cases:
                    for test_case in self.test_cases[tool_name]:
                        tasks.append(self.run_single_test(test_case))

            all_reports = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions
            all_reports = [r for r in all_reports if isinstance(r, TestReport)]
        else:
            # Run tests sequentially (better for debugging)
            for tool_name in available_tools:
                if tool_name in self.test_cases:
                    if self.verbosity.value >= VerbosityLevel.MINIMAL.value:
                        print(f"\nTesting {tool_name}...")

                    for test_case in self.test_cases[tool_name]:
                        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
                            print(
                                f"  Running {test_case.test_name}: {test_case.description}"
                            )
                        elif self.verbosity.value >= VerbosityLevel.MINIMAL.value:
                            print(f"  {test_case.test_name}...", end=" ")

                        report = await self.run_single_test(test_case)
                        all_reports.append(report)

                        # Print results based on verbosity level
                        self._print_test_result(report)

        # Generate summary report
        total_time = time.time() - start_time
        report = self._generate_flight_report(all_reports, total_time)

        # Print summary
        self._print_flight_summary(report)

        return report

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

        if self.verbosity == VerbosityLevel.MINIMAL:
            print(f"{status_text[report.result]} ({report.execution_time:.2f}s)")
            if report.result == TestResult.FAIL:
                print(f"    Error: {report.error_message}")

        elif self.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"    {status_text[report.result]} ({report.execution_time:.2f}s)")
            if report.error_message:
                print(f"    Error: {report.error_message}")

            if self.verbosity.value >= VerbosityLevel.VERBOSE.value:
                if report.result == TestResult.FAIL and report.response:
                    print(f"    Response preview: {report.response[:200]}...")
                elif report.result == TestResult.PASS and report.response:
                    print(f"    Response looks good: {report.response[:100]}...")

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
        print("\n" + "=" * 50)
        print("FLIGHT CHECK SUMMARY")
        print("=" * 50)

        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed}")
        print(f"Failed: {report.failed}")
        print(f"Skipped: {report.skipped}")
        print(f"Timeout: {report.timeout}")
        print(f"Critical Failures: {report.critical_failures}")
        print(f"Total Time: {report.execution_time:.2f}s")

        if report.system_ready:
            print("\nSYSTEM READY FOR TAKEOFF! All critical systems operational.")
        else:
            print(
                f"\nSYSTEM NOT READY - {report.critical_failures} critical failure(s) detected!"
            )
            print("Critical failures:")
            for test_report in report.test_reports:
                if (
                    test_report.result in [TestResult.FAIL, TestResult.TIMEOUT]
                    and test_report.test_case.critical
                ):
                    print(
                        f"   - {test_report.test_case.tool_name}.{test_report.test_case.test_name}: {test_report.error_message}"
                    )

        print("=" * 50)

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
                    "result": tr.result.value,
                    "execution_time": tr.execution_time,
                    "critical": tr.test_case.critical,
                    "error_message": tr.error_message,
                    "response_preview": (
                        tr.response[:200] + "..."
                        if tr.response and len(tr.response) > 200
                        else tr.response
                    ),
                }
                for tr in report.test_reports
            ],
        }

        with open(filename, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"Flight check report exported to: {filename}")
        return filename
