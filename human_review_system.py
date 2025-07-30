"""
Human Review System for Failed Test Cases
Allows easy human intervention to fix failing tests.

Save this as: human_review_system.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from utils import atomic_write_json
from color_utils import (
    system_print,
    success_print,
    warning_print,
    error_print,
    debug_print,
    colored_print,
    Colors,
)


class HumanReviewSystem:
    """System for collecting failed tests and allowing human review/fixes"""

    def __init__(self, review_file: str = "failed_tests_review.json"):
        self.review_file = review_file
        self.applied_fixes_file = "applied_fixes_log.json"

    def collect_failed_tests(self, test_reports: List) -> Dict[str, Any]:
        """Collect failed critical tests for human review"""
        failed_tests = []

        for report in test_reports:
            # Only collect critical failures that humans can reasonably fix
            if (
                report.result.value in ["FAIL", "TIMEOUT"]
                and report.test_case.critical
                and report.test_case.auto_generated
            ):

                failed_test = {
                    "tool_name": report.test_case.tool_name,
                    "test_name": report.test_case.test_name,
                    "test_type": getattr(report.test_case, "test_type", "basic"),
                    "current_prompt": report.test_case.prompt,
                    "suggested_prompt": "",  # Human fills this in
                    "failure_reason": report.error_message,
                    "response_preview": (
                        report.response[:200] if report.response else "No response"
                    ),
                    "tool_description": report.test_case.tool_schema.get(
                        "description", "No description"
                    ),
                    "status": "needs_review",  # "needs_review", "fixed", "skip"
                    "timestamp": datetime.now().isoformat(),
                }
                failed_tests.append(failed_test)

        return {
            "review_instructions": self._get_review_instructions(),
            "failed_tests": failed_tests,
            "created_at": datetime.now().isoformat(),
            "total_failures": len(failed_tests),
        }

    def _analyze_failure(self, report) -> Dict[str, Any]:
        """Analyze why a test failed and categorize the issue"""
        analysis = {"category": "unknown", "likely_cause": "", "fix_suggestion": ""}

        error_msg = (report.error_message or "").lower()
        response = (report.response or "").lower()
        tool_name = report.test_case.tool_name

        # File/path related issues
        if any(
            keyword in error_msg
            for keyword in ["no such file", "enoent", "file not found"]
        ):
            analysis.update(
                {
                    "category": "file_not_found",
                    "likely_cause": "Test is trying to access a file that doesn't exist",
                    "fix_suggestion": f"Change prompt to use existing files or ask the tool to handle missing files gracefully",
                }
            )

        # Directory issues
        elif any(
            keyword in error_msg for keyword in ["directory", "folder"]
        ) and tool_name in ["create_directory", "list_directory"]:
            analysis.update(
                {
                    "category": "directory_issue",
                    "likely_cause": "Directory operation failed - might be permissions or invalid path",
                    "fix_suggestion": "Use current directory '.' or ask tool to handle directories safely",
                }
            )

        # JSON expectation failures
        elif "json" in error_msg and "not found" in error_msg:
            analysis.update(
                {
                    "category": "json_format",
                    "likely_cause": "Expected JSON response but got plain text",
                    "fix_suggestion": "Either accept non-JSON responses or ask tool to return JSON format",
                }
            )

        # Network/fetch failures
        elif tool_name == "fetch" and any(
            keyword in error_msg for keyword in ["failed", "error"]
        ):
            analysis.update(
                {
                    "category": "network_issue",
                    "likely_cause": "URL fetch failed - might be invalid URL or network issue",
                    "fix_suggestion": "Use a reliable test URL like 'https://httpbin.org/json'",
                }
            )

        # Move file issues
        elif tool_name == "move_file":
            analysis.update(
                {
                    "category": "file_operation",
                    "likely_cause": "Cannot move files that don't exist",
                    "fix_suggestion": "Create test file first or use existing file",
                }
            )

        # Search issues
        elif "search" in tool_name:
            analysis.update(
                {
                    "category": "search_parameters",
                    "likely_cause": "Search parameters might be invalid",
                    "fix_suggestion": "Use simpler search terms or current directory",
                }
            )

        # Extract info issues (research tools)
        elif "extract_info" in tool_name:
            analysis.update(
                {
                    "category": "data_not_found",
                    "likely_cause": "Trying to extract info for non-existent paper",
                    "fix_suggestion": "Search for papers first, then extract existing paper info",
                }
            )

        # Error keyword issues
        elif "error keywords found" in error_msg:
            analysis.update(
                {
                    "category": "error_keywords",
                    "likely_cause": "Response contains words like 'error' which are flagged as failures",
                    "fix_suggestion": "Update prompt to request successful operation or graceful error handling",
                }
            )

        return analysis

    def _get_review_instructions(self) -> str:
        """Get general instructions for the review process"""
        return """
=== HUMAN REVIEW INSTRUCTIONS ===

This file contains failed test cases that need human review. For each failed test:

1. READ the failure_analysis and human_instructions
2. UPDATE the "suggested_prompt" field with a better prompt
3. CHANGE "status" from "needs_review" to "fixed" 
4. OPTIONALLY change status to "skip" if the test should be disabled

After editing this file, run the flight checker again. It will automatically apply your fixes.

TIPS:
- Make prompts more general and less specific
- Focus on tool capabilities rather than specific files/data
- Ask tools to handle missing data gracefully
- Use current directory "." for file operations
- Request successful demonstrations rather than assuming success

EXAMPLE EDIT:
Change: "status": "needs_review"
To:     "status": "fixed"

And fill in: "suggested_prompt": "Your improved prompt here"

COMMON FIXES:
- File tools: "Demonstrate file reading with current directory files"
- Network tools: "Fetch content from https://httpbin.org/json"
- Search tools: "Search for Python files in current directory"  
- Data tools: "Extract available information or show not found message"
        """.strip()

    def save_failed_tests_for_review(self, test_reports: List) -> bool:
        """Save failed tests to review file if there are any failures"""
        review_data = self.collect_failed_tests(test_reports)

        if not review_data["failed_tests"]:
            debug_print("No failed tests to save for review")
            return False

        atomic_write_json(review_data, self.review_file)

        warning_print(f"\n{'='*60}")
        warning_print("FAILED TESTS DETECTED - HUMAN REVIEW NEEDED")
        warning_print(f"{'='*60}")
        error_print(f"Found {len(review_data['failed_tests'])} failed critical tests")
        warning_print(f"Review file created: {self.review_file}")
        warning_print("\nTo fix these failures:")
        warning_print(f"1. Open {self.review_file}")
        warning_print("2. Read the instructions and failure analysis")
        warning_print("3. Update 'suggested_prompt' for each test")
        warning_print("4. Change 'status' from 'needs_review' to 'fixed'")
        warning_print("5. Run flight check again")
        warning_print(f"{'='*60}")

        return True

    def apply_human_fixes(self, flight_checker) -> int:
        """Apply human fixes from review file to test cases"""
        if not os.path.exists(self.review_file):
            return 0

        try:
            with open(self.review_file, "r") as f:
                review_data = json.load(f)
        except Exception as e:
            error_print(f"Error reading review file: {e}")
            return 0

        fixes_applied = 0
        applied_fixes = []

        for failed_test in review_data.get("failed_tests", []):
            if failed_test["status"] == "fixed" and failed_test["suggested_prompt"]:
                # Find the corresponding test case and update it
                tool_name = failed_test["tool_name"]
                test_name = failed_test["test_name"]
                new_prompt = failed_test["suggested_prompt"]

                if tool_name in flight_checker.test_cases:
                    for test_case in flight_checker.test_cases[tool_name]:
                        if test_case.test_name == test_name:
                            old_prompt = test_case.prompt
                            test_case.prompt = new_prompt

                            # Add to optimization history
                            fix_record = {
                                "timestamp": datetime.now().isoformat(),
                                "original_prompt": old_prompt,
                                "optimized_prompt": new_prompt,
                                "failure_context": failed_test["failure_reason"],
                                "strategy": "human_review",
                            }
                            test_case.optimization_history.append(fix_record)

                            # Log the applied fix
                            applied_fixes.append(
                                {
                                    "tool_name": tool_name,
                                    "test_name": test_name,
                                    "old_prompt": old_prompt,
                                    "new_prompt": new_prompt,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )

                            fixes_applied += 1
                            success_print(f"Applied fix for {tool_name}.{test_name}")
                            break
            elif failed_test["status"] == "skip":
                # Mark test as non-critical so it won't cause system failure
                tool_name = failed_test["tool_name"]
                test_name = failed_test["test_name"]

                if tool_name in flight_checker.test_cases:
                    for test_case in flight_checker.test_cases[tool_name]:
                        if test_case.test_name == test_name:
                            test_case.critical = False
                            debug_print(
                                f"Marked {tool_name}.{test_name} as non-critical"
                            )
                            fixes_applied += 1
                            break

        if fixes_applied > 0:
            # Save updated test cases
            flight_checker._save_test_cases()

            # Log applied fixes
            self._log_applied_fixes(applied_fixes)

            # Move review file to archive
            self._archive_review_file()

            success_print(f"Applied {fixes_applied} human fixes to test cases")

        return fixes_applied

    def _log_applied_fixes(self, applied_fixes: List[Dict]):
        """Log applied fixes for future reference"""
        if not applied_fixes:
            return

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "fixes_applied": applied_fixes,
            "total_fixes": len(applied_fixes),
        }

        # Append to existing log or create new one
        if os.path.exists(self.applied_fixes_file):
            try:
                with open(self.applied_fixes_file, "r") as f:
                    existing_log = json.load(f)
                    if not isinstance(existing_log, list):
                        existing_log = [existing_log]
            except:
                existing_log = []
        else:
            existing_log = []

        existing_log.append(log_data)
        atomic_write_json(existing_log, self.applied_fixes_file)

    def _archive_review_file(self):
        """Archive the review file after processing"""
        if os.path.exists(self.review_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"failed_tests_review_{timestamp}.json"
            try:
                os.rename(self.review_file, archive_name)
                debug_print(f"Archived review file as {archive_name}")
            except Exception as e:
                debug_print(f"Could not archive review file: {e}")


# Example usage and testing
if __name__ == "__main__":
    # This allows you to test the human review system independently
    print("Human Review System loaded successfully!")
    print("This module should be imported by flight_checker.py")
