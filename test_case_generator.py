"""
Auto-generation system for test cases based on available tools
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import color utilities with fallback
try:
    from color_utils import (
        debug_print,
        system_print,
        success_print,
        warning_print,
        error_print,
    )
except ImportError:

    def debug_print(text):
        print(f"[DEBUG] {text}")

    def system_print(text):
        print(f"[SYSTEM] {text}")

    def success_print(text):
        print(f"[SUCCESS] {text}")

    def warning_print(text):
        print(f"[WARNING] {text}")

    def error_print(text):
        print(f"[ERROR] {text}")


class TestCaseGenerator:
    """Generates test cases automatically based on available tools"""

    def __init__(self, config_path: str = "test_cases.json"):
        self.config_path = config_path
        self.default_templates = self._create_default_templates()

    def _create_default_templates(self) -> Dict[str, Dict]:
        """Create default test case templates for different tool types"""
        return {
            # File system tools
            "read_file": {
                "base_test": {
                    "test_name": "config_file_reading",
                    "description": "Test reading configuration files",
                    "prompt": "Read the server_config.json file and show me its contents.",
                    "expected_indicators": ["mcpServers", "config", "json"],
                    "expected_format": "json_content",
                    "timeout_seconds": 10.0,
                    "critical": True,
                    "success_criteria": {
                        "contains_json": True,
                        "contains_keywords": ["mcpServers"],
                    },
                }
            },
            "write_file": {
                "base_test": {
                    "test_name": "file_writing",
                    "description": "Test writing content to files",
                    "prompt": "Write a test message to a temporary file called 'test_output.txt'.",
                    "expected_indicators": ["written", "file", "success"],
                    "expected_format": "confirmation",
                    "timeout_seconds": 10.0,
                    "critical": True,
                    "success_criteria": {
                        "contains_keywords": ["written", "success"],
                        "no_error_keywords": ["error", "failed"],
                    },
                }
            },
            "list_directory": {
                "base_test": {
                    "test_name": "current_directory_listing",
                    "description": "Test listing current directory contents",
                    "prompt": "List all files and directories in the current folder. Show me what's available.",
                    "expected_indicators": ["file", "directory", ".py", ".json"],
                    "expected_format": "directory_listing",
                    "timeout_seconds": 10.0,
                    "critical": True,
                    "success_criteria": {
                        "contains_file_extensions": [".py", ".json"],
                        "min_items": 3,
                    },
                }
            },
            "create_directory": {
                "base_test": {
                    "test_name": "directory_creation",
                    "description": "Test creating new directories",
                    "prompt": "Create a temporary directory called 'test_dir' in the current location.",
                    "expected_indicators": ["created", "directory", "success"],
                    "expected_format": "confirmation",
                    "timeout_seconds": 10.0,
                    "critical": False,
                    "success_criteria": {
                        "contains_keywords": ["created", "directory"],
                        "no_error_keywords": ["error", "failed"],
                    },
                }
            },
            # Research tools
            "search_papers": {
                "setup_test": {
                    "test_name": "setup_search",
                    "description": "Setup test to ensure papers are available for extraction tests",
                    "prompt": "Search for papers about machine learning. Find exactly 1 paper and return the paper ID.",
                    "expected_indicators": ["paper", "machine learning", "results"],
                    "expected_format": "list_of_ids",
                    "timeout_seconds": 45.0,
                    "critical": False,
                    "success_criteria": {
                        "min_response_length": 5,
                        "contains_arxiv_ids": True,
                        "no_error_keywords": ["error", "failed", "exception"],
                    },
                },
                "basic_test": {
                    "test_name": "basic_search",
                    "description": "Test basic paper search functionality",
                    "prompt": "Search for academic papers about artificial intelligence. Return exactly 2 paper IDs.",
                    "expected_indicators": [
                        "paper",
                        "artificial intelligence",
                        "results",
                    ],
                    "expected_format": "list_of_ids",
                    "timeout_seconds": 45.0,
                    "critical": True,
                    "success_criteria": {
                        "min_response_length": 5,
                        "contains_arxiv_ids": True,
                        "no_error_keywords": ["error", "failed", "exception"],
                    },
                },
            },
            "extract_info": {
                "base_test": {
                    "test_name": "info_extraction",
                    "description": "Test paper information extraction",
                    "prompt": "Extract detailed information about a recent paper. Show me the title, authors, and summary.",
                    "expected_indicators": [
                        "title",
                        "authors",
                        "summary",
                        "information",
                    ],
                    "expected_format": "json_or_text",
                    "timeout_seconds": 15.0,
                    "critical": True,
                    "success_criteria": {
                        "min_response_length": 10,
                        "acceptable_not_found": True,
                        "contains_keywords": ["information"],
                    },
                }
            },
            # Network tools
            "fetch": {
                "base_test": {
                    "test_name": "web_content_fetch",
                    "description": "Test fetching web content",
                    "prompt": "Fetch the content from https://example.com and show me what data is returned.",
                    "expected_indicators": ["content", "data", "example"],
                    "expected_format": "web_content",
                    "timeout_seconds": 20.0,
                    "critical": False,
                    "success_criteria": {
                        "min_response_length": 20,
                        "no_error_keywords": ["failed", "connection issue", "error"],
                    },
                }
            },
            "download": {
                "base_test": {
                    "test_name": "file_download",
                    "description": "Test downloading files from URLs",
                    "prompt": "Download a small test file from a reliable source.",
                    "expected_indicators": ["downloaded", "file", "success"],
                    "expected_format": "confirmation",
                    "timeout_seconds": 30.0,
                    "critical": False,
                    "success_criteria": {
                        "contains_keywords": ["downloaded", "success"],
                        "no_error_keywords": ["failed", "error"],
                    },
                }
            },
            # Database/storage tools
            "query": {
                "base_test": {
                    "test_name": "basic_query",
                    "description": "Test basic database querying",
                    "prompt": "Execute a simple query to check database connectivity.",
                    "expected_indicators": ["query", "result", "data"],
                    "expected_format": "query_result",
                    "timeout_seconds": 15.0,
                    "critical": True,
                    "success_criteria": {
                        "min_response_length": 5,
                        "no_error_keywords": ["error", "failed", "connection"],
                    },
                }
            },
            # Generic fallback for unknown tools
            "generic": {
                "base_test": {
                    "test_name": "basic_functionality",
                    "description": "Test basic tool functionality",
                    "prompt": "Test the basic functionality of this tool with safe parameters.",
                    "expected_indicators": ["success", "result", "response"],
                    "expected_format": "text",
                    "timeout_seconds": 30.0,
                    "critical": False,
                    "success_criteria": {
                        "min_response_length": 5,
                        "no_error_keywords": ["error", "failed", "exception"],
                    },
                }
            },
        }

    def generate_missing_test_cases(self, available_tools: List[str]) -> bool:
        """Generate test cases for tools that don't have them"""
        system_print("Checking for missing test cases...")

        # Load or create base config
        config = self._load_or_create_config()

        # Track what we generate
        generated_count = 0
        existing_tools = set(config.get("test_cases", {}).keys())

        for tool_name in available_tools:
            if tool_name not in existing_tools:
                system_print(f"Generating test cases for tool: {tool_name}")
                test_cases = self._generate_test_cases_for_tool(tool_name)
                config["test_cases"][tool_name] = test_cases
                generated_count += 1

        if generated_count > 0:
            # Save updated config
            self._save_config(config)
            success_print(f"Generated test cases for {generated_count} tools")
            return True
        else:
            debug_print("All tools already have test cases")
            return False

    def _load_or_create_config(self) -> Dict:
        """Load existing config or create new one"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                debug_print(f"Loaded existing config from {self.config_path}")
                return config
            except Exception as e:
                warning_print(f"Error loading config: {e}")
                warning_print("Creating new config")

        # Create new config with default structure
        config = {
            "test_cases": {},
            "prompt_templates": {
                "search_papers": {
                    "base_template": "Search for papers about {topic}. {constraints}",
                    "constraints_options": [
                        "Find exactly {max_results} papers and return their IDs.",
                        "Limit results to {max_results} papers.",
                        "Get {max_results} relevant papers on this topic.",
                    ],
                    "optimization_history": [],
                },
                "extract_info": {
                    "base_template": "Get detailed information about {item}. {details}",
                    "details_options": [
                        "Show me the title, authors, and summary.",
                        "Provide comprehensive information including metadata.",
                        "Extract all available details.",
                    ],
                    "optimization_history": [],
                },
                "read_file": {
                    "base_template": "Read the {filename} file and {action}.",
                    "action_options": [
                        "show me its contents",
                        "display the file content",
                        "return the file data",
                    ],
                    "optimization_history": [],
                },
            },
            "dspy_config": {
                "optimization_enabled": True,
                "max_optimization_attempts": 3,
                "success_threshold": 0.8,
                "optimization_metric": "success_rate",
                "prompt_variation_strategies": [
                    "rephrase_instruction",
                    "add_context",
                    "modify_constraints",
                    "change_tone",
                ],
            },
        }

        system_print("Created new test configuration")
        return config

    def _generate_test_cases_for_tool(self, tool_name: str) -> List[Dict]:
        """Generate test cases for a specific tool"""
        # Check if we have specific templates for this tool
        if tool_name in self.default_templates:
            templates = self.default_templates[tool_name]
        else:
            # Use generic template
            templates = self.default_templates["generic"]

        test_cases = []
        for template_name, template in templates.items():
            test_case = template.copy()

            # Add metadata
            test_case["optimization_history"] = []
            test_case["generated_at"] = datetime.now().isoformat()
            test_case["generated_by"] = "auto_generator"

            # Customize prompt based on tool name if using generic template
            if tool_name not in self.default_templates:
                test_case["prompt"] = (
                    f"Test the {tool_name} tool with appropriate parameters. {test_case['prompt']}"
                )
                test_case["test_name"] = f"{tool_name}_functionality"
                test_case["description"] = f"Test {tool_name} tool functionality"

            test_cases.append(test_case)

        debug_print(f"Generated {len(test_cases)} test cases for {tool_name}")
        return test_cases

    def _save_config(self, config: Dict):
        """Save configuration to file"""
        try:
            # Create backup if file exists
            if os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.config_path, backup_path)
                debug_print(f"Created backup: {backup_path}")

            # Save new config
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)

            success_print(f"Saved test configuration to {self.config_path}")

        except Exception as e:
            error_print(f"Error saving config: {e}")
            raise

    def validate_config_schema(self) -> bool:
        """Validate that the config follows the expected schema"""
        if not os.path.exists(self.config_path):
            warning_print("Config file does not exist")
            return False

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            # Check required top-level keys
            required_keys = ["test_cases", "prompt_templates", "dspy_config"]
            for key in required_keys:
                if key not in config:
                    warning_print(f"Missing required key: {key}")
                    return False

            # Validate test cases structure
            for tool_name, test_cases in config["test_cases"].items():
                if not isinstance(test_cases, list):
                    warning_print(f"Test cases for {tool_name} should be a list")
                    return False

                for test_case in test_cases:
                    required_test_keys = [
                        "test_name",
                        "description",
                        "prompt",
                        "expected_indicators",
                        "expected_format",
                        "timeout_seconds",
                        "critical",
                        "success_criteria",
                    ]
                    for key in required_test_keys:
                        if key not in test_case:
                            warning_print(
                                f"Missing required test key '{key}' in {tool_name}.{test_case.get('test_name', 'unknown')}"
                            )
                            return False

            success_print("Config schema validation passed")
            return True

        except Exception as e:
            error_print(f"Error validating config: {e}")
            return False

    def update_existing_test_cases(self, available_tools: List[str]) -> bool:
        """Update existing test cases with missing fields"""
        if not os.path.exists(self.config_path):
            return False

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            updated = False

            for tool_name in config.get("test_cases", {}):
                if tool_name in available_tools:
                    for test_case in config["test_cases"][tool_name]:
                        # Add missing optimization_history if not present
                        if "optimization_history" not in test_case:
                            test_case["optimization_history"] = []
                            updated = True

                        # Ensure all required fields exist
                        defaults = {
                            "expected_format": "text",
                            "timeout_seconds": 30.0,
                            "critical": False,
                            "success_criteria": {"min_response_length": 5},
                        }

                        for key, default_value in defaults.items():
                            if key not in test_case:
                                test_case[key] = default_value
                                updated = True

            if updated:
                self._save_config(config)
                success_print("Updated existing test cases with missing fields")
                return True

        except Exception as e:
            error_print(f"Error updating test cases: {e}")

        return False


def auto_generate_test_cases(
    available_tools: List[str], config_path: str = "test_cases.json"
) -> bool:
    """Main function to auto-generate missing test cases"""
    generator = TestCaseGenerator(config_path)

    # Validate existing config
    if os.path.exists(config_path):
        if not generator.validate_config_schema():
            warning_print("Config validation failed, but continuing...")

    # Update existing test cases with missing fields
    generator.update_existing_test_cases(available_tools)

    # Generate missing test cases
    return generator.generate_missing_test_cases(available_tools)


if __name__ == "__main__":
    # Test the generator
    test_tools = ["read_file", "write_file", "search_papers", "unknown_tool"]
    auto_generate_test_cases(test_tools, "test_generated.json")
