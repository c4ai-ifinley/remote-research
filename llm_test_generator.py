"""
LLM-powered test case generation using DSPy
"""

import dspy
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Import color utilities with fallback
try:
    from color_utils import (
        debug_print,
        system_print,
        success_print,
        warning_print,
        error_print,
        dspy_print,
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

    def dspy_print(text):
        print(f"[DSPy] {text}")


@dataclass
class ToolInfo:
    """Information about a tool for test generation"""

    name: str
    description: str
    parameters: Dict[str, Any]
    output_format: str
    examples: List[str]


class TestCaseGenerationSignature(dspy.Signature):
    """DSPy signature for generating test cases"""

    tool_name = dspy.InputField(desc="The name of the tool to generate tests for")
    tool_description = dspy.InputField(
        desc="Detailed description of what the tool does"
    )
    tool_parameters = dspy.InputField(desc="Parameters the tool accepts (JSON format)")
    output_format = dspy.InputField(desc="Expected output format from the tool")
    schema_requirements = dspy.InputField(
        desc="Required schema and validation criteria for test cases"
    )

    test_cases = dspy.OutputField(
        desc="Generated test cases in JSON format, following the exact schema provided"
    )


class TestCaseGenerator(dspy.Module):
    """DSPy module for generating test cases"""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(TestCaseGenerationSignature)

    def forward(
        self,
        tool_name: str,
        tool_description: str,
        tool_parameters: str,
        output_format: str,
        schema_requirements: str,
    ):
        return self.generate(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters,
            output_format=output_format,
            schema_requirements=schema_requirements,
        )


class LLMTestCaseGenerator:
    """LLM-powered test case generator using DSPy"""

    def __init__(self, config_path: str = "test_cases.json"):
        self.config_path = config_path
        self.generator = None
        self.setup_dspy()
        self.schema_template = self._create_schema_template()

    def setup_dspy(self):
        """Initialize DSPy for test generation"""
        try:
            # Get configuration from environment variables
            openai_api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("BASE_URL")

            if not openai_api_key:
                warning_print(
                    "OPENAI_API_KEY not found - falling back to rule-based generation"
                )
                return

            # Configure DSPy
            lm_config = {
                "model": "openai/o3-mini-birthright",
                "api_key": openai_api_key,
                "max_tokens": 20000,
                "temperature": 1.0,
            }

            if base_url:
                lm_config["base_url"] = base_url

            lm = dspy.LM(**lm_config)
            dspy.configure(lm=lm)

            self.generator = TestCaseGenerator()
            success_print("LLM test generator initialized with DSPy")

        except Exception as e:
            error_print(f"Failed to initialize DSPy for test generation: {e}")
            self.generator = None

    def _create_schema_template(self) -> str:
        """Create schema template for the LLM"""
        return """
REQUIRED TEST CASE SCHEMA:
Each test case must be a JSON object with these exact fields:

{
  "test_name": "string - unique identifier for the test",
  "description": "string - human readable description of what the test does",
  "prompt": "string - natural language instruction to the tool",
  "expected_indicators": ["list of strings - keywords that should appear in successful responses"],
  "expected_format": "string - format type: list_of_ids, json_content, directory_listing, web_content, confirmation, text",
  "timeout_seconds": number - maximum time to wait (10-60 seconds),
  "critical": boolean - true if test failure should block system startup,
  "success_criteria": {
    "min_response_length": number - minimum character count,
    "contains_keywords": ["optional list of required keywords"],
    "contains_arxiv_ids": boolean - true if response should contain arXiv paper IDs,
    "contains_json": boolean - true if response should be valid JSON,
    "acceptable_not_found": boolean - true if "not found" responses are acceptable,
    "no_error_keywords": ["list of keywords that indicate failure"]
  },
  "optimization_history": []
}

EXAMPLES OF GOOD TEST CASES:

For file system tools:
{
  "test_name": "basic_file_read",
  "description": "Test reading a configuration file",
  "prompt": "Read the server_config.json file and display its contents.",
  "expected_indicators": ["config", "json", "content"],
  "expected_format": "json_content",
  "timeout_seconds": 10.0,
  "critical": true,
  "success_criteria": {
    "min_response_length": 20,
    "contains_json": true,
    "no_error_keywords": ["error", "not found", "permission denied"]
  },
  "optimization_history": []
}

For research tools:
{
  "test_name": "paper_search",
  "description": "Test searching for academic papers",
  "prompt": "Search for papers about machine learning. Return exactly 2 paper IDs.",
  "expected_indicators": ["paper", "machine learning", "id"],
  "expected_format": "list_of_ids",
  "timeout_seconds": 45.0,
  "critical": true,
  "success_criteria": {
    "min_response_length": 10,
    "contains_arxiv_ids": true,
    "no_error_keywords": ["error", "failed"]
  },
  "optimization_history": []
}

IMPORTANT: Return ONLY valid JSON array of test case objects. No explanations or markdown.
"""

    def generate_test_cases_for_tool(self, tool_info: ToolInfo) -> List[Dict]:
        """Generate test cases for a specific tool using LLM"""
        if not self.generator:
            dspy_print("DSPy not available, using fallback generation")
            return self._fallback_generation(tool_info)

        try:
            dspy_print(f"Generating test cases for {tool_info.name} using LLM...")

            # Prepare inputs for DSPy
            tool_description = self._create_tool_description(tool_info)
            tool_parameters = json.dumps(tool_info.parameters, indent=2)

            # Generate test cases using DSPy
            result = self.generator(
                tool_name=tool_info.name,
                tool_description=tool_description,
                tool_parameters=tool_parameters,
                output_format=tool_info.output_format,
                schema_requirements=self.schema_template,
            )

            # Parse the generated test cases
            test_cases = self._parse_generated_test_cases(result.test_cases)

            if test_cases:
                success_print(
                    f"Generated {len(test_cases)} test cases for {tool_info.name}"
                )
                return test_cases
            else:
                warning_print(
                    f"LLM generation failed for {tool_info.name}, using fallback"
                )
                return self._fallback_generation(tool_info)

        except Exception as e:
            error_print(f"Error generating test cases for {tool_info.name}: {e}")
            return self._fallback_generation(tool_info)

    def _create_tool_description(self, tool_info: ToolInfo) -> str:
        """Create comprehensive tool description for the LLM"""
        description = f"""
TOOL: {tool_info.name}

DESCRIPTION: {tool_info.description}

PARAMETERS: {json.dumps(tool_info.parameters, indent=2)}

OUTPUT FORMAT: {tool_info.output_format}

EXAMPLES OF USAGE:
"""
        for example in tool_info.examples:
            description += f"- {example}\n"

        description += """
CONTEXT: This tool is part of an MCP (Model Context Protocol) system that needs automated testing.
Test cases should verify that the tool works correctly and returns expected output formats.
Focus on realistic usage scenarios that would occur in production.
"""

        return description.strip()

    def _parse_generated_test_cases(self, test_cases_json: str) -> List[Dict]:
        """Parse and validate generated test cases"""
        try:
            # Clean up the response (remove markdown, extra text)
            cleaned = test_cases_json.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            test_cases = json.loads(cleaned)

            # Ensure it's a list
            if not isinstance(test_cases, list):
                test_cases = [test_cases]

            # Validate each test case
            validated_cases = []
            for test_case in test_cases:
                if self._validate_test_case(test_case):
                    # Add metadata
                    test_case["generated_at"] = datetime.now().isoformat()
                    test_case["generated_by"] = "llm_dspy"
                    test_case["optimization_history"] = test_case.get(
                        "optimization_history", []
                    )
                    validated_cases.append(test_case)
                else:
                    warning_print(f"Invalid test case generated: {test_case}")

            return validated_cases

        except Exception as e:
            error_print(f"Error parsing generated test cases: {e}")
            debug_print(f"Raw response: {test_cases_json}")
            return []

    def _validate_test_case(self, test_case: Dict) -> bool:
        """Validate that a test case follows the required schema"""
        required_fields = [
            "test_name",
            "description",
            "prompt",
            "expected_indicators",
            "expected_format",
            "timeout_seconds",
            "critical",
            "success_criteria",
        ]

        for field in required_fields:
            if field not in test_case:
                return False

        # Type validation
        if not isinstance(test_case["expected_indicators"], list):
            return False
        if not isinstance(test_case["success_criteria"], dict):
            return False
        if not isinstance(test_case["timeout_seconds"], (int, float)):
            return False
        if not isinstance(test_case["critical"], bool):
            return False

        return True

    def _fallback_generation(self, tool_info: ToolInfo) -> List[Dict]:
        """Fallback rule-based generation when LLM is not available"""
        debug_print(f"Using rule-based fallback for {tool_info.name}")

        # Import the original generator for fallback
        from test_case_generator import TestCaseGenerator

        original_generator = TestCaseGenerator(self.config_path)
        return original_generator._generate_test_cases_for_tool(tool_info.name)

    def extract_tool_info_from_mcp(self, chatbot) -> List[ToolInfo]:
        """Extract tool information from MCP chatbot for LLM generation"""
        tool_infos = []

        for tool in chatbot.available_tools:
            tool_name = tool["name"]
            tool_description = tool.get("description", f"Tool: {tool_name}")

            # Extract parameters from input schema
            input_schema = tool.get("input_schema", {})
            parameters = input_schema.get("properties", {})

            # Determine output format and examples based on tool type
            output_format, examples = self._infer_tool_characteristics(
                tool_name, tool_description
            )

            tool_info = ToolInfo(
                name=tool_name,
                description=tool_description,
                parameters=parameters,
                output_format=output_format,
                examples=examples,
            )

            tool_infos.append(tool_info)

        return tool_infos

    def _infer_tool_characteristics(
        self, tool_name: str, description: str
    ) -> tuple[str, List[str]]:
        """Infer output format and examples based on tool name and description"""
        tool_name_lower = tool_name.lower()
        description_lower = description.lower()

        # File operations
        if any(word in tool_name_lower for word in ["read", "file"]):
            return "json_content", [
                "Read the configuration file and show its contents",
                "Display the content of server_config.json",
            ]

        if any(word in tool_name_lower for word in ["list", "directory"]):
            return "directory_listing", [
                "List all files in the current directory",
                "Show directory contents with file types",
            ]

        if any(word in tool_name_lower for word in ["write", "create"]):
            return "confirmation", [
                "Create a new file with test content",
                "Write data to a temporary file",
            ]

        # Research operations
        if any(word in tool_name_lower for word in ["search", "paper"]):
            return "list_of_ids", [
                "Search for papers about machine learning",
                "Find recent papers on artificial intelligence",
            ]

        if any(word in tool_name_lower for word in ["extract", "info"]):
            return "json_or_text", [
                "Extract information about a specific paper",
                "Get details for paper ID 1234.5678",
            ]

        # Network operations
        if any(word in tool_name_lower for word in ["fetch", "download", "http"]):
            return "web_content", [
                "Fetch content from a web URL",
                "Download data from an API endpoint",
            ]

        # Database operations
        if any(word in tool_name_lower for word in ["query", "database", "sql"]):
            return "query_result", [
                "Execute a simple database query",
                "Check database connectivity",
            ]

        # Default
        return "text", [
            f"Test the basic functionality of {tool_name}",
            f"Verify {tool_name} works with standard parameters",
        ]


def llm_generate_test_cases(
    available_tools: List[str], chatbot, config_path: str = "test_cases.json"
) -> bool:
    """Main function to generate test cases using LLM"""
    generator = LLMTestCaseGenerator(config_path)

    if not generator.generator:
        warning_print("LLM generation not available, falling back to rule-based")
        from test_case_generator import auto_generate_test_cases

        return auto_generate_test_cases(available_tools, config_path)

    # Extract tool information from chatbot
    tool_infos = generator.extract_tool_info_from_mcp(chatbot)

    # Load or create config
    config = generator._load_or_create_config()

    generated_count = 0
    existing_tools = set(config.get("test_cases", {}).keys())

    for tool_info in tool_infos:
        if tool_info.name not in existing_tools:
            system_print(f"Generating LLM test cases for: {tool_info.name}")
            test_cases = generator.generate_test_cases_for_tool(tool_info)

            if test_cases:
                config["test_cases"][tool_info.name] = test_cases
                generated_count += 1

    if generated_count > 0:
        # Save config
        generator._save_config(config)
        success_print(f"Generated LLM test cases for {generated_count} tools")
        return True
    else:
        debug_print("All tools already have test cases")
        return False


def _load_or_create_config(self) -> Dict:
    """Load existing config or create new one"""
    if os.path.exists(self.config_path):
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            warning_print(f"Error loading config: {e}")

    # Create new config
    return {
        "test_cases": {},
        "prompt_templates": {},
        "dspy_config": {
            "optimization_enabled": True,
            "max_optimization_attempts": 2,
            "success_threshold": 0.8,
            "optimization_metric": "success_rate",
        },
    }


def _save_config(self, config: Dict):
    """Save configuration to file"""
    try:
        if os.path.exists(self.config_path):
            backup_path = (
                f"{self.config_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.rename(self.config_path, backup_path)

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        success_print(f"Saved LLM-generated test configuration to {self.config_path}")

    except Exception as e:
        error_print(f"Error saving config: {e}")
        raise


# Monkey patch methods to LLMTestCaseGenerator
LLMTestCaseGenerator._load_or_create_config = _load_or_create_config
LLMTestCaseGenerator._save_config = _save_config

if __name__ == "__main__":
    # Test the LLM generator
    generator = LLMTestCaseGenerator("test_llm_generated.json")

    # Mock tool info for testing
    test_tool = ToolInfo(
        name="search_papers",
        description="Searches academic papers on arXiv and returns paper IDs",
        parameters={
            "topic": {"type": "string", "description": "Research topic to search for"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results",
            },
        },
        output_format="list_of_ids",
        examples=[
            "Search for papers about machine learning",
            "Find recent papers on quantum computing",
        ],
    )

    test_cases = generator.generate_test_cases_for_tool(test_tool)
    print(f"Generated {len(test_cases)} test cases")
    for case in test_cases:
        print(f"- {case['test_name']}: {case['description']}")
