"""
DSPy-based prompt optimization for tool flight checks
"""

import dspy
from typing import Dict, List, Any, Optional
import json
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime

# Import color utilities with fallback
try:
    from color_utils import (
        debug_print,
        dspy_print,
        error_print,
        success_print,
        warning_print,
    )
except ImportError:
    # Fallback to regular print if colors not available
    def debug_print(text):
        print(f"[DEBUG] {text}")

    def dspy_print(text):
        print(f"[DSPy] {text}")

    def error_print(text):
        print(f"[ERROR] {text}")

    def success_print(text):
        print(f"[SUCCESS] {text}")

    def warning_print(text):
        print(f"[WARNING] {text}")


@dataclass
class OptimizationContext:
    """Context for prompt optimization"""

    tool_name: str
    original_prompt: str
    failure_reason: str
    expected_output_format: str
    success_criteria: Dict[str, Any]
    previous_attempts: List[str]
    tool_arguments: Dict[str, Any] = None


class ToolPromptSignature(dspy.Signature):
    """Signature for tool prompt optimization"""

    tool_name = dspy.InputField(desc="The name of the tool being optimized")
    tool_description = dspy.InputField(
        desc="Detailed description of what the tool does and how it works"
    )
    tool_arguments = dspy.InputField(
        desc="The specific arguments/parameters that will be passed to the tool"
    )
    failure_info = dspy.InputField(desc="Details about why the current prompt failed")
    original_prompt = dspy.InputField(desc="The original prompt that failed")
    optimized_prompt = dspy.OutputField(
        desc="A completely new, specific prompt that clearly instructs the tool what to do. Should be a complete replacement, not an addition to the original."
    )


class PromptOptimizer(dspy.Module):
    """DSPy module for optimizing tool prompts"""

    def __init__(self):
        super().__init__()
        self.optimize = dspy.ChainOfThought(ToolPromptSignature)

    def forward(
        self,
        tool_name: str,
        tool_description: str,
        tool_arguments: str,
        failure_info: str,
        original_prompt: str,
    ):
        return self.optimize(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_arguments=tool_arguments,
            failure_info=failure_info,
            original_prompt=original_prompt,
        )


class ToolCallSignature(dspy.Signature):
    """Signature for generating tool calls from natural language"""

    tool_description = dspy.InputField(desc="Description of what the tool does")
    user_intent = dspy.InputField(desc="What the user wants to accomplish")
    tool_call = dspy.OutputField(desc="A clear, specific instruction to the tool")


class ToolCallGenerator(dspy.Module):
    """DSPy module for generating tool calls from user intent"""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ToolCallSignature)

    def forward(self, tool_description: str, user_intent: str):
        return self.generate(tool_description=tool_description, user_intent=user_intent)


class DSPyFlightOptimizer:
    """Advanced DSPy-based flight check optimizer"""

    def __init__(self, anthropic_client=None):
        self.anthropic_client = anthropic_client
        self.optimizer = None
        self.call_generator = None
        self.setup_dspy()

    def setup_dspy(self):
        """Initialize DSPy with OpenAI o3-mini-birthright model"""
        dspy_print("Setting up DSPy optimizer...")

        try:
            # Get configuration from environment variables
            openai_api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("BASE_URL")

            debug_print(f"OpenAI API Key found: {'Yes' if openai_api_key else 'No'}")
            debug_print(f"Base URL: {base_url}")

            if not openai_api_key:
                warning_print("OPENAI_API_KEY not found in environment variables")
                warning_print(
                    "DSPy will not be available - falling back to rule-based optimization"
                )
                return

            # Try importing dspy first
            try:
                import dspy

                debug_print("DSPy imported successfully")
            except ImportError as e:
                error_print(f"Failed to import DSPy: {e}")
                return

            # Configure DSPy with o3-mini-birthright model
            lm_config = {
                "model": "openai/o3-mini-birthright",
                "api_key": openai_api_key,
                "max_tokens": 20000,
                "temperature": 1.0,
            }

            if base_url:
                lm_config["base_url"] = base_url
                debug_print(f"Using custom base URL: {base_url}")

            debug_print(f"Initializing DSPy with config: {lm_config}")

            # Initialize the language model
            lm = dspy.LM(**lm_config)
            dspy.configure(lm=lm)

            debug_print("DSPy LM configured successfully")

            # Initialize DSPy modules
            self.optimizer = PromptOptimizer()
            self.call_generator = ToolCallGenerator()

            debug_print("DSPy modules initialized successfully")
            success_print("DSPy setup complete!")

        except Exception as e:
            error_print(f"Error during DSPy setup: {e}")
            debug_print(f"Error type: {type(e).__name__}")
            import traceback

            debug_print(f"Full traceback:\n{traceback.format_exc()}")
            warning_print("Falling back to rule-based optimization")
            self.optimizer = None
            self.call_generator = None

    def optimize_prompt(self, context: OptimizationContext) -> str:
        """Optimize a prompt using DSPy"""
        if not self.optimizer:
            debug_print("DSPy optimizer not available, using fallback")
            return self._fallback_optimization(context)

        try:
            # Get detailed tool description
            tool_description = self._get_tool_description(context.tool_name)

            # Prepare tool arguments string
            tool_arguments = (
                json.dumps(context.tool_arguments, indent=2)
                if context.tool_arguments
                else "No arguments provided"
            )

            # Prepare failure info
            failure_info = self._prepare_failure_info(context)

            dspy_print(f"Calling DSPy optimizer with:")
            debug_print(f"  Tool: {context.tool_name}")
            debug_print(f"  Arguments: {tool_arguments}")
            debug_print(f"  Original prompt: '{context.original_prompt}'")

            # Use DSPy to optimize the prompt
            result = self.optimizer(
                tool_name=context.tool_name,
                tool_description=tool_description,
                tool_arguments=tool_arguments,
                failure_info=failure_info,
                original_prompt=context.original_prompt,
            )

            optimized_prompt = result.optimized_prompt.strip()
            success_print(f"  DSPy result: '{optimized_prompt}'")

            # Validate the optimized prompt
            if self._validate_optimized_prompt(optimized_prompt, context):
                return optimized_prompt
            else:
                warning_print("DSPy result failed validation, using fallback")
                return self._fallback_optimization(context)

        except Exception as e:
            error_print(f"DSPy optimization failed with error: {e}")
            return self._fallback_optimization(context)

    def generate_tool_call(
        self, tool_name: str, tool_description: str, user_intent: str
    ) -> str:
        """Generate a tool call from user intent"""
        if not self.call_generator:
            return self._fallback_call_generation(tool_name, user_intent)

        try:
            result = self.call_generator(
                tool_description=tool_description, user_intent=user_intent
            )
            return result.tool_call
        except Exception as e:
            error_print(f"DSPy call generation failed: {e}")
            return self._fallback_call_generation(tool_name, user_intent)

    def _get_tool_description(self, tool_name: str) -> str:
        """Get detailed description of the tool for optimization context"""
        tool_descriptions = {
            "search_papers": """
Tool Function: Searches academic papers on arXiv based on a topic query.
Input Parameters: 
- topic (string): The research topic or subject area to search for
- max_results (int): Maximum number of papers to return (default: 5)
Output Format: Returns a list of arXiv paper IDs (e.g., ["1909.03550v1", "2001.12345v2"])
Usage Notes: The tool searches arXiv's database and saves paper metadata locally. It expects specific topic keywords and returns paper identifiers that can be used with extract_info tool.
""",
            "extract_info": """
Tool Function: Extracts detailed information about a specific academic paper.
Input Parameters:
- paper_id (string): The arXiv paper ID to look up (e.g., "1909.03550v1")
Output Format: Returns JSON or structured text with paper details including title, authors, summary, publication date, and PDF URL.
Usage Notes: Searches local database of previously downloaded papers. If paper not found locally, returns a "no saved information" message. Requires exact paper ID format.
""",
            "read_file": """
Tool Function: Reads and returns the contents of a specified file.
Input Parameters:
- path (string): File path to read (e.g., "server_config.json", "data.txt")
Output Format: Returns the complete file contents as text.
Usage Notes: Can read any accessible file. Common use cases include configuration files, data files, logs. Returns raw file content without modification.
""",
            "list_directory": """
Tool Function: Lists all files and directories in a specified path.
Input Parameters:
- path (string): Directory path to list (e.g., ".", "/home/user", "data/")
Output Format: Returns formatted list showing [FILE] or [DIR] prefix followed by item names.
Usage Notes: Provides directory navigation and file discovery. Shows both files and subdirectories with clear type indicators.
""",
            "fetch": """
Tool Function: Fetches content from a web URL and returns the data.
Input Parameters:
- url (string): The web URL to fetch content from (e.g., "https://example.com/api/data")
Output Format: Returns the fetched content, often with content-type information and formatted display.
Usage Notes: Can fetch web pages, API responses, JSON data. Handles various content types and provides appropriate formatting for display.
""",
        }

        return tool_descriptions.get(
            tool_name, f"Tool: {tool_name} - No detailed description available."
        )

    def _prepare_failure_info(self, context: OptimizationContext) -> str:
        """Prepare failure information for DSPy"""
        failure_info = f"""
FAILURE ANALYSIS:
Original Prompt: "{context.original_prompt}"
Failure Reason: {context.failure_reason}
Tool Arguments Being Used: {json.dumps(context.tool_arguments, indent=2) if context.tool_arguments else "None"}

PROBLEMS IDENTIFIED:
1. The prompt is too vague and doesn't provide clear instructions
2. The prompt doesn't specify what output format is expected
3. The prompt doesn't give the tool enough context about what to do

REQUIREMENTS FOR NEW PROMPT:
- Must be completely different from the original
- Must be specific and actionable
- Must clearly state what the tool should do
- Must specify expected output format
- Must work with the provided tool arguments

EXAMPLE OF GOOD PROMPTS:
- For search_papers: "Search for academic papers about [topic]. Return exactly [number] arXiv paper IDs."
- For extract_info: "Extract detailed information about paper with ID '[paper_id]'. Show title, authors, and summary."
- For read_file: "Read the file '[filename]' and display its complete contents."
"""

        if context.previous_attempts:
            failure_info += f"\n\nPREVIOUS FAILED ATTEMPTS (DO NOT REPEAT):\n"
            for i, attempt in enumerate(context.previous_attempts[-3:], 1):
                failure_info += f"{i}. '{attempt}'\n"
            failure_info += "\nThe new prompt must be significantly different from all previous attempts."

        return failure_info.strip()

    def _validate_optimized_prompt(
        self, optimized_prompt: str, context: OptimizationContext
    ) -> bool:
        """Validate that the optimized prompt is reasonable"""
        if not optimized_prompt or len(optimized_prompt.strip()) < 10:
            return False

        if optimized_prompt.strip() == context.original_prompt.strip():
            return False

        for prev_attempt in context.previous_attempts:
            if optimized_prompt.strip() == prev_attempt.strip():
                return False

        if context.tool_name == "search_papers":
            required_elements = ["search", "papers"]
            if not any(elem in optimized_prompt.lower() for elem in required_elements):
                return False

        return True

    def _fallback_optimization(self, context: OptimizationContext) -> str:
        """Fallback rule-based optimization when DSPy is not available"""
        original = context.original_prompt
        tool_name = context.tool_name

        # Handle extremely vague prompts by replacing them entirely
        vague_indicators = ["stuff", "something", "show me", "find", "get"]
        if (
            any(indicator in original.lower() for indicator in vague_indicators)
            and len(original.split()) <= 4
        ):

            specific_prompts = {
                "search_papers": "Search for academic papers about machine learning. Return exactly 2 paper IDs from arXiv.",
                "extract_info": "Extract detailed information about paper with ID 'test_paper_123'. Show the title, authors, summary, and publication details.",
                "read_file": "Read the server_config.json file and display its complete contents.",
                "list_directory": "List all files and directories in the current folder. Show the complete directory structure.",
                "fetch": "Fetch content from https://example.com and display the returned content.",
            }

            if tool_name in specific_prompts:
                return specific_prompts[tool_name]

        # For less vague prompts, apply targeted improvements
        if "validation" in context.failure_reason.lower():
            if tool_name == "search_papers":
                return f"{original}. Search for academic papers and return a list of arXiv paper IDs."
            elif tool_name == "extract_info":
                return f"{original}. Extract paper information including title, authors, and summary. If no paper is found, clearly state that no information is available."

        # Add format specification based on expected format
        format_additions = {
            "list_of_ids": " Return the results as a list of paper IDs.",
            "json_or_text": " Provide the information in a clear, structured format.",
            "json_content": " Display the JSON content clearly.",
            "directory_listing": " Show a formatted list of all files and directories.",
            "web_content": " Return the fetched content in a readable format.",
        }

        expected_format = context.expected_output_format
        if expected_format in format_additions:
            return original + format_additions[expected_format]

        return (
            original
            + f" Please provide a clear, specific response that meets the requirements for the {tool_name} tool."
        )

    def _fallback_call_generation(self, tool_name: str, user_intent: str) -> str:
        """Fallback call generation when DSPy is not available"""
        templates = {
            "search_papers": f"Search for papers about {user_intent}. Return the paper IDs.",
            "extract_info": f"Extract information about the paper related to {user_intent}.",
            "read_file": f"Read the file that contains information about {user_intent}.",
            "list_directory": f"List the directory contents to find files related to {user_intent}.",
            "fetch": f"Fetch web content related to {user_intent}.",
        }

        return templates.get(
            tool_name, f"Use the {tool_name} tool to help with: {user_intent}"
        )

    def test_dspy_connection(self):
        """Test if DSPy is properly configured and can make a call"""
        if not self.optimizer:
            warning_print("DSPy optimizer not initialized")
            return False

        try:
            dspy_print("Testing DSPy connection...")

            result = self.optimizer(
                tool_name="test_tool",
                tool_description="A test tool for verification that returns simple responses",
                tool_arguments='{"test_param": "test_value"}',
                failure_info="This is a test to verify DSPy is working correctly",
                original_prompt="test prompt that needs improvement",
            )

            success_print(
                f"DSPy test successful! Optimized prompt: '{result.optimized_prompt}'"
            )
            return True

        except Exception as e:
            error_print(f"DSPy test failed: {e}")
            import traceback

            debug_print(f"Full traceback:\n{traceback.format_exc()}")
            return False


def test_dspy_optimizer():
    """Test the DSPy optimizer functionality"""
    optimizer = DSPyFlightOptimizer()

    context = OptimizationContext(
        tool_name="search_papers",
        original_prompt="Find papers about machine learning",
        failure_reason="Response validation failed - no paper IDs found",
        expected_output_format="list_of_ids",
        success_criteria={"contains_arxiv_ids": True, "min_response_length": 5},
        previous_attempts=[],
        tool_arguments={"topic": "machine learning", "max_results": 2},
    )

    optimized = optimizer.optimize_prompt(context)
    print(f"Original: {context.original_prompt}")
    print(f"Optimized: {optimized}")


if __name__ == "__main__":
    test_dspy_optimizer()
