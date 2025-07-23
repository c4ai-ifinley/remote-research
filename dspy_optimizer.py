"""
DSPy-based prompt optimization for MCP tools
Uses only MCP schema information, no tool-specific knowledge.
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
    tool_schema: Dict[str, Any]
    original_prompt: str
    failure_reason: str
    generated_arguments: Dict[str, Any]
    success_criteria: Dict[str, Any]
    previous_attempts: List[str]


class ToolPromptSignature(dspy.Signature):
    """Signature for MCP tool prompt optimization"""

    tool_name = dspy.InputField(desc="The name of the MCP tool being optimized")
    tool_schema = dspy.InputField(
        desc="The complete MCP schema for this tool including input parameters and description"
    )
    current_arguments = dspy.InputField(
        desc="The specific arguments that will be passed to the tool when testing"
    )
    failure_details = dspy.InputField(
        desc="Detailed information about why the current prompt failed, including any error messages"
    )
    original_prompt = dspy.InputField(
        desc="The original prompt that failed to work properly"
    )
    success_requirements = dspy.InputField(
        desc="What constitutes a successful response for this tool"
    )
    optimized_prompt = dspy.OutputField(
        desc="A new, specific prompt that clearly instructs what the tool should accomplish. Must be completely different from the original and work with the provided arguments."
    )


class PromptOptimizer(dspy.Module):
    """DSPy module for optimizing MCP tool prompts"""

    def __init__(self):
        super().__init__()
        self.optimize = dspy.ChainOfThought(ToolPromptSignature)

    def forward(
        self,
        tool_name: str,
        tool_schema: str,
        current_arguments: str,
        failure_details: str,
        original_prompt: str,
        success_requirements: str,
    ):
        return self.optimize(
            tool_name=tool_name,
            tool_schema=tool_schema,
            current_arguments=current_arguments,
            failure_details=failure_details,
            original_prompt=original_prompt,
            success_requirements=success_requirements,
        )


class ToolCallSignature(dspy.Signature):
    """Signature for generating tool calls from user intent"""

    tool_schema = dspy.InputField(
        desc="The MCP schema describing what this tool does and its parameters"
    )
    user_intent = dspy.InputField(
        desc="What the user wants to accomplish with this tool"
    )
    available_context = dspy.InputField(
        desc="Any available context like file names, IDs, or other relevant information"
    )
    tool_instruction = dspy.OutputField(
        desc="A clear, specific instruction that tells the tool exactly what to do"
    )


class ToolCallGenerator(dspy.Module):
    """DSPy module for generating tool calls from user intent"""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ToolCallSignature)

    def forward(self, tool_schema: str, user_intent: str, available_context: str = ""):
        return self.generate(
            tool_schema=tool_schema,
            user_intent=user_intent,
            available_context=available_context,
        )


class DSPyOptimizer:
    """DSPy-based optimizer for any MCP tool"""

    def __init__(self, config_path: str = "test_cases.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.optimization_enabled = self.config.get("dspy_config", {}).get(
            "optimization_enabled", True
        )
        self.optimizer = None
        self.call_generator = None
        self.setup_dspy()

    def load_config(self) -> Dict:
        """Load test configuration from JSON"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    content = f.read().strip()
                    if not content:
                        # File exists but is empty
                        print(
                            f"Warning: {self.config_path} is empty. Using default configuration."
                        )
                        return self._get_default_config()

                    config = json.loads(content)

                    # Ensure config has expected structure
                    if not isinstance(config, dict):
                        print(
                            f"Warning: Invalid config format in {self.config_path}. Using default configuration."
                        )
                        return self._get_default_config()

                    # Add missing keys with defaults
                    default_config = self._get_default_config()
                    for key, default_value in default_config.items():
                        if key not in config:
                            config[key] = default_value
                        elif key == "dspy_config" and not isinstance(config[key], dict):
                            config[key] = default_value

                    return config
            else:
                print(
                    f"Warning: {self.config_path} not found. Using default configuration."
                )
                return self._get_default_config()

        except json.JSONDecodeError as e:
            print(
                f"Warning: JSON decode error in {self.config_path}: {e}. Using default configuration."
            )
            return self._get_default_config()
        except Exception as e:
            print(
                f"Warning: Error loading config from {self.config_path}: {e}. Using default configuration."
            )
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "test_cases": {},
            "dspy_config": {
                "optimization_enabled": True,
                "model": "openai/o3-mini-birthright",
                "max_tokens": 20000,
                "temperature": 1.0,
            },
            "generation_config": {
                "auto_generate_enabled": True,
                "default_timeout": 30.0,
            },
        }

    def setup_dspy(self):
        """Initialize DSPy with available language model"""
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

            # Configure DSPy with available model
            lm_config = {
                "model": "openai/o3-mini-birthright",
                "api_key": openai_api_key,
                "max_tokens": 20000,
                "temperature": 1.0,
            }

            if base_url:
                lm_config["base_url"] = base_url
                debug_print(f"Using custom base URL: {base_url}")

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
        """Optimize a prompt using DSPy approach"""
        if not self.optimizer:
            debug_print("DSPy optimizer not available, using fallback")
            return self._fallback_optimization(context)

        try:
            # Prepare inputs for DSPy
            tool_schema_str = self._format_tool_schema(context.tool_schema)
            current_arguments_str = json.dumps(context.generated_arguments, indent=2)
            failure_details_str = self._format_failure_details(context)
            success_requirements_str = self._format_success_requirements(
                context.success_criteria
            )

            dspy_print(f"Calling DSPy optimizer with:")
            debug_print(f"  Tool: {context.tool_name}")
            debug_print(f"  Schema: {tool_schema_str[:100]}...")
            debug_print(f"  Arguments: {current_arguments_str}")
            debug_print(f"  Original prompt: '{context.original_prompt}'")

            # Use DSPy to optimize the prompt
            result = self.optimizer(
                tool_name=context.tool_name,
                tool_schema=tool_schema_str,
                current_arguments=current_arguments_str,
                failure_details=failure_details_str,
                original_prompt=context.original_prompt,
                success_requirements=success_requirements_str,
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

    def generate_tool_instruction(
        self,
        tool_schema: Dict[str, Any],
        user_intent: str,
        available_context: Dict[str, Any] = None,
    ) -> str:
        """Generate a tool instruction from user intent"""
        if not self.call_generator:
            return self._fallback_instruction_generation(tool_schema, user_intent)

        try:
            tool_schema_str = self._format_tool_schema(tool_schema)
            context_str = json.dumps(available_context or {}, indent=2)

            result = self.call_generator(
                tool_schema=tool_schema_str,
                user_intent=user_intent,
                available_context=context_str,
            )
            return result.tool_instruction
        except Exception as e:
            error_print(f"DSPy instruction generation failed: {e}")
            return self._fallback_instruction_generation(tool_schema, user_intent)

    def _format_tool_schema(self, tool_schema: Dict[str, Any]) -> str:
        """Format tool schema for DSPy input"""
        if not tool_schema:
            return "No schema available"

        formatted = f"Tool Schema:\n"

        # Add description if available
        if "description" in tool_schema:
            formatted += f"Description: {tool_schema['description']}\n"

        # Add input parameters
        properties = tool_schema.get("properties", {})
        if properties:
            formatted += f"Parameters:\n"
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "No description")
                formatted += f"  - {param_name} ({param_type}): {param_desc}\n"

        # Add required parameters
        required = tool_schema.get("required", [])
        if required:
            formatted += f"Required Parameters: {', '.join(required)}\n"

        return formatted

    def _format_failure_details(self, context: OptimizationContext) -> str:
        """Format failure details for DSPy input"""
        details = f"FAILURE ANALYSIS:\n"
        details += f"Original Prompt: '{context.original_prompt}'\n"
        details += f"Failure Reason: {context.failure_reason}\n"
        details += (
            f"Arguments Used: {json.dumps(context.generated_arguments, indent=2)}\n"
        )

        if context.previous_attempts:
            details += f"\nPREVIOUS FAILED ATTEMPTS (avoid these patterns):\n"
            for i, attempt in enumerate(context.previous_attempts[-3:], 1):
                details += f"{i}. '{attempt}'\n"

        details += f"\nREQUIREMENTS FOR NEW PROMPT:\n"
        details += (
            f"- Must be completely different from original and previous attempts\n"
        )
        details += f"- Must be specific and actionable\n"
        details += f"- Must work with the provided arguments\n"
        details += f"- Must clearly state what the tool should accomplish\n"

        return details

    def _format_success_requirements(self, success_criteria: Dict[str, Any]) -> str:
        """Format success criteria for DSPy input"""
        if not success_criteria:
            return "No specific success criteria defined"

        requirements = "SUCCESS REQUIREMENTS:\n"

        for criterion, value in success_criteria.items():
            if criterion == "min_response_length":
                requirements += f"- Response must be at least {value} characters long\n"
            elif criterion == "no_error_keywords":
                requirements += (
                    f"- Response must not contain error keywords: {', '.join(value)}\n"
                )
            elif criterion == "expects_json":
                requirements += f"- Response should contain valid JSON data\n"
            elif criterion == "expects_list":
                requirements += f"- Response should contain a list or array\n"
            elif criterion == "expects_results":
                requirements += f"- Response should contain search/query results\n"
            elif criterion == "acceptable_not_found":
                requirements += (
                    f"- 'Not found' responses are acceptable if no data exists\n"
                )
            elif criterion == "expects_content":
                requirements += (
                    f"- Response should contain actual content from the tool\n"
                )
            else:
                requirements += f"- {criterion}: {value}\n"

        return requirements

    def _validate_optimized_prompt(
        self, optimized_prompt: str, context: OptimizationContext
    ) -> bool:
        """Validate that the optimized prompt is reasonable"""
        if not optimized_prompt or len(optimized_prompt.strip()) < 10:
            debug_print("Optimized prompt too short")
            return False

        if optimized_prompt.strip() == context.original_prompt.strip():
            debug_print("Optimized prompt identical to original")
            return False

        # Check against previous attempts
        for prev_attempt in context.previous_attempts:
            if optimized_prompt.strip() == prev_attempt.strip():
                debug_print("Optimized prompt matches previous attempt")
                return False

        # Validation - ensure it mentions the tool or a relevant action
        prompt_lower = optimized_prompt.lower()
        tool_name_lower = context.tool_name.lower()

        # Should reference the tool or a relevant action word
        action_words = [
            "test",
            "use",
            "call",
            "execute",
            "run",
            "perform",
            "demonstrate",
        ]
        if tool_name_lower not in prompt_lower and not any(
            word in prompt_lower for word in action_words
        ):
            debug_print("Optimized prompt doesn't reference tool or action")
            return False

        return True

    def _fallback_optimization(self, context: OptimizationContext) -> str:
        """Rule-based optimization fallback"""
        original = context.original_prompt
        tool_name = context.tool_name

        # Check if prompt is extremely vague
        vague_indicators = ["test", "try", "use", "call", "run", "execute"]
        words = original.lower().split()

        if len(words) <= 4 and any(
            indicator in words for indicator in vague_indicators
        ):
            # Generate more specific prompt based on schema
            tool_desc = context.tool_schema.get("description", f"the {tool_name} tool")
            if context.generated_arguments:
                arg_desc = ", ".join(
                    f"{k}={v}" for k, v in context.generated_arguments.items()
                )
                return f"Use {tool_desc} with arguments: {arg_desc}. Execute the operation and return the results."
            else:
                return f"Execute {tool_desc} and return the results in the expected format."

        # Add specificity based on success criteria
        enhancements = []
        criteria = context.success_criteria

        if criteria.get("expects_json"):
            enhancements.append("Return results in JSON format")
        elif criteria.get("expects_list"):
            enhancements.append("Return results as a list")
        elif criteria.get("expects_content"):
            enhancements.append("Return the actual content")

        if criteria.get("min_response_length"):
            enhancements.append("Provide detailed output")

        # Combine original with enhancements
        enhanced_prompt = original
        if enhancements:
            enhanced_prompt += ". " + ". ".join(enhancements) + "."

        # If still the same, add improvement
        if enhanced_prompt == original:
            enhanced_prompt += f". Execute this operation using the {tool_name} tool and provide clear results."

        return enhanced_prompt

    def _fallback_instruction_generation(
        self, tool_schema: Dict[str, Any], user_intent: str
    ) -> str:
        """Fallback for instruction generation"""
        tool_name = tool_schema.get("name", "tool")
        description = tool_schema.get("description", f"use the {tool_name}")

        return f"Use the {tool_name} to {user_intent}. {description}"

    def test_dspy_connection(self) -> bool:
        """Test if DSPy is properly configured and can make a call"""
        if not self.optimizer:
            warning_print("DSPy optimizer not initialized")
            return False

        try:
            dspy_print("Testing DSPy connection...")

            # Create a test schema
            test_schema = {
                "name": "test_tool",
                "description": "A test tool for verification",
                "properties": {
                    "test_param": {"type": "string", "description": "A test parameter"}
                },
                "required": ["test_param"],
            }

            # Create test context
            test_context = OptimizationContext(
                tool_name="test_tool",
                tool_schema=test_schema,
                original_prompt="test the tool",
                failure_reason="Response validation failed - test",
                generated_arguments={"test_param": "test_value"},
                success_criteria={"min_response_length": 10},
                previous_attempts=[],
            )

            optimized = self.optimize_prompt(test_context)

            success_print(f"DSPy test successful! Optimized: '{optimized}'")
            return True

        except Exception as e:
            error_print(f"DSPy test failed: {e}")
            import traceback

            debug_print(f"Full traceback:\n{traceback.format_exc()}")
            return False


def test_optimizer():
    """Test the DSPy optimizer functionality"""
    optimizer = DSPyOptimizer()

    # Test with a tool schema
    test_schema = {
        "name": "example_tool",
        "description": "An example tool that processes data",
        "properties": {
            "input_data": {"type": "string", "description": "The data to process"},
            "format": {"type": "string", "description": "Output format preference"},
        },
        "required": ["input_data"],
    }

    context = OptimizationContext(
        tool_name="example_tool",
        tool_schema=test_schema,
        original_prompt="use the tool",
        failure_reason="Response too vague",
        generated_arguments={"input_data": "test data", "format": "json"},
        success_criteria={"expects_json": True, "min_response_length": 20},
        previous_attempts=[],
    )

    optimized = optimizer.optimize_prompt(context)
    print(f"Original: {context.original_prompt}")
    print(f"Optimized: {optimized}")


if __name__ == "__main__":
    test_optimizer()
