"""
Enhanced DSPy-based prompt optimization for dynamic tool flight checks
Adds dependency analysis and intelligent test case generation
"""

import dspy
from typing import Dict, List, Any, Optional
import json
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime

# Import your existing color utilities
try:
    from color_utils import (
        debug_print,
        dspy_print,
        error_print,
        success_print,
        warning_print,
        optimization_print,
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

    def optimization_print(text):
        print(f"[OPTIMIZE] {text}")


@dataclass
class OptimizationContext:
    """Enhanced context for prompt optimization"""

    tool_name: str
    original_prompt: str
    failure_reason: str
    expected_output_format: str
    success_criteria: Dict[str, Any]
    previous_attempts: List[str]
    tool_arguments: Dict[str, Any] = None
    dependency_context: str = ""  # New: dependency information
    execution_history: List[Dict] = None  # New: previous execution results


# Enhanced DSPy Signatures for o3-mini


class DependencyAnalysisSignature(dspy.Signature):
    """Analyze dependencies between MCP tools using o3-mini intelligence"""

    tool_descriptions = dspy.InputField(
        desc="Detailed descriptions of all available MCP tools with their schemas and purposes"
    )
    dependencies = dspy.OutputField(
        desc='JSON structure describing discovered dependencies between tools. Include dependency type (setup/data/prerequisite/optional), confidence score (0-1), and detailed description of the relationship. Format: {"tool_name": [{"provider": "provider_tool", "consumer": "consumer_tool", "type": "dependency_type", "description": "relationship_description", "confidence": 0.8}]}'
    )


class TestCaseGenerationSignature(dspy.Signature):
    """Generate intelligent test cases for MCP tools"""

    tool_name = dspy.InputField(desc="Name of the tool to test")
    tool_description = dspy.InputField(desc="Description of what the tool does")
    tool_schema = dspy.InputField(desc="JSON schema of tool input parameters")
    dependency_context = dspy.InputField(
        desc="Information about tool dependencies and execution order"
    )
    existing_tests = dspy.InputField(
        desc="Information about existing test cases to avoid duplication"
    )
    test_prompt = dspy.OutputField(
        desc="Natural language prompt that will effectively test this tool. Should be specific, actionable, and designed to validate the tool's core functionality."
    )
    expected_indicators = dspy.OutputField(
        desc="Comma-separated list of words/phrases that should appear in successful responses"
    )
    success_criteria = dspy.OutputField(
        desc="JSON object defining what constitutes a successful test response, including patterns to look for and acceptable error conditions"
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

    def forward(self, tool_description, user_intent):
        return self.generate(tool_description=tool_description, user_intent=user_intent)


class EnhancedPromptOptimizationSignature(dspy.Signature):
    """Enhanced prompt optimization with dependency and context awareness"""

    tool_name = dspy.InputField(desc="Name of the tool being tested")
    original_prompt = dspy.InputField(desc="The prompt that failed")
    failure_reason = dspy.InputField(desc="Detailed analysis of why the prompt failed")
    tool_context = dspy.InputField(
        desc="Complete context about the tool including dependencies and expected behavior"
    )
    execution_history = dspy.InputField(
        desc="History of previous optimization attempts and their results"
    )
    optimized_prompt = dspy.OutputField(
        desc="Completely rewritten prompt that addresses all identified failure points. Should be specific, clear, and likely to succeed based on the tool's actual capabilities and dependencies."
    )
    optimization_strategy = dspy.OutputField(
        desc="Brief explanation of the optimization strategy used and why it should be more effective"
    )


# Enhanced DSPy Modules


class DependencyAnalyzer(dspy.Module):
    """DSPy module for analyzing tool dependencies"""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(DependencyAnalysisSignature)

    def forward(self, tool_descriptions):
        return self.analyze(tool_descriptions=tool_descriptions)


class TestCaseGenerator(dspy.Module):
    """DSPy module for generating intelligent test cases"""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(TestCaseGenerationSignature)

    def forward(
        self,
        tool_name,
        tool_description,
        tool_schema,
        dependency_context,
        existing_tests,
    ):
        return self.generate(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_schema=tool_schema,
            dependency_context=dependency_context,
            existing_tests=existing_tests,
        )


class EnhancedPromptOptimizer(dspy.Module):
    """Enhanced DSPy module for optimizing tool prompts"""

    def __init__(self):
        super().__init__()
        self.optimize = dspy.ChainOfThought(EnhancedPromptOptimizationSignature)

    def forward(
        self,
        tool_name,
        original_prompt,
        failure_reason,
        tool_context,
        execution_history,
    ):
        return self.optimize(
            tool_name=tool_name,
            original_prompt=original_prompt,
            failure_reason=failure_reason,
            tool_context=tool_context,
            execution_history=execution_history,
        )


class DSPyFlightOptimizer:
    """Enhanced DSPy-based flight check optimizer with dependency analysis"""

    def __init__(self, anthropic_client=None):
        self.anthropic_client = anthropic_client

        # Enhanced DSPy modules
        self.optimizer = None
        self.dependency_analyzer = None
        self.test_generator = None

        # Your existing modules (for backward compatibility)
        self.call_generator = None

        self.setup_dspy()

    def setup_dspy(self):
        """Initialize DSPy with OpenAI o3-mini-birthright model"""
        dspy_print("Setting up Enhanced DSPy optimizer...")

        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("BASE_URL")

            if not openai_api_key:
                warning_print("OPENAI_API_KEY not found - DSPy will be unavailable")
                return

            # Import check should be at module level, not here
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

            self.optimizer = EnhancedPromptOptimizer()
            self.dependency_analyzer = DependencyAnalyzer()
            self.test_generator = TestCaseGenerator()
            self.call_generator = ToolCallGenerator()

            success_print("Enhanced DSPy setup complete with dependency analysis!")

        except ImportError:
            warning_print("DSPy not available - install with: pip install dspy-ai")
            self._disable_dspy()
        except Exception as e:
            error_print(f"DSPy setup failed: {e}")
            self._disable_dspy()

    def _disable_dspy(self):
        """Disable DSPy components gracefully"""
        self.optimizer = None
        self.dependency_analyzer = None
        self.test_generator = None
        self.call_generator = None

    def analyze_tool_dependencies(self, tools: List[Dict]) -> Dict[str, List[Dict]]:
        """Use o3-mini to analyze tool dependencies dynamically"""
        if not self.dependency_analyzer:
            debug_print("DSPy dependency analyzer not available, using fallback")
            return self._fallback_dependency_analysis(tools)

        try:
            dspy_print("Analyzing tool dependencies with o3-mini...")

            # Prepare comprehensive tool descriptions
            tool_descriptions = self._prepare_tool_descriptions_for_analysis(tools)

            # Call DSPy dependency analyzer
            result = self.dependency_analyzer(tool_descriptions=tool_descriptions)

            # Parse the JSON result
            dependencies = self._parse_dependency_result(result.dependencies, tools)

            optimization_print(
                f"DSPy discovered {len(dependencies)} dependency relationships"
            )
            return dependencies

        except Exception as e:
            error_print(f"DSPy dependency analysis failed: {e}")
            debug_print(f"Falling back to rule-based analysis")
            return self._fallback_dependency_analysis(tools)

    def generate_intelligent_test_case(
        self, tool: Dict, dependency_context: str, existing_tests: List[Dict] = None
    ) -> Dict[str, Any]:
        """Generate intelligent test case using o3-mini"""
        if not self.test_generator:
            debug_print("DSPy test generator not available, using fallback")
            return self._fallback_test_generation(tool)

        try:
            dspy_print(
                f"Generating intelligent test case for {tool['name']} with o3-mini..."
            )

            # Prepare inputs
            tool_schema = json.dumps(tool.get("input_schema", {}), indent=2)
            existing_tests_str = json.dumps(existing_tests or [], indent=2)

            # Call DSPy test generator
            result = self.test_generator(
                tool_name=tool["name"],
                tool_description=tool.get("description", ""),
                tool_schema=tool_schema,
                dependency_context=dependency_context,
                existing_tests=existing_tests_str,
            )

            # Parse and return test case data
            return {
                "prompt": result.test_prompt,
                "expected_indicators": [
                    indicator.strip()
                    for indicator in result.expected_indicators.split(",")
                    if indicator.strip()
                ],
                "success_criteria": self._parse_success_criteria(
                    result.success_criteria
                ),
            }

        except Exception as e:
            error_print(f"DSPy test case generation failed for {tool['name']}: {e}")
            return self._fallback_test_generation(tool)

    def optimize_prompt(self, context: OptimizationContext) -> str:
        """Enhanced prompt optimization using o3-mini"""
        if not self.optimizer:
            debug_print("Enhanced DSPy optimizer not available, using fallback")
            return self._enhanced_fallback_optimization(context)

        try:
            optimization_print(
                f"Optimizing prompt for {context.tool_name} with enhanced o3-mini analysis..."
            )

            # Prepare enhanced context
            tool_context = self._prepare_tool_context(context)
            execution_history = json.dumps(context.execution_history or [], indent=2)

            # Call enhanced DSPy optimizer
            result = self.optimizer(
                tool_name=context.tool_name,
                original_prompt=context.original_prompt,
                failure_reason=context.failure_reason,
                tool_context=tool_context,
                execution_history=execution_history,
            )

            optimized_prompt = result.optimized_prompt.strip()
            optimization_strategy = result.optimization_strategy.strip()

            success_print(f"DSPy optimization strategy: {optimization_strategy}")
            debug_print(f"Optimized prompt: '{optimized_prompt}'")

            # Validate the optimized prompt
            if self._validate_optimized_prompt(optimized_prompt, context):
                return optimized_prompt
            else:
                warning_print("Enhanced DSPy result failed validation, using fallback")
                return self._enhanced_fallback_optimization(context)

        except Exception as e:
            error_print(f"Enhanced DSPy optimization failed: {e}")
            return self._enhanced_fallback_optimization(context)

    def _prepare_tool_descriptions_for_analysis(self, tools: List[Dict]) -> str:
        """Prepare comprehensive tool descriptions for dependency analysis"""
        descriptions = []

        for tool in tools:
            desc = f"""
TOOL: {tool['name']}
DESCRIPTION: {tool.get('description', 'No description available')}
INPUT_SCHEMA: {json.dumps(tool.get('input_schema', {}), indent=2)}
ANNOTATIONS: {json.dumps(tool.get('annotations', {}), indent=2)}

ANALYSIS_QUESTIONS:
1. What type of data does this tool produce?
2. What type of data does this tool consume?
3. Which other tools might depend on this tool's output?
4. Which other tools might this tool depend on for setup or data?
5. What is the logical execution order relative to other tools?
"""
            descriptions.append(desc)

        combined = "\n" + "=" * 80 + "\n".join(descriptions)
        combined += f"\n\nAVAILABLE_TOOLS: {[t['name'] for t in tools]}"

        return combined

    def _parse_dependency_result(
        self, dependencies_json: str, tools: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Parse DSPy dependency analysis result"""
        try:
            # Try to parse JSON response
            deps_data = json.loads(dependencies_json)

            # Convert to standard format
            parsed_deps = {}
            tool_names = [t["name"] for t in tools]

            for tool_name, tool_deps in deps_data.items():
                if tool_name in tool_names:
                    parsed_deps[tool_name] = []

                    for dep in tool_deps:
                        if all(
                            key in dep
                            for key in ["provider", "consumer", "type", "description"]
                        ):
                            parsed_deps[tool_name].append(
                                {
                                    "provider_tool": dep["provider"],
                                    "consumer_tool": dep["consumer"],
                                    "dependency_type": dep["type"],
                                    "description": dep["description"],
                                    "confidence": dep.get("confidence", 0.8),
                                    "auto_discovered": True,
                                }
                            )

            return parsed_deps

        except (json.JSONDecodeError, KeyError) as e:
            warning_print(f"Failed to parse DSPy dependency result: {e}")
            return self._fallback_dependency_analysis(tools)

    def _parse_success_criteria(self, criteria_str: str) -> Dict[str, Any]:
        """Parse success criteria from DSPy result"""
        try:
            return json.loads(criteria_str)
        except json.JSONDecodeError:
            # Fallback to basic criteria
            return {"min_response_length": 10, "acceptable_not_found": False}

    def _prepare_tool_context(self, context: OptimizationContext) -> str:
        """Prepare comprehensive tool context for optimization"""
        tool_context = f"""
TOOL_NAME: {context.tool_name}
ORIGINAL_PROMPT: {context.original_prompt}
FAILURE_ANALYSIS: {context.failure_reason}
EXPECTED_FORMAT: {context.expected_output_format}
SUCCESS_CRITERIA: {json.dumps(context.success_criteria, indent=2)}
TOOL_ARGUMENTS: {json.dumps(context.tool_arguments or {}, indent=2)}
DEPENDENCY_CONTEXT: {context.dependency_context}

PREVIOUS_ATTEMPTS: {len(context.previous_attempts)} attempts
{chr(10).join(f"  {i+1}. {attempt}" for i, attempt in enumerate(context.previous_attempts[-3:]))}

OPTIMIZATION_REQUIREMENTS:
1. The new prompt must be completely different from all previous attempts
2. Must address the specific failure reason provided
3. Must be actionable and specific to the tool's capabilities
4. Must work with the provided tool arguments
5. Must consider tool dependencies and execution context
"""
        return tool_context

    def _validate_optimized_prompt(
        self, optimized_prompt: str, context: OptimizationContext
    ) -> bool:
        """Enhanced validation of optimized prompts"""
        if not optimized_prompt or len(optimized_prompt.strip()) < 10:
            debug_print("Prompt too short")
            return False

        if optimized_prompt.strip() == context.original_prompt.strip():
            debug_print("Prompt unchanged")
            return False

        # Check against previous attempts
        for prev_attempt in context.previous_attempts:
            if optimized_prompt.strip() == prev_attempt.strip():
                debug_print("Prompt matches previous attempt")
                return False

        # Tool-specific validation
        tool_name_lower = context.tool_name.lower()
        prompt_lower = optimized_prompt.lower()

        required_elements = []
        if "search" in tool_name_lower:
            required_elements = ["search", "papers"]
        elif "extract" in tool_name_lower:
            required_elements = ["extract", "information"]
        elif "read" in tool_name_lower:
            required_elements = ["read", "file"]
        elif "list" in tool_name_lower:
            required_elements = ["list", "directory"]
        elif "fetch" in tool_name_lower:
            required_elements = ["fetch", "content"]

        if required_elements:
            if not any(element in prompt_lower for element in required_elements):
                debug_print(
                    f"Missing required elements for {context.tool_name}: {required_elements}"
                )
                return False

        return True

    def _fallback_dependency_analysis(self, tools: List[Dict]) -> Dict[str, List[Dict]]:
        """Enhanced fallback dependency analysis"""
        dependencies = {}
        tool_names = [t["name"] for t in tools]

        for tool in tools:
            name = tool["name"]
            deps = []

            # Enhanced pattern matching
            if "search" in name.lower():
                # Search tools provide data for extraction and info tools
                for other_tool in tools:
                    other_name = other_tool["name"]
                    if (
                        "extract" in other_name.lower()
                        or "info" in other_name.lower()
                        or "get" in other_name.lower()
                    ):
                        deps.append(
                            {
                                "provider_tool": name,
                                "consumer_tool": other_name,
                                "dependency_type": "data",
                                "description": f"{name} provides search results that {other_name} can process",
                                "confidence": 0.8,
                                "auto_discovered": True,
                            }
                        )

            elif "list" in name.lower() or "directory" in name.lower():
                # Directory tools provide setup for file operations
                for other_tool in tools:
                    other_name = other_tool["name"]
                    if (
                        "read" in other_name.lower()
                        or "file" in other_name.lower()
                        or "write" in other_name.lower()
                    ):
                        deps.append(
                            {
                                "provider_tool": name,
                                "consumer_tool": other_name,
                                "dependency_type": "prerequisite",
                                "description": f"{name} helps {other_name} discover available files and paths",
                                "confidence": 0.7,
                                "auto_discovered": True,
                            }
                        )

            elif "fetch" in name.lower():
                # Fetch tools can provide data for processing tools
                for other_tool in tools:
                    other_name = other_tool["name"]
                    if (
                        "extract" in other_name.lower()
                        or "process" in other_name.lower()
                    ):
                        deps.append(
                            {
                                "provider_tool": name,
                                "consumer_tool": other_name,
                                "dependency_type": "optional",
                                "description": f"{name} can fetch web content for {other_name} to process",
                                "confidence": 0.5,
                                "auto_discovered": True,
                            }
                        )

            dependencies[name] = deps

        return dependencies

    def _fallback_test_generation(self, tool: Dict) -> Dict[str, Any]:
        """Enhanced fallback test generation with better pattern matching"""
        name = tool["name"]
        name_lower = name.lower()

        # Enhanced prompt templates with better specificity
        prompt_templates = {
            "search_papers": {
                "prompt": "Search for academic papers about machine learning. Return exactly 2 arXiv paper IDs.",
                "indicators": ["paper", "arxiv", "id", "machine", "learning"],
                "criteria": {
                    "min_response_length": 15,
                    "required_patterns": ["paper"],
                    "acceptable_not_found": False,
                },
            },
            "extract_info": {
                "prompt": "Extract detailed information about paper with ID 'test_paper_123'. Show title, authors, summary, and publication details. If no paper is found, clearly state that no information is available.",
                "indicators": ["title", "author", "summary", "information"],
                "criteria": {
                    "min_response_length": 20,
                    "acceptable_not_found": True,
                    "required_patterns": [],
                },
            },
            "read_file": {
                "prompt": "Read the server_config.json file and display its complete contents. Show the JSON structure clearly.",
                "indicators": ["json", "config", "content", "file"],
                "criteria": {
                    "min_response_length": 30,
                    "required_patterns": ["json", "config"],
                    "expect_json": False,  # Content might contain JSON but response itself isn't JSON
                },
            },
            "list_directory": {
                "prompt": "List all files and directories in the current folder. Show both files and directories with clear indicators.",
                "indicators": ["file", "directory", "dir", "["],
                "criteria": {
                    "min_response_length": 20,
                    "required_patterns": ["file"],
                    "acceptable_not_found": False,
                },
            },
            "fetch": {
                "prompt": "Fetch content from https://example.com and display the returned content with proper formatting.",
                "indicators": ["content", "response", "http", "fetch"],
                "criteria": {
                    "min_response_length": 25,
                    "required_patterns": ["content"],
                    "acceptable_not_found": False,
                },
            },
        }

        # Try exact match first
        if name in prompt_templates:
            template = prompt_templates[name]
            return {
                "prompt": template["prompt"],
                "expected_indicators": template["indicators"],
                "success_criteria": template["criteria"],
            }

        # Pattern-based matching with enhanced logic
        best_match = None
        best_score = 0

        for pattern, template in prompt_templates.items():
            score = 0
            pattern_parts = pattern.split("_")

            for part in pattern_parts:
                if part in name_lower:
                    score += 1

            # Bonus for exact substring matches
            if pattern.replace("_", "") in name_lower.replace("_", ""):
                score += 2

            if score > best_score:
                best_score = score
                best_match = template

        if best_match:
            return {
                "prompt": best_match["prompt"],
                "expected_indicators": best_match["indicators"],
                "success_criteria": best_match["criteria"],
            }

        # Ultimate fallback - generic test
        return {
            "prompt": f"Test the {name} tool by using it effectively. Provide a clear, detailed response showing the tool's functionality.",
            "expected_indicators": ["response", "result", "output"],
            "success_criteria": {
                "min_response_length": 10,
                "acceptable_not_found": True,
            },
        }

    def _enhanced_fallback_optimization(self, context: OptimizationContext) -> str:
        """Enhanced fallback optimization with better pattern recognition"""
        original = context.original_prompt
        tool_name = context.tool_name
        failure_reason = context.failure_reason.lower()

        # Analyze failure type and apply targeted improvements
        if "timeout" in failure_reason:
            return f"{original}. Please respond quickly and efficiently."

        if "empty response" in failure_reason or "no response" in failure_reason:
            return f"Use the {tool_name} tool to provide a detailed response. {original}. Ensure you return meaningful output."

        if "validation" in failure_reason or "indicators" in failure_reason:
            if "search" in tool_name.lower():
                return f"Search for academic papers and return specific arXiv paper IDs. {original}. List the paper IDs found."
            elif "extract" in tool_name.lower():
                return f"Extract detailed information including title, authors, and summary. {original}. If no information is found, state this clearly."
            elif "read" in tool_name.lower():
                return f"Read the specified file and display its complete contents. {original}. Show the file data clearly."
            elif "list" in tool_name.lower():
                return f"List all files and directories with clear formatting. {original}. Use [FILE] and [DIR] indicators."
            elif "fetch" in tool_name.lower():
                return f"Fetch web content and display the response. {original}. Show the retrieved data."

        if "error" in failure_reason and "not found" in failure_reason:
            if "extract" in tool_name.lower():
                return f"Extract information about the specified item. {original}. If the item is not found, respond with 'No saved information available for this item.'"

        # Pattern-based improvements
        improvements = {
            "search": "Search for academic papers and return a list of specific paper IDs from arXiv.",
            "extract": "Extract detailed information about the specified item. Include title, authors, summary, and other relevant details.",
            "read": "Read the specified file and display its complete contents in a clear format.",
            "list": "List all files and directories in the specified path with clear type indicators.",
            "fetch": "Fetch content from the specified URL and display the response data.",
            "get": "Retrieve the requested information and present it in a structured format.",
            "find": "Find and return the requested items with detailed information.",
        }

        tool_lower = tool_name.lower()
        for pattern, improvement in improvements.items():
            if pattern in tool_lower:
                return f"{improvement} Original context: {original}"

        # Final fallback
        return f"Execute the {tool_name} tool effectively. {original}. Provide a clear, detailed response that demonstrates the tool's functionality."

    # Backward compatibility methods (keeping your existing interface)

    def generate_tool_call(
        self, tool_name: str, tool_description: str, user_intent: str
    ) -> str:
        """Generate a tool call from user intent (backward compatibility)"""
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
        """Get detailed description of the tool for optimization context (backward compatibility)"""
        tool_descriptions = {
            "search_papers": """
Tool Function: Searches academic papers on arXiv based on a topic query.
Input Parameters: 
- topic (string): The research topic or subject area to search for
- max_results (int): Maximum number of papers to return (default: 5)
Output Format: Returns a list of arXiv paper IDs (e.g., ["1909.03550v1", "2001.12345v2"])
Usage Notes: The tool searches arXiv's database and saves paper metadata locally. It expects specific topic keywords and returns paper identifiers that can be used with extract_info tool.
Dependencies: Provides data for extract_info tool.
""",
            "extract_info": """
Tool Function: Extracts detailed information about a specific academic paper.
Input Parameters:
- paper_id (string): The arXiv paper ID to look up (e.g., "1909.03550v1")
Output Format: Returns JSON or structured text with paper details including title, authors, summary, publication date, and PDF URL.
Usage Notes: Searches local database of previously downloaded papers. If paper not found locally, returns a "no saved information" message. Requires exact paper ID format.
Dependencies: Consumes data from search_papers tool.
""",
            "read_file": """
Tool Function: Reads and returns the contents of a specified file.
Input Parameters:
- path (string): File path to read (e.g., "server_config.json", "data.txt")
Output Format: Returns the complete file contents as text.
Usage Notes: Can read any accessible file. Common use cases include configuration files, data files, logs. Returns raw file content without modification.
Dependencies: May benefit from list_directory for path discovery.
""",
            "list_directory": """
Tool Function: Lists all files and directories in a specified path.
Input Parameters:
- path (string): Directory path to list (e.g., ".", "/home/user", "data/")
Output Format: Returns formatted list showing [FILE] or [DIR] prefix followed by item names.
Usage Notes: Provides directory navigation and file discovery. Shows both files and subdirectories with clear type indicators.
Dependencies: Provides setup information for read_file and other file operations.
""",
            "fetch": """
Tool Function: Fetches content from a web URL and returns the data.
Input Parameters:
- url (string): The web URL to fetch content from (e.g., "https://example.com/api/data")
Output Format: Returns the fetched content, often with content-type information and formatted display.
Usage Notes: Can fetch web pages, API responses, JSON data. Handles various content types and provides appropriate formatting for display.
Dependencies: Independent tool, but can provide data for extract_info and other processing tools.
""",
        }

        return tool_descriptions.get(
            tool_name, f"Tool: {tool_name} - No detailed description available."
        )

    def _prepare_failure_info(self, context: OptimizationContext) -> str:
        """Prepare failure information for DSPy (backward compatibility)"""
        failure_info = f"""
ENHANCED FAILURE ANALYSIS:
Original Prompt: "{context.original_prompt}"
Failure Reason: {context.failure_reason}
Tool Arguments: {json.dumps(context.tool_arguments, indent=2) if context.tool_arguments else "None"}
Dependency Context: {context.dependency_context}

IDENTIFIED PROBLEMS:
1. The prompt may be too vague or unclear
2. The prompt might not align with tool capabilities
3. The prompt may not consider tool dependencies
4. The expected output format might not match tool behavior

OPTIMIZATION REQUIREMENTS:
- Must be completely different from the original
- Must be specific and actionable
- Must clearly state expected output
- Must consider tool dependencies and execution context
- Must work with the provided tool arguments

ENHANCED EXAMPLES:
- For search_papers: "Search for academic papers about [topic]. Return exactly [number] arXiv paper IDs in a list format."
- For extract_info: "Extract detailed information about paper with ID '[paper_id]'. Show title, authors, summary, and publication date. If paper not found, state 'No saved information available.'"
- For read_file: "Read the file '[filename]' and display its complete contents with proper formatting."
- For list_directory: "List all files and directories in '[path]'. Use [FILE] and [DIR] prefixes for clear identification."
- For fetch: "Fetch content from '[url]' and display the response with appropriate formatting."
"""

        if context.previous_attempts:
            failure_info += f"\n\nPREVIOUS FAILED ATTEMPTS (AVOID THESE):\n"
            for i, attempt in enumerate(context.previous_attempts[-3:], 1):
                failure_info += f"{i}. '{attempt}'\n"
            failure_info += "\nThe new prompt must be significantly different from all previous attempts."

        return failure_info.strip()

    def _fallback_call_generation(self, tool_name: str, user_intent: str) -> str:
        """Enhanced fallback call generation"""
        templates = {
            "search_papers": f"Search for academic papers about {user_intent}. Return the paper IDs found.",
            "extract_info": f"Extract detailed information about the paper related to {user_intent}. Show title, authors, and summary.",
            "read_file": f"Read the file that contains information about {user_intent}. Display the file contents.",
            "list_directory": f"List the directory contents to find files related to {user_intent}. Show files and directories clearly.",
            "fetch": f"Fetch web content related to {user_intent}. Display the retrieved data.",
        }

        return templates.get(
            tool_name,
            f"Use the {tool_name} tool effectively to help with: {user_intent}",
        )

    def test_dspy_connection(self):
        """Enhanced test of DSPy connection and modules"""
        if not self.optimizer:
            warning_print("Enhanced DSPy optimizer not initialized")
            return False

        try:
            dspy_print("Testing enhanced DSPy connection and modules...")

            # Test enhanced prompt optimizer
            result = self.optimizer(
                tool_name="test_tool",
                original_prompt="test prompt that needs improvement",
                failure_reason="This is a test to verify enhanced DSPy is working correctly",
                tool_context="Test tool context with enhanced capabilities",
                execution_history="[]",
            )

            success_print(f"Enhanced DSPy test successful!")
            success_print(f"Optimized prompt: '{result.optimized_prompt}'")
            success_print(f"Strategy: '{result.optimization_strategy}'")
            return True

        except Exception as e:
            error_print(f"Enhanced DSPy test failed: {e}")
            import traceback

            debug_print(f"Full traceback:\n{traceback.format_exc()}")
            return False


def test_enhanced_dspy_optimizer():
    """Test the enhanced DSPy optimizer functionality"""
    optimizer = DSPyFlightOptimizer()

    # Test dependency analysis
    tools = [
        {
            "name": "search_papers",
            "description": "Search academic papers",
            "input_schema": {},
        },
        {
            "name": "extract_info",
            "description": "Extract paper information",
            "input_schema": {},
        },
        {"name": "read_file", "description": "Read file contents", "input_schema": {}},
    ]

    print("Testing dependency analysis...")
    dependencies = optimizer.analyze_tool_dependencies(tools)
    print(f"Dependencies discovered: {len(dependencies)}")

    # Test test case generation
    print("\nTesting test case generation...")
    test_case = optimizer.generate_intelligent_test_case(
        tools[0], "search_papers provides data for extract_info"
    )
    print(f"Generated test case: {test_case}")

    # Test enhanced optimization
    print("\nTesting enhanced optimization...")
    context = OptimizationContext(
        tool_name="search_papers",
        original_prompt="Find papers about machine learning",
        failure_reason="Response validation failed - no paper IDs found",
        expected_output_format="list_of_ids",
        success_criteria={"contains_arxiv_ids": True, "min_response_length": 5},
        previous_attempts=[],
        tool_arguments={"topic": "machine learning", "max_results": 2},
        dependency_context="Provides data for extract_info tool",
    )

    optimized = optimizer.optimize_prompt(context)
    print(f"Original: {context.original_prompt}")
    print(f"Optimized: {optimized}")


if __name__ == "__main__":
    test_enhanced_dspy_optimizer()
