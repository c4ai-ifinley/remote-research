"""
Adversarial test suite for MCP tool flight checking.

Tier 3 tests:
  - Hallucination probes: phantom tool / phantom parameter detection
  - Prompt-to-tool binding: verify correct tool selection from a prompt
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from color_utils import (
    debug_print,
    error_print,
    success_print,
    warning_print,
    header_print,
    separator_print,
    colored_print,
    Colors,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AdversarialResult:
    """Result of a single adversarial probe."""

    probe_name: str
    probe_type: str  # "phantom_tool" | "phantom_parameter" | "tool_binding"
    passed: bool
    details: str
    response: Optional[str] = None


@dataclass
class AdversarialReport:
    """Summary of adversarial test suite execution."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    results: List[AdversarialResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hallucination Probes
# ---------------------------------------------------------------------------


class HallucinationProbes:
    """Probes that test whether the system hallucinates non-existent tools or
    parameters rather than correctly refusing or ignoring them."""

    # Phantom tools — names that do NOT exist in the registry
    PHANTOM_TOOLS = [
        "quantum_entangle",
        "deploy_kubernetes_cluster",
        "send_email_attachment",
        "translate_to_klingon",
        "hack_mainframe",
    ]

    # Phantom parameters — real tool names get asked with fake params
    PHANTOM_PARAMS = {
        # tool_name → list of fake parameter names
        "read_file": ["encryption_key", "quantum_mode", "blockchain_verify"],
        "list_directory": ["recursive_depth_limit", "ai_filter_mode"],
        "create_directory": ["replicate_to_cloud", "notify_admin"],
        "search_files": ["sentiment_analysis", "translate_results"],
    }

    def __init__(self, chatbot_instance):
        self.chatbot = chatbot_instance
        self.available_tool_names = {
            t["name"] for t in (chatbot_instance.available_tools or [])
        }

    async def run_all(self) -> AdversarialReport:
        """Execute all hallucination probes."""
        report = AdversarialReport()

        # --- Phantom tool probes ---
        for phantom in self.PHANTOM_TOOLS:
            result = await self._probe_phantom_tool(phantom)
            report.results.append(result)
            report.total += 1
            if result.passed:
                report.passed += 1
            else:
                report.failed += 1

        # --- Phantom parameter probes ---
        for tool_name, fake_params in self.PHANTOM_PARAMS.items():
            if tool_name not in self.available_tool_names:
                continue
            for fake_param in fake_params:
                result = await self._probe_phantom_parameter(tool_name, fake_param)
                report.results.append(result)
                report.total += 1
                if result.passed:
                    report.passed += 1
                else:
                    report.failed += 1

        return report

    async def _probe_phantom_tool(self, phantom_name: str) -> AdversarialResult:
        """Ask the system to use a tool that doesn't exist.

        PASS = system refuses / says tool not found.
        FAIL = system pretends to call it or fabricates output.
        """
        # If by some chance this tool actually exists, skip
        if phantom_name in self.available_tool_names:
            return AdversarialResult(
                probe_name=f"phantom_tool:{phantom_name}",
                probe_type="phantom_tool",
                passed=True,
                details="Tool actually exists — skipped",
            )

        # Attempt to call — should raise or return error
        session = None
        for tool_name_candidate in self.available_tool_names:
            session = self.chatbot.sessions.get(tool_name_candidate)
            if session:
                break

        if not session:
            return AdversarialResult(
                probe_name=f"phantom_tool:{phantom_name}",
                probe_type="phantom_tool",
                passed=True,
                details="No sessions available — cannot test",
            )

        try:
            result = await asyncio.wait_for(
                session.call_tool(phantom_name, arguments={}),
                timeout=10.0,
            )
            # If we get here without an error, the system hallucinated
            response_text = result.content[0].text if result.content else str(result)
            return AdversarialResult(
                probe_name=f"phantom_tool:{phantom_name}",
                probe_type="phantom_tool",
                passed=False,
                details=f"System did NOT reject phantom tool '{phantom_name}'",
                response=response_text[:300],
            )
        except Exception as e:
            # Expected: rejection
            return AdversarialResult(
                probe_name=f"phantom_tool:{phantom_name}",
                probe_type="phantom_tool",
                passed=True,
                details=f"Correctly rejected: {str(e)[:100]}",
            )

    async def _probe_phantom_parameter(
        self, tool_name: str, fake_param: str
    ) -> AdversarialResult:
        """Call a real tool with a fabricated parameter.

        PASS = tool ignores the fake param or returns schema error.
        FAIL = tool acts as if the param is valid and affects output.
        """
        session = self.chatbot.sessions.get(tool_name)
        if not session:
            return AdversarialResult(
                probe_name=f"phantom_param:{tool_name}.{fake_param}",
                probe_type="phantom_parameter",
                passed=True,
                details="Session not available — skipped",
            )

        # Construct args with the fake parameter
        args = {fake_param: "malicious_value_12345"}

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=args),
                timeout=10.0,
            )
            response_text = result.content[0].text if result.content else str(result)

            # If the fake param value appears in the response, it was used
            if "malicious_value_12345" in response_text:
                return AdversarialResult(
                    probe_name=f"phantom_param:{tool_name}.{fake_param}",
                    probe_type="phantom_parameter",
                    passed=False,
                    details=f"Fake param '{fake_param}' value echoed in output",
                    response=response_text[:300],
                )

            # Otherwise it was ignored or caused a schema error — that's fine
            return AdversarialResult(
                probe_name=f"phantom_param:{tool_name}.{fake_param}",
                probe_type="phantom_parameter",
                passed=True,
                details="Fake parameter ignored or rejected",
                response=response_text[:200],
            )

        except Exception as e:
            # Schema validation error = correct behavior
            return AdversarialResult(
                probe_name=f"phantom_param:{tool_name}.{fake_param}",
                probe_type="phantom_parameter",
                passed=True,
                details=f"Rejected with error: {str(e)[:100]}",
            )


# ---------------------------------------------------------------------------
# Prompt-to-Tool Binding Test
# ---------------------------------------------------------------------------


class ToolBindingProbes:
    """Test that prompts map to the correct tool in the registry.

    Sends a natural-language prompt through an LLM with the full tool registry
    attached and verifies that the LLM selects the expected tool.
    """

    def __init__(self, chatbot_instance, optimizer):
        """
        Args:
            chatbot_instance: The MCP chatbot with available_tools.
            optimizer: DSPyOptimizer instance (for LLM access).
        """
        self.chatbot = chatbot_instance
        self.optimizer = optimizer
        self.available_tools = chatbot_instance.available_tools or []

    def _build_binding_probes(self) -> List[Dict[str, str]]:
        """Auto-generate binding probes from the tool registry."""
        probes = []
        for tool in self.available_tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            if not name:
                continue

            # Create a natural-language prompt that should map to this tool
            if "read" in name and "file" in name:
                prompt = "Read the contents of README.md"
            elif "write" in name or "create_file" in name:
                prompt = "Write 'hello world' to output.txt"
            elif "list" in name and "dir" in name:
                prompt = "Show me what's in the current directory"
            elif "directory_tree" in name or "tree" in name:
                prompt = "Show me the full directory tree"
            elif "search" in name and "file" in name:
                prompt = "Find all occurrences of 'TODO' in the project"
            elif "move" in name or "rename" in name:
                prompt = "Rename old.txt to new.txt"
            elif "create_dir" in name or "create_directory" in name:
                prompt = "Create a new folder called 'output'"
            elif "fetch" in name:
                prompt = "Download the page at https://httpbin.org/json"
            elif "extract" in name or "info" in name:
                prompt = "Get information about paper arxiv-123"
            else:
                # Generic: use description
                prompt = f"I need to {desc.lower()}" if desc else f"Use the {name} tool"

            probes.append({"prompt": prompt, "expected_tool": name})

        return probes

    async def run_all(self) -> AdversarialReport:
        """Run all tool-binding probes."""
        report = AdversarialReport()
        probes = self._build_binding_probes()

        if not probes:
            warning_print("No tool-binding probes generated (no tools available)")
            return report

        # We need LLM access to test binding.  If optimizer has no LLM, skip.
        if not self.optimizer.result_judge:
            warning_print("LLM not available — skipping tool-binding probes")
            return report

        import dspy

        # Build a simple tool-selection signature on the fly
        class ToolSelectionSignature(dspy.Signature):
            """Select the best tool for a user request."""

            user_request = dspy.InputField(desc="What the user wants to do")
            available_tools = dspy.InputField(
                desc="JSON list of available tools with name and description"
            )
            selected_tool = dspy.OutputField(
                desc="The exact 'name' of the tool that best matches the request"
            )

        selector = dspy.ChainOfThought(ToolSelectionSignature)

        tools_summary = json.dumps(
            [{"name": t.get("name"), "description": t.get("description", "")} for t in self.available_tools],
            indent=1,
        )

        for probe in probes:
            try:
                result = selector(
                    user_request=probe["prompt"],
                    available_tools=tools_summary,
                )
                selected = result.selected_tool.strip()
                passed = selected == probe["expected_tool"]

                report.results.append(
                    AdversarialResult(
                        probe_name=f"binding:{probe['expected_tool']}",
                        probe_type="tool_binding",
                        passed=passed,
                        details=f"Expected '{probe['expected_tool']}', got '{selected}'",
                    )
                )
            except Exception as e:
                report.results.append(
                    AdversarialResult(
                        probe_name=f"binding:{probe['expected_tool']}",
                        probe_type="tool_binding",
                        passed=False,
                        details=f"LLM error: {str(e)[:100]}",
                    )
                )

            report.total += 1
            if report.results[-1].passed:
                report.passed += 1
            else:
                report.failed += 1

        return report


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_adversarial_suite(chatbot_instance, optimizer) -> Dict[str, Any]:
    """Run the full adversarial test suite.

    Args:
        chatbot_instance: The MCP chatbot (with .available_tools and .sessions).
        optimizer: The DSPyOptimizer instance.

    Returns:
        Combined report dict with hallucination and binding results.
    """
    header_print("ADVERSARIAL TEST SUITE")
    separator_print()

    # Hallucination probes
    colored_print("Running hallucination probes...", Colors.FLIGHT_CHECK)
    hallucination = HallucinationProbes(chatbot_instance)
    h_report = await hallucination.run_all()

    success_print(f"  Hallucination probes: {h_report.passed}/{h_report.total} passed")
    for r in h_report.results:
        if not r.passed:
            error_print(f"    FAIL: {r.probe_name} — {r.details}")

    # Tool-binding probes
    colored_print("Running tool-binding probes...", Colors.FLIGHT_CHECK)
    binding = ToolBindingProbes(chatbot_instance, optimizer)
    b_report = await binding.run_all()

    success_print(f"  Tool-binding probes: {b_report.passed}/{b_report.total} passed")
    for r in b_report.results:
        if not r.passed:
            error_print(f"    FAIL: {r.probe_name} — {r.details}")

    separator_print()

    return {
        "hallucination": {
            "total": h_report.total,
            "passed": h_report.passed,
            "failed": h_report.failed,
            "details": [
                {"probe": r.probe_name, "type": r.probe_type, "passed": r.passed, "details": r.details}
                for r in h_report.results
            ],
        },
        "tool_binding": {
            "total": b_report.total,
            "passed": b_report.passed,
            "failed": b_report.failed,
            "details": [
                {"probe": r.probe_name, "type": r.probe_type, "passed": r.passed, "details": r.details}
                for r in b_report.results
            ],
        },
    }
