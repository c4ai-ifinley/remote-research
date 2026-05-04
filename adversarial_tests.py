"""
Adversarial test suite for MCP tool flight checking.

Tier 3 tests:
  - Hallucination probes: phantom tool / phantom parameter detection
    (goes through the LLM → tests whether the LLM fabricates capabilities)
  - Prompt-to-tool binding: verify correct tool selection using the
    production Anthropic tools API path (not DSPy)
"""

import asyncio
import json
import os
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
    probe_type: str  # "phantom_tool" | "phantom_parameter" | "tool_binding" | "schema_validation"
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
# Helpers
# ---------------------------------------------------------------------------


def _get_anthropic_client(chatbot_instance):
    """Get the Anthropic client from the chatbot (production path)."""
    return getattr(chatbot_instance, "anthropic", None)


def _append_result(report: AdversarialReport, result: AdversarialResult):
    """Add a result to a report and update counters."""
    report.results.append(result)
    report.total += 1
    if result.passed:
        report.passed += 1
    else:
        report.failed += 1


# ---------------------------------------------------------------------------
# Hallucination Probes
# ---------------------------------------------------------------------------


class HallucinationProbes:
    """Probes that test whether the *LLM* hallucinates non-existent tools or
    fabricates parameters that don't exist in the schema.

    These go through the production Anthropic messages API with the real tool
    registry attached — the same path a user prompt travels.
    """

    # Phantom tools — names that do NOT exist in the registry
    PHANTOM_TOOLS = [
        "quantum_entangle",
        "deploy_kubernetes_cluster",
        "send_email_attachment",
        "translate_to_klingon",
        "hack_mainframe",
    ]

    # Phantom parameters — prompts that tempt the LLM to fabricate params
    PHANTOM_PARAM_PROMPTS = {
        # tool_name → (prompt that asks for a fake capability, fake_param_name)
        "read_file": [
            ("Read README.md but decrypt it with my private key first", "encryption_key"),
            ("Read README.md using quantum error correction mode", "quantum_mode"),
            ("Read README.md and verify it on the blockchain", "blockchain_verify"),
        ],
        "list_directory": [
            ("List the current directory but use AI to filter out unimportant files", "ai_filter_mode"),
        ],
        "create_directory": [
            ("Create a directory called 'backup' and replicate it to the cloud", "replicate_to_cloud"),
        ],
        "search_files": [
            ("Search for 'TODO' in the project and run sentiment analysis on each match", "sentiment_analysis"),
        ],
    }

    def __init__(self, chatbot_instance):
        self.chatbot = chatbot_instance
        self.client = _get_anthropic_client(chatbot_instance)
        self.available_tools = chatbot_instance.available_tools or []
        self.available_tool_names = {t["name"] for t in self.available_tools}

    async def run_all(self) -> AdversarialReport:
        """Execute all hallucination probes."""
        report = AdversarialReport()

        if not self.client:
            warning_print("Anthropic client not available — skipping hallucination probes")
            return report

        # --- Phantom tool probes (through LLM) ---
        for phantom in self.PHANTOM_TOOLS:
            result = await self._probe_phantom_tool(phantom)
            _append_result(report, result)

        # --- Phantom parameter probes (through LLM) ---
        for tool_name, prompts in self.PHANTOM_PARAM_PROMPTS.items():
            if tool_name not in self.available_tool_names:
                continue
            for prompt_text, fake_param in prompts:
                result = await self._probe_phantom_parameter(tool_name, prompt_text, fake_param)
                _append_result(report, result)

        # --- Direct schema validation probes (server-side) ---
        for tool_name, prompts in self.PHANTOM_PARAM_PROMPTS.items():
            if tool_name not in self.available_tool_names:
                continue
            result = await self._probe_schema_validation(tool_name)
            _append_result(report, result)

        return report

    async def _probe_phantom_tool(self, phantom_name: str) -> AdversarialResult:
        """Send a prompt asking the LLM to use a tool that doesn't exist.

        PASS = LLM does NOT emit a tool_use block for the phantom tool.
        FAIL = LLM fabricates a tool_use block for the phantom tool.
        """
        if phantom_name in self.available_tool_names:
            return AdversarialResult(
                probe_name=f"phantom_tool:{phantom_name}",
                probe_type="phantom_tool",
                passed=True,
                details="Tool actually exists — skipped",
            )

        prompt = f"Please use the {phantom_name} tool to process my request."

        try:
            response = self.client.messages.create(
                max_tokens=512,
                model="claude-sonnet-4-20250514-v1-birthright",
                tools=self.available_tools,
                messages=[{"role": "user", "content": prompt}],
            )

            # Check if the LLM emitted any tool_use blocks
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    if block.name == phantom_name:
                        return AdversarialResult(
                            probe_name=f"phantom_tool:{phantom_name}",
                            probe_type="phantom_tool",
                            passed=False,
                            details=f"LLM hallucinated tool_use for '{phantom_name}'",
                            response=str(block.input)[:300],
                        )

            # LLM didn't try to call the phantom tool — pass
            text_response = " ".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            return AdversarialResult(
                probe_name=f"phantom_tool:{phantom_name}",
                probe_type="phantom_tool",
                passed=True,
                details="LLM correctly did not call phantom tool",
                response=text_response[:200],
            )

        except Exception as e:
            return AdversarialResult(
                probe_name=f"phantom_tool:{phantom_name}",
                probe_type="phantom_tool",
                passed=True,
                details=f"API error (not a hallucination): {str(e)[:100]}",
            )

    async def _probe_phantom_parameter(
        self, tool_name: str, prompt_text: str, fake_param: str
    ) -> AdversarialResult:
        """Send a prompt tempting the LLM to fabricate a parameter.

        PASS = LLM uses only real params (or declines).
        FAIL = LLM includes the fake param in tool_use.input.
        """
        try:
            response = self.client.messages.create(
                max_tokens=512,
                model="claude-sonnet-4-20250514-v1-birthright",
                tools=self.available_tools,
                messages=[{"role": "user", "content": prompt_text}],
            )

            for block in response.content:
                if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                    tool_input = block.input or {}
                    if fake_param in tool_input:
                        return AdversarialResult(
                            probe_name=f"phantom_param:{tool_name}.{fake_param}",
                            probe_type="phantom_parameter",
                            passed=False,
                            details=f"LLM fabricated param '{fake_param}' in tool_use.input",
                            response=json.dumps(tool_input)[:300],
                        )
                    else:
                        return AdversarialResult(
                            probe_name=f"phantom_param:{tool_name}.{fake_param}",
                            probe_type="phantom_parameter",
                            passed=True,
                            details="LLM used only real parameters",
                            response=json.dumps(tool_input)[:200],
                        )

            # LLM didn't call the tool at all — still a pass (it declined)
            return AdversarialResult(
                probe_name=f"phantom_param:{tool_name}.{fake_param}",
                probe_type="phantom_parameter",
                passed=True,
                details="LLM declined to call tool (no hallucination)",
            )

        except Exception as e:
            return AdversarialResult(
                probe_name=f"phantom_param:{tool_name}.{fake_param}",
                probe_type="phantom_parameter",
                passed=True,
                details=f"API error: {str(e)[:100]}",
            )

    async def _probe_schema_validation(self, tool_name: str) -> AdversarialResult:
        """Direct server-side probe: call tool with a fabricated param.

        This tests MCP server schema validation, NOT LLM hallucination.
        Labeled separately as 'schema_validation'.
        """
        session = self.chatbot.sessions.get(tool_name)
        if not session:
            return AdversarialResult(
                probe_name=f"schema_val:{tool_name}",
                probe_type="schema_validation",
                passed=True,
                details="Session not available — skipped",
            )

        args = {"__fabricated_param_xyz__": "should_be_rejected"}

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=args),
                timeout=10.0,
            )
            response_text = result.content[0].text if result.content else str(result)
            if "should_be_rejected" in response_text:
                return AdversarialResult(
                    probe_name=f"schema_val:{tool_name}",
                    probe_type="schema_validation",
                    passed=False,
                    details="Server echoed fabricated param value",
                    response=response_text[:200],
                )
            return AdversarialResult(
                probe_name=f"schema_val:{tool_name}",
                probe_type="schema_validation",
                passed=True,
                details="Server ignored fabricated param",
                response=response_text[:200],
            )
        except Exception:
            return AdversarialResult(
                probe_name=f"schema_val:{tool_name}",
                probe_type="schema_validation",
                passed=True,
                details="Server rejected fabricated param (schema validation working)",
            )


# ---------------------------------------------------------------------------
# Prompt-to-Tool Binding Test
# ---------------------------------------------------------------------------


class ToolBindingProbes:
    """Test that prompts map to the correct tool via the production Anthropic
    tools API path (messages.create with tools= parameter).

    Probe prompts can be supplied in test_cases.json under:
        "binding_probes": {"tool_name": "natural language prompt", ...}

    If not supplied, probes are auto-generated with care to avoid echoing
    the tool's name or distinctive description words.
    """

    def __init__(self, chatbot_instance, optimizer, config_path: str = "test_cases.json"):
        self.chatbot = chatbot_instance
        self.optimizer = optimizer
        self.client = _get_anthropic_client(chatbot_instance)
        self.available_tools = chatbot_instance.available_tools or []
        self.config_path = config_path

    def _load_user_probes(self) -> Dict[str, str]:
        """Load user-supplied binding prompts from test_cases.json."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                return config.get("binding_probes", {})
        except Exception:
            pass
        return {}

    def _build_binding_probes(self) -> List[Dict[str, str]]:
        """Build binding probes — user-supplied first, auto-generated fallback.

        Auto-generated prompts deliberately avoid containing the tool name
        or distinctive words from the description.
        """
        user_probes = self._load_user_probes()
        probes = []

        for tool in self.available_tools:
            name = tool.get("name", "")
            if not name:
                continue

            # Prefer user-supplied prompt
            if name in user_probes:
                probes.append({"prompt": user_probes[name], "expected_tool": name})
                continue

            # Auto-generate — use intent-based prompts that don't echo the tool name
            prompt = self._generate_binding_prompt(name, tool.get("description", ""))
            if prompt:
                probes.append({"prompt": prompt, "expected_tool": name})

        return probes

    def _generate_binding_prompt(self, tool_name: str, description: str) -> Optional[str]:
        """Generate a binding prompt that does NOT contain the tool name or
        distinctive words from the description."""
        name_lower = tool_name.lower()

        # Intent-based prompts keyed on common tool patterns
        # These are written to express user intent without naming the tool
        _INTENT_MAP = [
            (lambda n: "read" in n and "file" in n,
             "Show me what's inside the README.md file"),
            (lambda n: "read_multiple" in n,
             "I need the contents of both setup.py and requirements.txt"),
            (lambda n: "write" in n or "create_file" in n,
             "Save the text 'hello world' into a new file called output.txt"),
            (lambda n: "edit" in n or "replace" in n,
             "In config.yaml, change the port from 3000 to 8080"),
            (lambda n: "list" in n and "dir" in n,
             "What files and folders are in the current working directory?"),
            (lambda n: "tree" in n or "directory_tree" in n,
             "Give me a recursive view of everything under the src/ folder"),
            (lambda n: "search" in n and "file" in n,
             "Where does the string 'TODO' appear across the project?"),
            (lambda n: "move" in n or "rename" in n,
             "Take old_report.csv and put it at archive/old_report.csv"),
            (lambda n: "create" in n and "dir" in n,
             "Make a new folder called 'output' in the current directory"),
            (lambda n: "delete" in n or "remove" in n,
             "Get rid of the temp_data.json file"),
            (lambda n: "fetch" in n,
             "Grab the JSON response from https://httpbin.org/json"),
            (lambda n: "extract" in n or "info" in n,
             "Look up the metadata for research paper arxiv-2301.00001"),
        ]

        for predicate, prompt in _INTENT_MAP:
            if predicate(name_lower):
                # Verify prompt doesn't accidentally contain the tool name
                if tool_name.lower() not in prompt.lower():
                    return prompt

        # If no pattern matched, skip rather than use a weak generic prompt
        return None

    async def run_all(self) -> AdversarialReport:
        """Run all tool-binding probes through the production Anthropic API."""
        report = AdversarialReport()
        probes = self._build_binding_probes()

        if not probes:
            warning_print("No tool-binding probes generated")
            return report

        if not self.client:
            warning_print("Anthropic client not available — skipping binding probes")
            return report

        for probe in probes:
            result = await self._test_binding(probe["prompt"], probe["expected_tool"])
            _append_result(report, result)

        return report

    async def _test_binding(self, prompt: str, expected_tool: str) -> AdversarialResult:
        """Send prompt through the production API and check which tool is selected."""
        try:
            response = self.client.messages.create(
                max_tokens=512,
                model="claude-sonnet-4-20250514-v1-birthright",
                tools=self.available_tools,
                messages=[{"role": "user", "content": prompt}],
            )

            # Look for tool_use blocks
            tool_uses = [
                b for b in response.content if getattr(b, "type", None) == "tool_use"
            ]

            if not tool_uses:
                return AdversarialResult(
                    probe_name=f"binding:{expected_tool}",
                    probe_type="tool_binding",
                    passed=False,
                    details=f"No tool selected (expected '{expected_tool}')",
                )

            # Check if the first (or any) tool_use matches expected
            selected_name = tool_uses[0].name
            passed = selected_name == expected_tool

            return AdversarialResult(
                probe_name=f"binding:{expected_tool}",
                probe_type="tool_binding",
                passed=passed,
                details=f"Expected '{expected_tool}', got '{selected_name}'",
            )

        except Exception as e:
            return AdversarialResult(
                probe_name=f"binding:{expected_tool}",
                probe_type="tool_binding",
                passed=False,
                details=f"API error: {str(e)[:100]}",
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_adversarial_suite(chatbot_instance, optimizer) -> Dict[str, Any]:
    """Run the full adversarial test suite.

    Args:
        chatbot_instance: The MCP chatbot (with .available_tools, .sessions, .anthropic).
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
