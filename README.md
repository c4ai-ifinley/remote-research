# MCP Flight Checker

A universal testing and validation system for Model Context Protocol (MCP) tools that automatically generates test cases from tool schemas and optimizes prompts using DSPy.

## Overview

The Flight Checker automatically tests MCP tools by analyzing their schemas to generate appropriate test cases, executing them safely, and optimizing failed tests through intelligent prompt engineering. It works with any MCP-compliant tools without requiring tool-specific configuration.

## Features

- **Universal Compatibility**: Works with any MCP tool using only schema information
- **Automatic Test Generation**: Creates comprehensive test suites from MCP tool schemas
- **Safe Parameter Handling**: Generates context-aware test values without affecting real data
- **DSPy Optimization**: Improves failed test prompts using AI-powered optimization
- **Interactive Testing**: Optional manual context resolution for complex scenarios
- **MCP Sampling Support**: Demonstrates advanced MCP capabilities with AI-powered analysis

## Quick Start

1. Ensure your MCP servers are configured in `server_config.json`
2. Run the chatbot: `python mcp_chatbot.py`
3. The system will automatically generate and execute test cases for all connected tools
4. Use `/flight-check` to run tests manually or `/flight-check-interactive` for guided testing

## Architecture

### Core Components

- **TestGenerator**: Analyzes MCP schemas to create test cases (basic functionality, parameter validation, error handling)
- **FlightChecker**: Executes tests against live MCP tools and validates responses
- **DSPyOptimizer**: Uses schema-based prompt optimization for failed tests
- **Safe Context Handling**: Generates appropriate test values for files, URLs, and IDs

### Test Case Types

1. **Basic Functionality**: Tests core tool operation with generated parameters
2. **Parameter Validation**: Validates tool parameter handling and requirements
3. **Error Handling**: Tests tool behavior with edge cases and invalid inputs

## Configuration

The system automatically creates configuration files:

- `test_cases.json`: Stores generated test cases and optimization history
- `learned_tests.json`: Tracks successful test patterns for reuse

## Safety Features

- Never uses real files, credentials, or production data for testing
- Generates clearly marked test values with timestamps
- Validates all tool responses against schema-derived success criteria
- Provides detailed context requirements for manual resolution when needed

## MCP Sampling Demo

Includes enhanced research server demonstrating MCP sampling capabilities:

- `analyze_research_trends()`: AI-powered research landscape analysis
- `compare_research_topics()`: Comparative analysis between research domains

## Integration

The Flight Checker integrates seamlessly with existing MCP chatbot systems, running automatically at startup and providing manual testing capabilities through chat commands.